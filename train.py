# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

from scipy.spatial.transform import Rotation as R
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import numpy as np
import wandb
import json
import os
import random

# load outside of main function to avoid reloading on every iteration.
DATASET_PTH = Path('../datasets/memmap_datasets_xformers/ops_swissprot')
meta = json.load(open(DATASET_PTH / 'meta.test.json', 'r'))
POS_OFFSET = np.prod(meta['pos_shape'])
T_OFFSET = np.prod(meta['T7_shape'])
dtype = meta['dtype']
assert meta['dtype'] == 'float32', f"Expected dtype to be float32, got {meta['dtype']}"

TRAIN_STARTS = np.load(DATASET_PTH / f'starts.train.npy') # takes 100 us
TEST_STARTS = np.load(DATASET_PTH / f'starts.test.npy') # takes 100 us

# gram schmdits to identity
BASE = torch.tensor([[-1, 0, 0], [0,0,0], [0,1,0], [0,0,0]]).unsqueeze(0).unsqueeze(0)

def load_from_mmap(split, batch_size, unit='nm', block_size=512, device='cuda:0'):
    assert unit in ['nm', 'angstrom']

    pos_mmap = np.memmap(DATASET_PTH / f'positions.{split}.dat', dtype=np.float32)
    pos_inv_mmap = np.memmap(DATASET_PTH / f'positions_inv.{split}.dat', dtype=np.float32)
    sse_mmap = np.memmap(DATASET_PTH / f'sse.{split}.dat', dtype=np.int64)
    protid_mmap = np.memmap(DATASET_PTH / f'prot_ids.{split}.dat', dtype=np.int64)

    max_ix = len(sse_mmap)
    start_ix = [random.randint(0, max_ix - block_size) for _ in range(batch_size)]

    pos = torch.empty(batch_size, block_size, *meta['pos_shape'], dtype=torch.float32)
    pos_inv = torch.empty(batch_size, block_size, *meta['pos_shape'], dtype=torch.float32)
    sse = torch.empty(batch_size, block_size, dtype=torch.int64)
    protid = torch.empty(batch_size, block_size, dtype=torch.int64)

    for m, ix in enumerate(start_ix):
        pos[m] = torch.from_numpy(pos_mmap[ix*POS_OFFSET:(ix+block_size)*POS_OFFSET].reshape(block_size, *meta['pos_shape']))
        pos_inv[m] = torch.from_numpy(pos_inv_mmap[ix*POS_OFFSET:(ix+block_size)*POS_OFFSET].reshape(block_size, *meta['pos_shape']))
        sse[m] = torch.from_numpy(sse_mmap[ix:ix+block_size].reshape(block_size,))
        protid[m] = torch.from_numpy(protid_mmap[ix:ix+block_size].reshape(block_size,))
    
    if unit == 'nm':
        pos = pos / 10.0
        pos_inv = pos_inv / 10.0
    
    # if 'cuda' in device:
    pos, pos_inv, sse, protid = (
        pos.pin_memory().to(device, non_blocking=True),
        pos_inv.pin_memory().to(device, non_blocking=True),
        sse.pin_memory().to(device, non_blocking=True),
        protid.pin_memory().to(device, non_blocking=True)
    )
    # else:
    #     pos, pos_inv, sse, protid = pos.to(device), pos_inv.to(device), sse.to(device), protid.to(device)
    
    # from sse to tokens, bos eos latent
    sse = torch.where(sse < 2, sse, 2)

    return pos, pos_inv, sse, protid #, start_ix

def get_dit_batch(split, batch_size, block_size, device, ignore_oxygen=False):
    """
    Loads a batch of data and performs DiT-specific preprocessing.
    1. Identifies and masks out invalid/placeholder coordinates.
    2. Optionally removes oxygen atoms.
    3. Performs per-protein centering of mass on valid residues.
    4. Applies a single random rotation to all coordinates in the batch item.
    5. Creates attention and diffusion masks that respect protein boundaries and invalid residues.
    """
    # Load raw data using your existing function
    pos, _, _, protid = load_from_mmap(split, batch_size, unit='nm', block_size=block_size, device=device)
    # pos shape: (B, L, 4, 3), protid shape: (B, L)
    
    # 1. Create a mask to identify valid residues based on coordinate values
    # A value like 1e4 is a safe threshold since real coordinates in nm are small.
    invalid_coord_mask = torch.any(torch.abs(pos) > 1e4, dim=(-1, -2)) # Shape: (B, L)
    
    # A residue is valid if it does NOT have invalid coordinates.
    valid_residue_mask = ~invalid_coord_mask # Shape: (B, L)

    # 2. Optionally remove oxygen atom
    if ignore_oxygen:
        pos = pos[:, :, :3, :]  # Keep N, CA, C -> shape (B, L, 3, 3)

    # Set invalid positions to zero to prevent them from affecting CoM
    pos[~valid_residue_mask] = 0.0
    
    # 3. Perform per-protein centering on valid residues
    centered_pos = torch.zeros_like(pos)
    for i in range(pos.shape[0]):
        # Get unique protein IDs present in this sequence
        unique_ids = torch.unique(protid[i])
        
        for uid in unique_ids:
            # Mask for the current protein fragment
            protid_mask = (protid[i] == uid)
            # Mask for the valid residues WITHIN this fragment
            valid_segment_mask = protid_mask & valid_residue_mask[i]
            
            if not torch.any(valid_segment_mask):
                continue
            
            segment_coords = pos[i, valid_segment_mask]
            
            # Calculate CoM only on the valid residues of this fragment
            num_valid_atoms = segment_coords.numel() / 3 # segment_coords is (num_valid_res, num_atoms, 3)
            if num_valid_atoms > 0:
                com = torch.sum(segment_coords.view(-1, 3), dim=0) / num_valid_atoms
                # Apply centering to all residues belonging to this protein ID.
                # Invalid residues were already zeroed, so this moves them to -com.
                centered_pos[i, protid_mask] = pos[i, protid_mask] - com
            
    pos = centered_pos

    # 4. Apply a single random rotation
    rot_matrix = torch.from_numpy(R.random().as_matrix()).float().to(device)
    pos = torch.einsum('ij,blaj->blai', rot_matrix, pos)

    # 5. Create Attention Mask, incorporating the valid_residue_mask
    # Attention is allowed between two residues if they have the same protid AND both are valid
    same_protid_mask = (protid.unsqueeze(1) == protid.unsqueeze(2))
    # A valid pair requires both residues to be valid
    valid_pair_mask = same_protid_mask & valid_residue_mask.unsqueeze(1) & valid_residue_mask.unsqueeze(2)

    attn_mask = torch.zeros_like(same_protid_mask, dtype=torch.float, device=device)
    attn_mask.masked_fill_(~valid_pair_mask, -float('inf'))

    # 6. Flatten coordinates for the model input
    x = pos.reshape(batch_size, block_size, -1)

    # 7. Create Diffusion Mask, incorporating the valid_residue_mask
    num_atoms = 3 if ignore_oxygen else 4
    num_features = num_atoms * 3
    
    diffusion_mask = torch.ones(batch_size, block_size, num_features, device=device)
    diffusion_mask[:, :, 3:6] = 0 # Freeze CA atoms

    # Mask out invalid residues entirely from the diffusion process and loss
    # Reshape valid_residue_mask from (B, L) to (B, L, 1) and broadcast to (B, L, F)
    diffusion_mask = diffusion_mask * valid_residue_mask.unsqueeze(-1)
    
    return x, attn_mask, diffusion_mask

def analyze_noise_schedule(diffusion_obj, max_timestep, unit_of_input_data='nm'):
    """
    Analyzes and prints the physical noise level at a given timestep.
    
    Args:
        diffusion_obj: The created diffusion object.
        max_timestep: The maximum timestep being trained on
        unit_of_input_data: The physical unit of the coordinate data ('nm' or 'angstrom').
    """
    # Access the base diffusion schedule, as SpacedDiffusion wraps it
    if hasattr(diffusion_obj, 'original_num_steps'):
        base_diffusion = diffusion_obj.base_diffusion
    else:
        base_diffusion = diffusion_obj
    
    # Calculate the standard deviation of the noise added at timestep `t`
    # This is sqrt(1 - alpha_bar_t), a unitless scaling factor.
    alphas_cumprod = base_diffusion.alphas_cumprod
    std_dev_scaling_factors = np.sqrt(1.0 - alphas_cumprod)
    
    # Get the scaling factor at your specific max_timestep (0-indexed)
    t_index = max_timestep - 1
    if not (0 <= t_index < len(std_dev_scaling_factors)):
        print(f"Error: max_timestep {max_timestep} is out of bounds for schedule of length {len(std_dev_scaling_factors)}.")
        return

    std_dev_at_t_max = std_dev_scaling_factors[t_index]
    
    # The 3-sigma rule gives a practical "max" shift
    max_shift_at_t_max = 3 * std_dev_at_t_max
    
    print("\n" + "="*50)
    print("           DiT Refinement Budget Analysis")
    print("="*50)
    print(f"Max Timestep for Training:       {max_timestep}")
    print(f"Noise StdDev at this timestep:   {std_dev_at_t_max:.4f} {unit_of_input_data}")
    print(f"Practical Max Shift (3σ):        {max_shift_at_t_max:.4f} {unit_of_input_data}")
    print("="*50 + "\n")

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT refiner model on protein backbones.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # ==========================================================================
    # DDP SETUP
    # ==========================================================================
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    world_size = dist.get_world_size()
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    # ==========================================================================
    # WANDB & EXPERIMENT SETUP
    # ==========================================================================
    if rank == 0:
        # Create a unique experiment directory
        model_string_name = args.model.replace("/", "-")
        run_name = f"{model_string_name}-len{args.block_size}-bs{args.global_batch_size}"
        if args.use_wandb:
            wandb.init(
                project="protein-dit-refiner",
                name=run_name,
                config=args,
            )
            # Use wandb run ID for checkpoint directory to sync with the run
            checkpoint_dir = f"{args.results_dir}/{wandb.run.id}/checkpoints"
        else:
            # Fallback for local-only runs
            checkpoint_dir = f"{args.results_dir}/{run_name}/checkpoints"
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to {checkpoint_dir}")
    dist.barrier() # Ensure all processes see the created directory

    # ==========================================================================
    # MODEL, DIFFUSION, and OPTIMIZER SETUP
    # ==========================================================================
    # 1. Determine input features
    num_atoms = 3 if args.ignore_oxygen else 4
    input_features = num_atoms * 3

    # 2. Create the DiT model
    dit_model = DiT_models[args.model](
        max_len=args.block_size,
        in_features=input_features,
        num_classes=1,  # Unconditional training
    ).to(device)

    # 3. Create the EMA model for high-quality inference
    ema = deepcopy(dit_model).to(device)
    requires_grad(ema, False)
    update_ema(ema, dit_model, decay=0) # Initialize EMA with model weights

    # 4. Create the diffusion process helper
    diffusion = create_diffusion(timestep_respacing="")

    if rank == 0:
        analyze_noise_schedule(diffusion, args.max_timestep, unit_of_input_data='nm')

    # 5. Wrap model in DDP
    dit_model = DDP(dit_model, device_ids=[rank])
    raw_dit_model = dit_model.module
    print(f"Rank {rank} | DiT Parameters: {sum(p.numel() for p in raw_dit_model.parameters()):,}")

    # 6. Setup Optimizer
    optimizer = torch.optim.AdamW(
        dit_model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )

    # ==========================================================================
    # TRAINING LOOP
    # ==========================================================================
    print("Starting training...")
    start_time = time()
    local_batch_size = args.global_batch_size // world_size

    # Loop for a fixed number of iterations
    for train_steps in range(1, args.train_iters + 1):
        raw_dit_model.train()

        # Get a batch of preprocessed data for DiT
        x, attn_mask, diffusion_mask = get_dit_batch(
            'train', 
            batch_size=local_batch_size, 
            block_size=args.block_size, 
            device=device,
            ignore_oxygen=args.ignore_oxygen
        )

        # Sample timesteps from the low-noise regime
        t = torch.randint(0, args.max_timestep, (x.shape[0],), device=device)
        
        # Prepare model arguments, including the attention mask
        model_kwargs = dict(
            y=torch.zeros(x.shape[0], dtype=torch.long, device=device), 
            attn_mask=attn_mask
        )
        
        # Calculate loss
        loss_dict = diffusion.training_losses(dit_model, x, t, model_kwargs, mask=diffusion_mask)
        loss = loss_dict["loss"].mean()
        
        # Backward pass and optimization step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(dit_model.parameters(), 1.0)
        optimizer.step()

        # Update EMA model
        update_ema(ema, raw_dit_model, decay=args.ema_decay)

        # ==========================================================================
        # LOGGING AND CHECKPOINTING
        # ==========================================================================
        if train_steps % args.log_every == 0:
            # Reduce loss for logging
            reduced_loss = loss.detach().clone()
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
            avg_loss = reduced_loss.item() / world_size

            # Measure training speed
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = args.log_every / (end_time - start_time)
            
            if rank == 0:
                print(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if args.use_wandb:
                    wandb.log({
                        "iteration": train_steps,
                        "train_loss": avg_loss,
                        "steps_per_sec": steps_per_sec,
                        "lr": optimizer.param_groups[0]['lr']
                    })
            start_time = time() # Reset timer for next log interval

        if train_steps % args.ckpt_every == 0 and train_steps > 0:
            if rank == 0:
                checkpoint = {
                    "model": raw_dit_model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": optimizer.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            dist.barrier()

    print("Training finished.")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Core Model & Data Args
    parser.add_argument("--model", type=str, default="DiT-XL-Protein", choices=list(DiT_models.keys()))
    parser.add_argument("--block-size", type=int, default=256, help="Sequence length to train on.")
    parser.add_argument("--ignore-oxygen", action='store_true', help="If set, removes oxygen atom from input.")
    
    # Diffusion Args
    parser.add_argument("--max-timestep", type=int, default=25, help="Max noise step to train on (e.g., 25 out of 1000).")
    
    # Training Args
    parser.add_argument("--train-iters", type=int, default=400_000, help="Total number of training iterations.")
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999, help="Decay for the EMA model.")

    # System & Logging Args
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=20_000)
    parser.add_argument("--use-wandb", action='store_true', help="Enable wandb logging.")
    
    args = parser.parse_args()
    main(args)

"""
torchrun --standalone --nproc_per_node=4 train.py \
    --model "DiT-XL-Protein" \
    --block-size 256 \
    --global-batch-size 64 \
    --train-iters 400000 \
    --lr 1e-4 \
    --log-every 100 \
    --ckpt-every 20000 \
    --ignore-oxygen \
    --use-wandb
"""
