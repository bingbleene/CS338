"""
================================================================================
DETAILED CODE EXPLANATION - CRIS.pytorch vs CRIS_SNN
================================================================================

This file provides line-by-line explanation of the main training code.
"""

# ============================================================================
# 1. TRAIN.PY - MAIN TRAINING SCRIPT
# ============================================================================

import argparse                          # Parse command line arguments
import datetime                          # For timing
import os                                # File system operations
import shutil                            # For copying files
import sys                               # System operations
import time                              # For timing
import warnings                          # Show/hide warnings
from functools import partial            # Create partial functions

import cv2                               # OpenCV for image processing
import torch                             # PyTorch library
import torch.cuda.amp as amp             # Automatic Mixed Precision
import torch.distributed as dist         # Multi-GPU communication
import torch.multiprocessing as mp       # Multi-process spawning
import torch.nn as nn                    # Neural network modules
import torch.nn.parallel                 # Parallel model wrapper
import torch.optim                       # Optimizers (Adam, SGD, etc)
import torch.utils.data as data          # Data loading utilities
from loguru import logger                # Logging library
from torch.optim.lr_scheduler import MultiStepLR  # Learning rate scheduler

import utils.config as config            # Config loading (YAML)
import wandb                             # Weights & Biases (experiment tracking)
from utils.dataset import RefDataset     # Custom dataset class
from engine.engine import train, validate  # Training/validation functions
from model import build_segmenter        # Model builder
from utils.misc import (init_random_seed, set_random_seed, setup_logger,
                        worker_init_fn)  # Utility functions

warnings.filterwarnings("ignore")        # Suppress warnings
cv2.setNumThreads(0)                     # Disable OpenCV threading


# ============================================================================
# 2. ARGUMENT PARSER
# ============================================================================

def get_parser():
    """Parse and load configuration from command line and YAML file"""
    
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    
    # Accept config file path
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    
    # Allow command-line overrides (e.g., --opts TRAIN.batch_size 32)
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')

    args = parser.parse_args()
    assert args.config is not None  # Config file must be provided
    
    # Load YAML config file
    cfg = config.load_cfg_from_cfg_file(args.config)
    
    # Merge command-line overrides into config
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    
    return cfg


# ============================================================================
# 3. MAIN FUNCTION - MULTI-GPU SETUP
# ============================================================================

@logger.catch                           # Catch exceptions with logging
def main():
    """Initialize distributed training"""
    
    args = get_parser()                 # Load config
    args.manual_seed = init_random_seed(args.manual_seed)  # Set random seed
    set_random_seed(args.manual_seed, deterministic=False)  # Reproducibility
    
    # Count available GPUs
    args.ngpus_per_node = torch.cuda.device_count()
    
    # Calculate total world size (number of processes)
    args.world_size = args.ngpus_per_node * args.world_size
    
    # Spawn training process on each GPU
    mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args, ))


# ============================================================================
# 4. MAIN WORKER - PER-GPU TRAINING LOGIC
# ============================================================================

def main_worker(gpu, args):
    """Training logic for each GPU"""
    
    # === SETUP ===
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    
    # Set GPU ID for this process
    args.gpu = gpu
    args.rank = args.rank * args.ngpus_per_node + gpu  # Global rank
    torch.cuda.set_device(args.gpu)                     # Use specific GPU
    
    # === LOGGER SETUP ===
    setup_logger(args.output_dir,
                 distributed_rank=args.gpu,
                 filename="train.log",
                 mode="a")

    # === DISTRIBUTED SETUP ===
    dist.init_process_group(backend=args.dist_backend,      # 'nccl' for GPU
                            init_method=args.dist_url,       # Master URL
                            world_size=args.world_size,      # Total GPUs
                            rank=args.rank)                  # This GPU rank
    
    # === WANDB SETUP (only on rank 0) ===
    if args.rank == 0:
        wandb.init(job_type="training",
                   mode="online",
                   config=args,                # Log all hyperparameters
                   project="CRIS",
                   name=args.exp_name,
                   tags=[args.dataset, args.clip_pretrain])
    dist.barrier()  # Wait for rank 0 to finish WandB init
    
    # === BUILD MODEL ===
    model, param_list = build_segmenter(args)  # Create CRIS model
    logger.info(model)                          # Log model architecture
    
    # Convert batch norm to sync batch norm for distributed training
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # Wrap model for distributed training across GPUs
    model = nn.parallel.DistributedDataParallel(model.cuda(),
                                                device_ids=[args.gpu],
                                                find_unused_parameters=True)
    
    # === BUILD OPTIMIZER ===
    # Adam optimizer with different LR for backbone and head
    optimizer = torch.optim.Adam(param_list,
                                 lr=args.base_lr,
                                 weight_decay=args.weight_decay)
    
    # Learning rate scheduler: multiply by gamma at milestones
    scheduler = MultiStepLR(optimizer,
                            milestones=args.milestones,  # [35] epochs
                            gamma=args.lr_decay)         # 0.1x
    
    # Automatic Mixed Precision for faster training + less memory
    scaler = amp.GradScaler()
    
    # === BUILD DATASET ===
    # Adjust batch size for this GPU (divide by number of GPUs)
    args.batch_size = int(args.batch_size / args.ngpus_per_node)
    args.batch_size_val = int(args.batch_size_val / args.ngpus_per_node)
    args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
    
    # Load training dataset (from LMDB files)
    train_data = RefDataset(lmdb_dir=args.train_lmdb,
                            mask_dir=args.mask_root,
                            dataset=args.dataset,
                            split=args.train_split,
                            mode='train',
                            input_size=args.input_size,
                            word_length=args.word_len)
    
    # Load validation dataset
    val_data = RefDataset(lmdb_dir=args.val_lmdb,
                          mask_dir=args.mask_root,
                          dataset=args.dataset,
                          split=args.val_split,
                          mode='val',
                          input_size=args.input_size,
                          word_length=args.word_len)
    
    # === BUILD DATALOADER ===
    # Worker init function for reproducibility
    init_fn = partial(worker_init_fn,
                      num_workers=args.workers,
                      rank=args.rank,
                      seed=args.manual_seed)
    
    # Distributed sampler (each GPU gets different batches)
    train_sampler = data.distributed.DistributedSampler(train_data,
                                                        shuffle=True)
    val_sampler = data.distributed.DistributedSampler(val_data, 
                                                      shuffle=False)
    
    # DataLoader with distributed sampler
    train_loader = data.DataLoader(train_data,
                                   batch_size=args.batch_size,
                                   shuffle=False,  # Sampler handles shuffling
                                   num_workers=args.workers,
                                   pin_memory=True,  # Faster GPU transfer
                                   worker_init_fn=init_fn,
                                   sampler=train_sampler,
                                   drop_last=True)  # Discard incomplete batch
    
    val_loader = data.DataLoader(val_data,
                                 batch_size=args.batch_size_val,
                                 shuffle=False,
                                 num_workers=args.workers_val,
                                 pin_memory=True,
                                 sampler=val_sampler,
                                 drop_last=False)
    
    # === RESUME FROM CHECKPOINT (optional) ===
    best_IoU = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            # Load checkpoint (model, optimizer, scheduler states)
            checkpoint = torch.load(
                args.resume, map_location=lambda storage: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            best_IoU = checkpoint["best_iou"]
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            raise ValueError("No checkpoint found at '{}'".format(args.resume))
    
    # === TRAINING LOOP ===
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1
        
        # Set epoch for distributed sampler (changes random seed for shuffling)
        train_sampler.set_epoch(epoch_log)
        
        # Training pass
        train(train_loader, model, optimizer, scheduler, scaler, epoch_log, args)
        
        # Validation pass
        iou, prec_dict = validate(val_loader, model, epoch_log, args)
        
        # === SAVE CHECKPOINT (rank 0 only) ===
        if dist.get_rank() == 0:
            lastname = os.path.join(args.output_dir, "last_model.pth")
            torch.save(
                {
                    'epoch': epoch_log,
                    'cur_iou': iou,
                    'best_iou': best_IoU,
                    'prec': prec_dict,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, lastname)
            
            # Save best model if current IoU is better
            if iou >= best_IoU:
                best_IoU = iou
                bestname = os.path.join(args.output_dir, "best_model.pth")
                shutil.copyfile(lastname, bestname)
        
        # Update learning rate
        scheduler.step(epoch_log)
        torch.cuda.empty_cache()  # Clear GPU cache
    
    # === CLEANUP ===
    time.sleep(2)
    if dist.get_rank() == 0:
        wandb.finish()  # Finalize WandB logging
    
    # Log final results
    logger.info("* Best IoU={} * ".format(best_IoU))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('* Training time {} *'.format(total_time_str))


# ============================================================================
# 5. MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    main()  # Start distributed training
    sys.exit(0)  # Exit cleanly


# ============================================================================
# KEY CONCEPTS:
# ============================================================================

"""
1. DISTRIBUTED TRAINING:
   - mp.spawn() creates one process per GPU
   - Each process gets its own main_worker() with different GPU ID
   - Gradients are averaged across GPUs using collective communication

2. LEARNING RATE SCHEDULE:
   - MultiStepLR: multiply LR by gamma at specific epochs (milestones)
   - Example: LR=0.0001 at epoch 0-34, then 0.00001 at epoch 35+

3. MIXED PRECISION TRAINING:
   - amp.GradScaler(): automatically scales loss to prevent underflow
   - Reduces memory usage by ~50% and speeds up training by ~20%

4. DISTRIBUTED SAMPLER:
   - DistributedSampler: ensures each GPU gets different batches
   - set_epoch(epoch): changes random seed to shuffle differently each epoch
   - Prevents data leakage between GPUs

5. CHECKPOINT SAVING:
   - Save after each epoch: model weights, optimizer state, scheduler state
   - Allows resuming training from any epoch
"""


# ============================================================================
# DIFFERENCES: CRIS.pytorch vs CRIS_SNN
# ============================================================================

"""
CRIS.pytorch (Original):
├── Model building:
│   └── Uses CLIP encoder loaded from pretrained JIT file
│       - Takes CLIP.pt checkpoint
│       - Returns (C3, C4, C5) from ResNet backbone
│
└── Training loop:
    └── Standard CNN training with CLIP features

CRIS_SNN (Modified):
├── Model building:
│   └── Uses SNN encoders initialized randomly
│       - No pretrained weights (SNNVisionEncoder, SNNTextEncoder)
│       - Returns (C3, C4, C5) from SNN layers
│       - Includes num_steps parameter for temporal processing
│
└── Training loop:
    └── Same as CRIS.pytorch (no changes needed!)
        - Both use binary cross-entropy loss
        - Both use same optimizer and scheduler
        - Both save checkpoints identically

KEY DIFFERENCE IN MODEL:
- CRIS: backbone = CLIP (requires .pt file)
- CRIS-SNN: backbone = SNNVisionEncoder (no pretrained file)

DATA FORMAT:
- CRIS: image (B, 3, 416, 416) → CLIP.rate_encoding (internal)
- CRIS-SNN: image (B, 3, 416, 416) → SNNVisionEncoder.rate_encode (internal)

BOTH PRODUCE SAME OUTPUT FORMAT:
- vis: [(B, 256, 52, 52), (B, 512, 26, 26), (B, 1024, 13, 13)]
- state: (B, 1024)
"""
