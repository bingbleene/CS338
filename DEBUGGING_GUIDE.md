"""
================================================================================
DEBUGGING GUIDE & COMMON ISSUES
================================================================================

How to troubleshoot and understand common problems in CRIS-SNN training
"""

# ============================================================================
# 1. SHAPE MISMATCH ERRORS
# ============================================================================

"""
ERROR: RuntimeError: Expected input to have n channels, but got m
───────────────────────────────────────────────────────────────

Common at:
    - FPN fusion layers
    - Transformer decoder
    - Projector

DIAGNOSIS:
    Print tensor shapes at each layer:
    
    import sys
    def debug_shapes(name, tensor):
        print(f"{name}: {tensor.shape}")
        sys.stdout.flush()
    
    In forward pass:
        debug_shapes("Input image", img)  # (B, 3, 416, 416)
        vis = self.backbone(img)
        debug_shapes("Vision features C3", vis[0])  # Should be (B, 256, 52, 52)
        debug_shapes("Vision features C4", vis[1])  # Should be (B, 512, 26, 26)
        debug_shapes("Vision features C5", vis[2])  # Should be (B, 1024, 13, 13)

EXPECTED SHAPES (CRITICAL):
────────────────────────
    SNNVisionEncoder output:
        C3: (B, 256, 52, 52)   ✗ WRONG: (B, 512, 13, 13)
        C4: (B, 512, 26, 26)   ✗ WRONG: (B, 256, 26, 26)
        C5: (B, 1024, 13, 13)  ✗ WRONG: (B, 1024, 26, 26)
    
    SNNTextEncoder output:
        word_emb: (B, 77, 512)  ✗ WRONG: (B, 77, 1024)
        state: (B, 1024)        ✗ WRONG: (B, 512)
    
    FPN output:
        fq: (B, 512, 26, 26)    ✗ WRONG: (B, 256, 52, 52)
    
    Projector output:
        pred: (B, 1, 104, 104)  ✗ WRONG: (B, 1, 416, 416)

FIX STRATEGY:
─────────
    1. Add shape checks after each component:
       assert vis[0].shape == (B, 256, 52, 52), f"C3 wrong: {vis[0].shape}"
       assert vis[1].shape == (B, 512, 26, 26), f"C4 wrong: {vis[1].shape}"
       assert vis[2].shape == (B, 1024, 13, 13), f"C5 wrong: {vis[2].shape}"
    
    2. Use tensor.size() instead of tensor.shape for clarity
    
    3. Check upsampling/downsampling factors:
       416 → 208 (÷2), 52 (÷4), 26 (÷8), 13 (÷16 from 416)
    
    4. Review layer output dimensions in model code
"""

# ============================================================================
# 2. OUT OF MEMORY (OOM) ERRORS
# ============================================================================

"""
ERROR: RuntimeError: CUDA out of memory
───────────────────────────────────────

DIAGNOSIS:
    torch.cuda.empty_cache()
    torch.cuda.memory_allocated() / 1e9  # GB
    torch.cuda.max_memory_allocated() / 1e9

MEMORY BREAKDOWN:
────────────────
    Model weights: ~200 MB
    Activations (forward): ~300 MB per batch
    Gradients (backward): ~150 MB
    Optimizer state: ~100 MB
    
    Total per batch at size 64: ~1 GB
    With 4 GPUs: 250 MB per GPU

SOLUTIONS (in order):
──────────────────
    1. Reduce batch_size:
       batch_size: 64 → 32 → 16 → 8
       (will train slower, but fit in memory)
    
    2. Use gradient accumulation:
       for i, batch in enumerate(dataloader):
           outputs = model(batch)
           loss = criterion(outputs, targets)
           loss.backward()
           
           if (i+1) % accumulation_steps == 0:
               optimizer.step()
               optimizer.zero_grad()
       
       Equivalent to larger batch without needing more memory
    
    3. Reduce num_steps:
       num_steps: 10 → 5
       Faster but less accurate (fewer temporal iterations)
    
    4. Use lower precision (already enabled with AMP):
       with amp.autocast():
           outputs = model(batch)
       Uses FP16 internally (half memory)
    
    5. Delete unused tensors:
       del tensor_name
       torch.cuda.empty_cache()

EXAMPLE FIX:
────────
    cfg.batch_size = 8  # Reduced from 64
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=2,  # Reduce workers too
        pin_memory=False  # Can reduce memory
    )
"""

# ============================================================================
# 3. NaN / INFINITE LOSS
# ============================================================================

"""
ERROR: Loss = nan or inf, training fails
─────────────────────────────────────────

DIAGNOSIS:
    import torch.nn.functional as F
    
    # Check for NaN early
    for i, (img, text, target) in enumerate(train_loader):
        if i == 0:  # Check first batch
            print(f"Image min: {img.min()}, max: {img.max()}")
            print(f"Image has NaN: {(img != img).any()}")
            
            pred, _, loss = model(img, text, target)
            print(f"Loss: {loss}")
            print(f"Pred has NaN: {(pred != pred).any()}")
            print(f"Pred min: {pred.min()}, max: {pred.max()}")

ROOT CAUSES:
───────────
    1. Unnormalized input:
       ✗ WRONG: img = img / 255  (not normalized)
       ✓ CORRECT: 
           img = (img / 255 - mean) / std
           where mean=[0.48, 0.46, 0.41]
                 std=[0.27, 0.26, 0.28]
    
    2. Learning rate too high:
       ✗ WRONG: base_lr: 0.1
       ✓ CORRECT: base_lr: 0.0001
    
    3. Zero gradients being clipped:
       ✗ WRONG: max_norm: 0 (causes NaN in grad clipping)
       ✓ CORRECT: max_norm: 0.1 or None
    
    4. Division by zero in metrics:
       If union=0: iou = inter/union = x/0 = NaN
       Add epsilon: iou = inter / (union + 1e-6)
    
    5. Spike encoding issue:
       If rate encoding produces all zeros/ones
       spike = torch.rand() < intensity (can fail if not in [0,1])

FIXES:
────
    # Fix 1: Verify normalization
    assert img.min() >= -10 and img.max() <= 10
    
    # Fix 2: Lower learning rate
    base_lr: 0.0001
    
    # Fix 3: Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    
    # Fix 4: Add epsilon to metrics
    iou = inter / (union + 1e-6)
    
    # Fix 5: Ensure proper spike encoding
    assert intensity.min() >= 0 and intensity.max() <= 1
    spikes = torch.rand_like(intensity) < intensity
"""

# ============================================================================
# 4. TRAINING DIVERGENCE (Loss not decreasing)
# ============================================================================

"""
SYMPTOM: Loss stays constant or increases every iteration
────────────────────────────────────────────────────────

CHECKLIST:
────────
    ☐ Model actually training? Check model.train() is called
    ☐ Gradients computed? Check loss.backward() is called
    ☐ Weights updated? Check optimizer.step() is called
    
    ☐ Learning rate > 0? Check optimizer.param_groups[0]['lr']
    ☐ Learning rate reasonable? Should be ~0.0001, not 0.00001
    ☐ Data actually changes? Different batches each iteration
    ☐ Target mask correct format? (0 or 1, not one-hot)

DIAGNOSTICS:
────────────
    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"PROBLEM: {name} has no gradient!")
        elif (param.grad == 0).all():
            print(f"WARNING: {name} gradients all zero")
        elif (param.grad != param.grad).any():  # NaN check
            print(f"ERROR: {name} has NaN gradients")
    
    # Check if weights are changing
    old_weights = model.backbone.conv1.weight.clone()
    # ... do one training step ...
    new_weights = model.backbone.conv1.weight
    if torch.allclose(old_weights, new_weights):
        print("PROBLEM: Weights not updating!")
    
    # Check learning rate scheduler
    print(f"Current LR: {scheduler.get_last_lr()}")
    scheduler.step()
    print(f"Next LR: {scheduler.get_last_lr()}")

COMMON FIXES:
─────────────
    1. Add model.train():
        ✗ WRONG:
            pred = model(img, text)
        ✓ CORRECT:
            model.train()
            pred = model(img, text)
    
    2. Enable gradient tracking:
        ✗ WRONG:
            with torch.no_grad():
                pred = model(img, text)
        ✓ CORRECT:
            pred = model(img, text)  # Without no_grad
    
    3. Call backward and step:
        ✗ WRONG:
            loss = criterion(pred, target)
        ✓ CORRECT:
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    4. Increase learning rate:
        ✗ WRONG: base_lr: 0.00001
        ✓ CORRECT: base_lr: 0.0001 (10x higher)
    
    5. Use data correctly:
        ✗ WRONG:
            dataloader = DataLoader(dataset, shuffle=False, batch_size=1)
        ✓ CORRECT:
            dataloader = DataLoader(dataset, shuffle=True, batch_size=64)
"""

# ============================================================================
# 5. SLOW TRAINING / HIGH GPU MEMORY USAGE
# ============================================================================

"""
ISSUE: Training very slow or GPU memory constantly maxing out
────────────────────────────────────────────────────────────

PROFILING:
─────────
    # Time breakdown
    import time
    
    start = time.time()
    for i, batch in enumerate(train_loader):
        data_time = time.time()
        
        img, text, target = batch
        # Data transfer
        img = img.cuda()
        text = text.cuda()
        target = target.cuda()
        
        transfer_time = time.time()
        
        # Forward
        pred, _, loss = model(img, text, target)
        forward_time = time.time()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        back_time = time.time()
        
        # Optimizer step
        optimizer.step()
        step_time = time.time()
        
        # Summary
        print(f"Data: {data_time-start:.3f}s, "
              f"Transfer: {transfer_time-data_time:.3f}s, "
              f"Forward: {forward_time-transfer_time:.3f}s, "
              f"Backward: {back_time-forward_time:.3f}s, "
              f"Step: {step_time-back_time:.3f}s")
        
        if i >= 5:  # Just profile first few batches
            break

COMMON BOTTLENECKS:
───────────────────
    1. Data loading too slow:
       num_workers=4  # Increase from 1
       pin_memory=True  # Enable
       persistent_workers=True  # Keep workers alive
    
    2. GPU transfer overhead:
       Non_blocking=True (transfer while compute happens)
       img = img.cuda(non_blocking=True)
    
    3. num_steps too high:
       num_steps: 10 → 5 (halves computation time)
    
    4. Unnecessary model evaluations:
       model.eval()  # Only for validation
       model.train()  # For training

OPTIMIZATIONS:
──────────────
    # Faster data loading
    train_loader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,  # CPU workers
        pin_memory=True,  # Transfer to GPU faster
        persistent_workers=True  # Keep alive between epochs
    )
    
    # Overlap compute and transfer
    for img, text, target in train_loader:
        img = img.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # GPU can transfer while CPU loads next batch
        
        # Now do training
        pred, _, loss = model(img, text, target)
    
    # Reduce num_steps for faster training (for testing)
    # cfg.num_steps = 5  # 2x faster, slightly less accurate
"""

# ============================================================================
# 6. VALIDATION IoU MUCH LOWER THAN TRAINING IoU
# ============================================================================

"""
SYMPTOM: Training IoU = 0.80, Validation IoU = 0.50
──────────────────────────────────────────────────

DIAGNOSIS: Model is OVERFITTING

ROOT CAUSES:
───────────
    1. Training set too small:
       Can memorize instead of learning generalizable features
    
    2. Model too large:
       3 transformer layers might be overkill
    
    3. Training too long:
       Fitting to noise in training data
    
    4. No regularization:
       Model can overfit without constraints
    
    5. Different preprocessing in validation:
       Check normalization is identical

FIXES:
────
    1. Early stopping:
        best_iou = 0
        patience = 3
        no_improve_count = 0
        
        for epoch in range(max_epochs):
            train(...)
            val_iou = validate(...)
            
            if val_iou > best_iou:
                best_iou = val_iou
                save_checkpoint(model)
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count >= patience:
                print("Stopping early")
                break
    
    2. Increase regularization:
        # In config:
        weight_decay: 0.01 → 0.05  # Stronger L2
        dropout: 0.1 → 0.3  # More dropout
    
    3. Data augmentation:
        img = random_rotate(img)
        img = random_flip(img)
        img = random_color_jitter(img)
    
    4. Reduce model size:
        num_layers: 3 → 2  # Fewer transformer layers
    
    5. Verify preprocessing:
        In training:
            img = (img / 255 - mean) / std
        In validation:
            img = (img / 255 - mean) / std  # MUST be identical
"""

# ============================================================================
# 7. CUDA DEVICE ERRORS IN DISTRIBUTED TRAINING
# ============================================================================

"""
ERROR: CUDA runtime error (701) - an illegal memory access was encountered
───────────────────────────────────────────────────────────────────────────

SOLUTIONS:
────────
    1. Reduce batch_size (memory access issues):
       batch_size: 64 → 32
    
    2. Check all tensors on same device:
        ✗ WRONG: pred (GPU) vs target (CPU)
        ✓ CORRECT:
            target = target.cuda()
            loss = criterion(pred, target)
    
    3. Explicit device placement:
        device = torch.device(f'cuda:{rank}')
        model = model.to(device)
        img = img.to(device)
        text = text.to(device)
        target = target.to(device)
    
    4. Use rank-specific GPU:
        torch.cuda.set_device(local_rank)
        # (already done in distributed setup)
    
    5. Verify distributed setup:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.distributed.init_process_group(backend='nccl')

DEBUGGING DISTRIBUTED:
─────────────────────
    # In main_worker
    rank = args.rank
    world_size = args.world_size
    
    print(f"[GPU {rank}] Starting, world_size={world_size}")
    print(f"[GPU {rank}] Using device: cuda:{rank}")
    
    torch.cuda.set_device(rank)
    
    # After init
    print(f"[GPU {rank}] Distributed initialized")
    print(f"[GPU {rank}] Rank: {dist.get_rank()}, World size: {dist.get_world_size()}")
"""

# ============================================================================
# 8. CHECKPOINT LOADING ISSUES
# ============================================================================

"""
ERROR: KeyError when loading checkpoint
───────────────────────────────────────

CAUSE: Model structure changed, saved checkpoint is incompatible

SOLUTION:
────────
    # Flexible checkpoint loading
    checkpoint = torch.load('checkpoint.pth.tar')
    
    # Get state dict from checkpoint
    state_dict = checkpoint['state_dict']
    
    # Remove 'module.' prefix if needed (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.'
        else:
            new_state_dict[k] = v
    
    # Load with mismatched keys allowed
    missing_keys, unexpected_keys = model.load_state_dict(
        new_state_dict,
        strict=False  # Allow missing/extra keys
    )
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")

CHECKPOINT STRUCTURE:
───────────────────
    ✓ Correct:
        {
            'epoch': 25,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_iou': 0.75
        }
    
    ✗ Wrong (incomplete):
        {
            'model': model.state_dict()  # No epoch info
        }

RESUMING TRAINING:
─────────────────
    checkpoint = torch.load('checkpoint_25.pth.tar')
    
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_iou = checkpoint['best_iou']
    
    for epoch in range(start_epoch, max_epochs):
        train(...)
        val_iou = validate(...)
"""

# ============================================================================
# 9. REPRODUCIBILITY & RANDOMNESS
# ============================================================================

"""
ISSUE: Same code gives different results each run
─────────────────────────────────────────────────

CAUSE: PyTorch randomness (shuffling, initialization, etc.)

SOLUTION - Set random seeds:
───────────────────────────
    import torch
    import numpy as np
    import random
    
    def set_seed(seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Enable determinism (might be slower)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Call at start of main_worker
    set_seed(args.seed)
    
    # In config:
    seed: 42

IMPORTANT:
─────────
    torch.backends.cudnn.deterministic = True
    Makes CUDA operations deterministic but SLOWER
    
    Only use when you need reproducibility
    For actual training, use benchmark=True for speed

STILL NOT REPRODUCIBLE?
──────────────────────
    1. DistributedSampler shuffle:
       sampler = DistributedSampler(
           dataset,
           shuffle=False  # Deterministic if False
       )
    
    2. DataLoader workers:
       num_workers=0  # No parallel workers for determinism
       (slower but reproducible)
    
    3. Dropout/batch norm randomness:
       model.eval()  # Disable dropout/bn randomness
"""

# ============================================================================
# 10. QUICK CHECKLIST FOR DEBUGGING
# ============================================================================

"""
GENERAL DEBUGGING FLOW:
──────────────────────

1. FIRST RUN FAILED?
   ☐ Check Python syntax: python -m py_compile train.py
   ☐ Check imports: python -c "from model import *"
   ☐ Check config file: python -c "from utils.config import load_cfg"
   ☐ Check GPU: python -c "import torch; print(torch.cuda.is_available())"

2. SHAPES WRONG?
   ☐ Add print statements: print(f"shape: {tensor.shape}")
   ☐ Use debugger: import pdb; pdb.set_trace()
   ☐ Compare with original CRIS.pytorch

3. MEMORY ERROR?
   ☐ Reduce batch_size (test with batch_size=1)
   ☐ Reduce num_steps: 10 → 5
   ☐ Reduce num_workers: 4 → 0
   ☐ Check for memory leaks: model.eval(), del tensor

4. TRAINING NOT CONVERGING?
   ☐ Check if model.train() is called
   ☐ Check if optimizer.step() is called
   ☐ Print loss every iteration: print(f"Epoch {e}, Loss {loss}")
   ☐ Check learning rate: print(optimizer.param_groups[0]['lr'])
   ☐ Verify data: print(img.min(), img.max(), target.min(), target.max())

5. VALIDATION IoU LOW?
   ☐ Check thresholds: 0.35 for segmentation
   ☐ Verify mask file paths
   ☐ Check image preprocessing matches training
   ☐ Ensure model.eval() during validation

6. DISTRIBUTED TRAINING SLOW?
   ☐ Check torch.distributed status: print(dist.is_available())
   ☐ Verify all GPUs are used: nvidia-smi
   ☐ Check communication bottleneck: reduce num_epochs for profiling
   ☐ Enable profiling: torch.profiler.profile()

MINIMAL TEST CASE:
─────────────────
    python -c "
    import torch
    from model.segmenter import CRIS
    from utils.config import Config
    
    # Create dummy config
    cfg = Config()
    cfg.input_size = 416
    cfg.num_steps = 10
    cfg.vis_dim = 512
    cfg.word_dim = 1024
    cfg.num_layers = 3
    cfg.num_head = 8
    cfg.dim_ffn = 2048
    
    # Create model
    model = CRIS(cfg).cuda()
    
    # Create dummy inputs
    img = torch.randn(1, 3, 416, 416).cuda()
    text = torch.randint(0, 49408, (1, 77)).cuda()
    mask = torch.randint(0, 2, (1, 1, 416, 416)).float().cuda()
    
    # Forward pass
    pred, target, loss = model(img, text, mask)
    
    print(f'Success! Pred shape: {pred.shape}, Loss: {loss.item()}')
    "
"""
