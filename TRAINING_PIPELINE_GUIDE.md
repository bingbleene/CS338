"""
================================================================================
COMPREHENSIVE GUIDE - TRAINING PIPELINE
================================================================================

How all components work together in CRIS-SNN training
"""

# ============================================================================
# 1. FULL PIPELINE VISUALIZATION
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING DATA FLOW                              │
├─────────────────────────────────────────────────────────────────────────┤

Dataset (LMDB)
    ↓ [Load image, text, mask]
    ↓
RefDataset.__getitem__()
    ├─ Image: (H, W, 3) RGB → Resize to (416, 416, 3) → Normalize
    ├─ Text: "A person sitting on a bench" → Tokenize → (77,) token IDs
    ├─ Mask: (416, 416, 1) binary ground truth
    ↓ [Return batch]
    ↓
DataLoader (batch_size=64, num_workers=4)
    ├─ Image: (64, 3, 416, 416)
    ├─ Text: (64, 77)
    ├─ Mask: (64, 1, 416, 416)
    ↓
GPU Transfer (amp.autocast for mixed precision)
    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         FORWARD PASS                                    │
├─────────────────────────────────────────────────────────────────────────┤

STEP 1: Vision Encoding (SNNVisionEncoder)
────────────────────────────────────
Input:  (64, 3, 416, 416) normalized images
↓
Rate Encoding: Convert pixel intensity to spike trains
    pixel_intensity=0.7 → spike_train=[0,1,1,0,1,1,1,1,0,1] (prob 70%)
Output: (64, 3, 416, 416, 10) spike sequences
↓
Temporal Processing (num_steps=10):
    For each time step t=1..10:
        1. STEM: Conv1 + BN + LIF neuron + MaxPool
           (64, 3, 416, 416) → (64, 64, 104, 104)
        2. LAYER1: Conv + BN + LIF neuron
           (64, 64, 104, 104) → (64, 256, 52, 52)  [C3 output]
        3. LAYER2: Conv(stride=2) + BN + LIF neuron
           (64, 256, 52, 52) → (64, 512, 26, 26)  [C4 output]
        4. LAYER3: Conv(stride=2) + BN + LIF neuron
           (64, 512, 26, 26) → (64, 1024, 13, 13)  [C5 output]
        
        LIF Neuron Dynamics:
        ├─ V(t) = β * V(t-1) + input(t)  [membrane integration]
        ├─ spike(t) = 1 if V(t) > 1 else 0  [fire if above threshold]
        └─ V(t) = V(t) - spike(t)  [reset after firing]
        
        β = 0.9: exponential decay (retains 90% of previous state)
↓
Output: Multi-scale features
    C3: (64, 256, 52, 52)
    C4: (64, 512, 26, 26)
    C5: (64, 1024, 13, 13)

STEP 2: Text Encoding (SNNTextEncoder)
───────────────────────
Input: (64, 77) token IDs
↓
Token Embedding: ID → dense vector
    token_id=2043 → embedding(512,)
Output: (64, 77, 512)
↓
Add Positional Encoding:
    embedding(77, 512) contains position info
    e.g., position 0 has different encoding than position 76
Output: (64, 77, 512) with position info
↓
Temporal Processing (num_steps=10):
    For each time step t=1..10:
        1. SELF-ATTENTION: tokens attend to all other tokens
           Query, Key, Value = embeddings
           attn_weights = softmax(QK^T / √d)  [learn token relationships]
           Output: (64, 77, 512)
        
        2. LIF NEURON: spike-based processing
           V(t) = β * V(t-1) + attn_out(t)
           spike(t) = 1 if V(t) > 1 else 0
        
        3. OUTPUT SPIKES: (64, 77, 512) spike embeddings
↓
Extract Outputs:
    word_embeddings: (64, 77, 512) - for transformer decoder attention
    state (global feature): (64, 512) → FC(512→1024) → (64, 1024) - for FPN fusion
↓

STEP 3: Multi-Modal FPN (Feature Pyramid Network)
───────────────────────────────
Input:
    ├─ C3: (64, 256, 52, 52)
    ├─ C4: (64, 512, 26, 26)
    ├─ C5: (64, 1024, 13, 13)
    └─ text_state: (64, 1024)
↓
Text Projection: 1024 → 1024
    spatial_text = text_state.view(64, 1024, 1, 1)  # Make it spatial
    broadcasts to (64, 1024, 1, 1) then multiplies with C5
↓
Fusion 1: C5 + Text
    f5 = C5 * spatial_text  [element-wise text modulation]
    Output: (64, 1024, 13, 13) text-guided features
↓
Fusion 2: C4 + Upsampled C5
    f5_up = upsample(f5, 2x)  # 13×13 → 26×26
    f4_fused = concat(C4, f5_up) → Conv → (64, 512, 26, 26)
↓
Fusion 3: C3 + C4
    f3_pooled = avgpool(C3, 2x)  # 52×52 → 26×26
    f3_fused = concat(f3_pooled, f4_fused) → Conv → (64, 512, 26, 26)
↓
Aggregation: Combine all scales back to 26×26
    Extract features from each scale:
        - f5 at 13×13 → Conv → Upsample to 26×26
        - f4 at 26×26 → Conv (keep size)
        - f3 at 52×52 → Conv → Downsample to 26×26
    
    fused = concat(3 feature maps) → Conv → (64, 512, 26×26)
    
    Add coordinate information (CoordConv):
        Append pixel (x, y) normalized coordinates to each location
        Helps decoder learn spatial structure
↓
Output: (64, 512, 26, 26) multi-modal fused features

STEP 4: Transformer Decoder (Cross-modal Attention)
──────────────────────────────
Input:
    ├─ Visual: (64, 512, 26, 26) = 676 spatial locations
    ├─ Text: (64, 77, 512) = 77 word embeddings
    └─ Padding mask: (64, 77) indicating which tokens are padding
↓
Reshape for sequential attention:
    Visual: (64, 512, 26, 26) → (676, 64, 512)
    Text: (64, 77, 512) → (77, 64, 512)
↓
Add Positional Encodings:
    ├─ Visual positional encoding: (676, 1, 512)
    │  Each spatial location has unique position
    │  E.g., position (0,0) differs from (13,13)
    │
    └─ Text positional encoding: (77, 1, 512)
       Each token position has unique encoding
↓
Process through 3 Transformer Layers:
────────────────────────────────────

    LAYER 1:
    ─────
    Input: vis=(676,64,512), txt=(77,64,512)
    
    1a. SELF-ATTENTION (Vision attends to vision):
        Query = Visual features + positional encoding
        Key = Visual features + positional encoding
        Value = Visual features
        Output: (676, 64, 512)
        
        For each spatial location p1:
            attn_weights = softmax(over all other locations p2)
            Learn which locations matter together
            E.g., "chair" and "person" locations strengthen
    
    1b. RESIDUAL CONNECTION:
        updated_vis = vis + self_attn_output
    
    1c. CROSS-ATTENTION (Vision attends to text):
        Query = Updated visual + positional encoding
        Key = Text + positional encoding
        Value = Text
        Output: (676, 64, 512)
        Padding mask: ignore padding tokens
        
        For each spatial location:
            attn_weights = softmax(over all 77 tokens)
            Learn which words are relevant to this location
            E.g., location (5,3) might focus on "red" and "chair"
    
    1d. RESIDUAL CONNECTION:
        updated_vis = updated_vis + cross_attn_output
    
    1e. FEED-FORWARD:
        MLPs: 512 → 2048 → 512 (non-linear transformation)
        Output: (676, 64, 512)
    
    1f. RESIDUAL CONNECTION:
        vis = vis + ffn_output
    
    LAYER 2 & 3:
    ────────
    Same process, but input is output from previous layer
    Allows multi-hop reasoning:
        Layer 1: Direct attention (1-hop)
        Layer 2: Combines information from neighbors (2-hop)
        Layer 3: Combines neighborhood information (3-hop)
↓
Reshape back to spatial:
    (676, 64, 512) → (64, 512, 26, 26)
↓
Output: (64, 512, 26, 26) text-informed visual features

STEP 5: Projector (Pixel-level Prediction)
────────────────────
Input:
    ├─ Visual features: (64, 512, 26, 26)
    └─ Text global: (64, 1024)
↓
Visual Processing:
    1. Upsample 2×: 26×26 → 52×52
       Conv: (64, 512) channels → (64, 512) channels
    
    2. Upsample 2×: 52×52 → 104×104
       Conv: (64, 512) channels → (64, 256) channels
    
    Output: (64, 256, 104, 104)
↓
Text-Guided Convolution Parameters:
    text_params = FC(1024 → 2305)  # 2304 weights + 1 bias
                = FC(1024 → 256*3*3 + 1)
    
    weight = text_params[:2304].reshape(64, 256, 3, 3)
    bias = text_params[2304]
    
    Each sample has unique convolution kernel based on its text!
↓
Apply Text-Guided Grouped Convolution:
    Input: (1, 64*256, 104, 104)
    Kernel: (64, 256, 3, 3) with groups=64
    
    Each group i:
        Apply kernel i to feature group i
        (1, 256, 104, 104) * (1, 256, 3, 3) → (1, 1, 104, 104)
    
    Combine all: (64, 1, 104, 104)
↓
Output: (64, 1, 104, 104) logit predictions
    Values typically in range [-10, 10]
    Negative = background, Positive = object

┌─────────────────────────────────────────────────────────────────────────┐
│                         LOSS COMPUTATION                                │
├─────────────────────────────────────────────────────────────────────────┤

STEP 6: Loss Calculation
────────────
Input:
    ├─ Predictions: (64, 1, 104, 104) logits
    └─ Target: (64, 1, 416, 416) binary masks
↓
Resize Target:
    If shapes don't match:
        Interpolate target to match prediction size
        (64, 1, 416, 416) → (64, 1, 104, 104)
↓
Binary Cross-Entropy with Logits:
    BCE(pred, target) = -mean(target * log(sigmoid(pred)) +
                              (1-target) * log(1 - sigmoid(pred)))
    
    For each pixel:
        If target=1 (object): loss = -log(sigmoid(pred))
        If target=0 (background): loss = -log(1 - sigmoid(pred))
↓
Loss Value: Single scalar (e.g., 0.45)

┌─────────────────────────────────────────────────────────────────────────┐
│                      BACKWARD PASS & UPDATES                            │
├─────────────────────────────────────────────────────────────────────────┤

STEP 7: Gradient Computation
──────────────────
Backpropagation through all layers:
    
    ∂loss/∂pred → ∂loss/∂(Projector params)
            ↓
            → ∂loss/∂(visual features from decoder)
            ↓
            → ∂loss/∂(Decoder params)
            ↓
            → ∂loss/∂(FPN params)
            ↓
            → ∂loss/∂(Encoder params)
            ↓
            → ∂loss/∂(input image) [not used, just detached]
↓
Mixed Precision (AMP):
    Forward: FP32 → FP16 (mixed precision for speed)
    Loss scaling: multiply by large number to prevent underflow
    Backward: compute gradients in FP16
    Update: convert back to FP32 before optimizer step
↓

STEP 8: Gradient Clipping
──────────────────
If max_norm > 0:
    total_norm = sqrt(sum of squared gradients)
    If total_norm > max_norm:
        Clip all gradients: grad *= max_norm / total_norm
    
    Prevents exploding gradients (numerical instability)
↓

STEP 9: Optimizer Step (AdamW)
──────────────────────
For each parameter:
    m(t) = β1 * m(t-1) + (1-β1) * grad(t)     [momentum]
    v(t) = β2 * v(t-1) + (1-β2) * grad²(t)    [2nd moment]
    
    param(t) = param(t-1) - lr * m(t) / (sqrt(v(t)) + eps) - lr * λ * param(t-1)
               └───────────────────────────────────────────  └─────────────────
               Adaptive learning rate                        Weight decay
    
    Default: β1=0.9, β2=0.999, lr=0.0001, weight_decay=0.01
↓

STEP 10: Learning Rate Scheduling
──────────────────────────
Multiple strategies:
    
    1. COSINE ANNEALING:
       lr(t) = base_lr * (1 + cos(π*t/total_steps)) / 2
       Gradually decrease from high to low over training
    
    2. WARMUP:
       First N epochs: lr = base_lr * (epoch / warmup_epochs)
       Helps training stability at start
    
    3. WITH RESTARTS:
       Periodically reset to higher lr and decay again
       Helps escape local minima

    At epoch 10 / 50:
        lr ≈ 0.00005 (halfway through, decreased from 0.0001)

┌─────────────────────────────────────────────────────────────────────────┐
│                      DISTRIBUTED TRAINING                               │
├─────────────────────────────────────────────────────────────────────────┤

Multi-GPU Setup (4 GPUs example):
────────────────────────

    GPU 0          GPU 1           GPU 2           GPU 3
    ├─Model      ├─Model        ├─Model        ├─Model
    ├─Batch 0-15 ├─Batch 16-31  ├─Batch 32-47  ├─Batch 48-63
    └─Loss=0.45  └─Loss=0.43    └─Loss=0.46    └─Loss=0.44
        ↓            ↓               ↓               ↓
    grad_i[0]   grad_i[1]       grad_i[2]       grad_i[3]
        └────────────┬───────────────┬──────────────┘
                     ↓
            ALL_REDUCE (average)
                     ↓
        avg_grad = (0.45+0.43+0.46+0.44) / 4 = 0.445
        
        Updated by all GPUs (synchronized)
    ↓
    GPU 0, 1, 2, 3 all have identical models
    Process next batch
↓

DistributedSampler:
    Ensures each GPU gets different data:
        GPU 0: batch indices [0, 64, 128, 192, ...]   (stride=4)
        GPU 1: batch indices [1, 65, 129, 193, ...]
        GPU 2: batch indices [2, 66, 130, 194, ...]
        GPU 3: batch indices [3, 67, 131, 195, ...]
    
    Maintains data order while distributing

┌─────────────────────────────────────────────────────────────────────────┐
│                       METRICS & LOGGING                                 │
├─────────────────────────────────────────────────────────────────────────┤

Training Metrics (every print_freq=20 batches):
─────────────────────────────────────

    Batch IoU = trainMetricGPU(pred, target, 0.35, 0.5)
        For predictions > 0.35 threshold:
        IoU = |pred ∩ gt| / |pred ∪ gt|
        Measures prediction accuracy on this batch
    
    Prec@50 = percentage of samples with IoU > 0.5
        If 50 out of 64 samples have IoU > 0.5: Prec@50 = 78%
    
    ALL-REDUCE across GPUs:
        loss = average loss across all 4 GPUs
        iou = average IoU across all 4 GPUs
        pr5 = average Prec@50 across all 4 GPUs

    Log to WandB:
        time/batch: how long this batch took (e.g., 0.3s)
        time/data: data loading time (e.g., 0.05s)
        training/lr: current learning rate
        training/loss: current batch loss
        training/iou: current batch IoU
        training/prec@50: current batch precision

Validation Metrics (every epoch):
──────────────────────────

    For each validation sample:
        1. Run model: pred = model(image, text)
        2. Sigmoid: pred = sigmoid(pred) → [0, 1]
        3. Upsample to original size
        4. Threshold at 0.35: binary mask = pred > 0.35
        5. Compute IoU vs ground truth
    
    After all validation samples:
        All-reduce to gather results from all GPUs
        
        mean IoU (e.g., 0.742)
        Pr@50: percentage with IoU > 0.5 (e.g., 85.3%)
        Pr@60: percentage with IoU > 0.6 (e.g., 78.2%)
        ...
        Pr@90: percentage with IoU > 0.9 (e.g., 12.1%)

┌─────────────────────────────────────────────────────────────────────────┐
│                    CHECKPOINT SAVING                                    │
├─────────────────────────────────────────────────────────────────────────┤

Every epoch, save:
    1. Model weights: model.state_dict()
    2. Optimizer state: optimizer.state_dict()
    3. Learning rate scheduler: scheduler.state_dict()
    4. Current epoch number
    5. Best validation IoU so far

Checkpoint structure:
    {
        'epoch': 10,
        'state_dict': {...model weights...},
        'optimizer': {...optimizer state...},
        'scheduler': {...scheduler state...},
        'best_iou': 0.742
    }

Can resume training:
    loaded_ckpt = torch.load('checkpoint_10.pth.tar')
    model.load_state_dict(loaded_ckpt['state_dict'])
    optimizer.load_state_dict(loaded_ckpt['optimizer'])
    start_epoch = loaded_ckpt['epoch'] + 1  # Continue from next epoch
"""

# ============================================================================
# 2. CODE FLOW SUMMARY
# ============================================================================

"""
EXECUTION ORDER:

train.py (main entry point)
├─ Argument parsing (load config)
├─ mp.spawn(main_worker, nprocs=4) [launches 4 processes]
│
└─ main_worker(rank=0..3, args):
    ├─ setup(rank, world_size) [initialize distributed]
    ├─ model = CRIS(cfg)
    ├─ optimizer = AdamW(model.parameters(), lr=0.0001)
    ├─ scheduler = CosineAnnealingWarmRestarts(...)
    ├─ scaler = GradScaler() [mixed precision]
    │
    └─ for epoch in range(50):
        ├─ train_sampler.set_epoch(epoch)  [shuffle data]
        │
        ├─ for batch_idx, (image, text, target) in train_loader:
        │   ├─ pred, target, loss = model(image, text, target)
        │   ├─ scaler.scale(loss).backward()
        │   ├─ scaler.step(optimizer)
        │   ├─ compute metrics (IoU, Prec@50)
        │   └─ log to WandB
        │
        ├─ scheduler.step()  [update learning rate]
        │
        ├─ for image, text, params in val_loader:
        │   ├─ pred = model(image, text)
        │   ├─ compute IoU vs ground truth
        │   └─ save predictions
        │
        ├─ if rank == 0:  [only rank 0 saves]
        │   └─ save checkpoint if best validation IoU
        │
        └─ dist.barrier()  [sync all processes]
"""

# ============================================================================
# 3. KEY PARAMETERS & THEIR MEANINGS
# ============================================================================

"""
CONFIG (cris_r50.yaml):
─────────────────────

INPUT:
  size: 416              # Resize image to 416×416
  
CLIP:
  name: 'ViT-B/32'       # (CRIS.pytorch only) CLIP model
  
MODEL:
  backbone: 'clip'       # (CRIS.pytorch) or 'snn' (CRIS_SNN)
  input_size: 416
  vis_dim: 512           # Visual feature dimension
  word_dim: 1024         # Text feature dimension
  num_layers: 3          # Transformer decoder layers
  num_head: 8            # Attention heads
  dim_ffn: 2048          # Feed-forward hidden dimension
  
DATASET:
  refcoco_dir: '/path/to/refcoco'
  mask_dir: '/path/to/masks'
  word_len: 77           # Text length (CLIP tokenizer max)
  num_steps: 10          # SNN temporal steps (CRIS_SNN only)
  
TRAIN:
  batch_size: 64         # Per-GPU batch size
  base_lr: 0.0001        # Initial learning rate
  epochs: 50             # Total training epochs
  warmup_epochs: 5       # Warmup phase
  weight_decay: 0.01     # L2 regularization
  max_norm: 0.1          # Gradient clipping
  
OPTIMIZER:
  type: 'adamw'          # Adam with weight decay
  beta1: 0.9
  beta2: 0.999
  
SCHEDULER:
  type: 'cosine'         # Cosine annealing
  T_max: 50              # Cycle length = total epochs
  eta_min: 1e-6          # Minimum learning rate

EVALUATION:
  threshold: 0.35        # IoU threshold for binarization
  print_freq: 20         # Print every 20 batches
"""

# ============================================================================
# 4. TRAINING CONVERGENCE PATTERNS
# ============================================================================

"""
EXPECTED TRAINING PROGRESSION:

Epoch 1:
    IoU: ~0.35-0.40 (random initialization, poor predictions)
    Loss: ~0.70 (high entropy)
    Time: ~10 min

Epoch 10 (warmup complete):
    IoU: ~0.55-0.65 (learning kicks in)
    Loss: ~0.35
    Time: ~10 min
    LR: ~0.00008 (still high during warmup)

Epoch 25:
    IoU: ~0.70 (good predictions)
    Loss: ~0.25
    LR: ~0.00005 (cosine decrease)

Epoch 50:
    IoU: ~0.75 (convergence)
    Loss: ~0.15
    LR: ~0.00001 (near minimum)

FAILURE SIGNS:
──────────
    1. Loss → ∞: Exploding gradients
       Solution: Enable gradient clipping, reduce learning rate
    
    2. Loss doesn't decrease: Too low learning rate
       Solution: Increase base_lr or warmup_epochs
    
    3. Loss oscillates: Batch size too small
       Solution: Increase batch_size (if GPU memory allows)
    
    4. Training stuck at 0.5 IoU: Initialization issue
       Solution: Check model weights are trainable, no frozen layers

CONVERGENCE FACTORS:
─────────────────
    ✓ Higher batch_size: more stable gradients (64 good)
    ✓ Higher num_steps: better SNN processing (10 standard)
    ✓ Longer warmup: stability in early phases
    ✗ Too high learning rate: exploding loss
    ✗ Too low learning rate: slow convergence
    ✗ No gradient clipping: unstable training

GOOD SIGNS:
─────────
    ✓ Loss decreases smoothly
    ✓ IoU increases steadily
    ✓ Prec@50 > 70% by epoch 30
    ✓ Validation IoU within 5% of training IoU
    ✓ No sudden spikes in loss or metrics
"""
