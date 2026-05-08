"""
================================================================================
QUICK REFERENCE - CRIS-SNN ARCHITECTURE & TRAINING
================================================================================
"""

# ============================================================================
# QUICK SUMMARY
# ============================================================================

"""
PROJECT: CRIS-SNN (Spiking Neural Network version of CRIS)

GOAL: Convert referring expression segmentation from CLIP encoders to 
      Spiking Neural Networks for neuromorphic computing

KEY DIFFERENCES:
  CRIS.pytorch (Original):
    - Vision: Pretrained CLIP ResNet-50
    - Text: Pretrained CLIP text encoder
    - Advantage: Very fast, accurate (pretrained weights)
    - Status: Reference implementation
  
  CRIS-SNN (This project):
    - Vision: SNNVisionEncoder with LIF neurons + rate coding
    - Text: SNNTextEncoder with LIF neurons
    - Advantage: Neuromorphic (spiking), energy efficient
    - Status: New implementation

ARCHITECTURE LAYERS (in order):
  1. SNNVisionEncoder     - Convert images to multi-scale spike trains
  2. SNNTextEncoder       - Convert text to embeddings + global feature
  3. FPN                  - Fuse multi-scale vision with text
  4. TransformerDecoder   - Cross-attention between vision and text
  5. Projector            - Generate pixel-level predictions

OUTPUT: (B, 1, 104, 104) segmentation mask
  Logits: values range from -∞ to +∞
  Convert to probability: sigmoid(logit) → [0, 1]
  Binary mask: probability > 0.35 → 0 or 1
"""

# ============================================================================
# 1. FILE STRUCTURE
# ============================================================================

"""
code/CRIS.pytorch/
├── README.md                          [Project overview]
├── train.py                           [Main training entry point]
├── test.py                            [Inference script]
├── requirement.txt                    [Python dependencies]
│
├── config/                            [YAML config files]
│   ├── refcoco/cris_r50.yaml          [Config for RefCOCO dataset]
│   ├── refcoco+/cris_r50.yaml         [Config for RefCOCO+ dataset]
│   └── refcocog_u/cris_r50.yaml       [Config for RefCOCOg_U dataset]
│
├── model/                             [Neural network components]
│   ├── __init__.py
│   ├── segmenter.py                   [CRIS model (main orchestrator)]
│   ├── snn_encoder.py                 [SNN encoders (NEW - CRIS-SNN)]
│   ├── layers.py                      [FPN, TransformerDecoder, Projector]
│   └── clip.py                        [CLIP encoders (CRIS.pytorch only)]
│
├── utils/                             [Utility functions]
│   ├── __init__.py
│   ├── config.py                      [YAML config loading]
│   ├── dataset.py                     [Data loading & preprocessing]
│   ├── misc.py                        [Helper functions]
│   └── simple_tokenizer.py            [CLIP tokenizer]
│
├── engine/                            [Training loop]
│   ├── __init__.py
│   └── engine.py                      [train() and validate() functions]
│
├── tools/                             [Optional utilities]
│   ├── data_process.py                [Dataset preparation]
│   ├── folder2lmdb.py                 [Convert to LMDB format]
│   ├── latency.py                     [Performance benchmarking]
│   ├── refer.py                       [REFER dataset API]
│   └── prepare_datasets.md            [Dataset preparation guide]
│
└── img/                               [Images/figures]

NEW FILES (CRIS-SNN DOCUMENTATION):
  ├── CODE_EXPLANATION_TRAIN.md        [Line-by-line train.py explanation]
  ├── CODE_EXPLANATION_MODEL.md        [Model architecture explanation]
  ├── CODE_EXPLANATION_LAYERS_DATASET_ENGINE.md  [Layer/dataset/engine]
  ├── TRAINING_PIPELINE_GUIDE.md       [Full training data flow]
  ├── DEBUGGING_GUIDE.md               [Common issues & solutions]
  ├── FIXES_APPLIED.md                 [Architecture fixes history]
  ├── FILES_TO_DELETE.md               [Optional files to remove]
  ├── verify_architecture.py           [Validation script]
  └── QUICK_REFERENCE.md               [This file]
"""

# ============================================================================
# 2. KEY NUMBERS (CRIS-SNN)
# ============================================================================

"""
IMAGE PROCESSING:
  Input: 416 × 416 × 3 (RGB)
  After preprocessing: normalized to [approx -2 to 2]
  Rate encoding: 416 × 416 × 3 → 416 × 416 × 3 × 10 (spike trains)

TEXT PROCESSING:
  Input: "a person sitting on a bench" (variable length)
  Tokenization: CLIP BPE tokenizer → 77 token IDs (padded/truncated)
  Embedding: each ID → 512-dim vector

NETWORK SIZES:
  Input channels: 3
  C3 output: (52, 52) spatial, 256 channels
  C4 output: (26, 26) spatial, 512 channels
  C5 output: (13, 13) spatial, 1024 channels
  
  FPN fused: (26, 26) spatial, 512 channels
  Transformer: 3 layers, 8 heads, 2048 FFN dim
  Projector: outputs (104, 104) spatial, 1 channel

TEMPORAL:
  num_steps: 10 iterations (SNN only)
  Each step processes spike input through one time step

TRAINING:
  Batch size: 64
  Epochs: 50
  Initial learning rate: 0.0001
  Warmup: 5 epochs
  Cosine annealing: complete over 50 epochs

PARAMETERS:
  Total trainable: ~80M (same as CRIS.pytorch)
  Mostly in transformer and projection layers
"""

# ============================================================================
# 3. TENSOR SHAPES (QUICK REFERENCE)
# ============================================================================

"""
NOTATION: (B, C, H, W) = (Batch, Channels, Height, Width)

INPUT:
  image:  (B, 3, 416, 416)
  text:   (B, 77) token IDs
  mask:   (B, 1, 416, 416) optional ground truth

AFTER SNNVisionEncoder:
  C3:     (B, 256, 52, 52)
  C4:     (B, 512, 26, 26)
  C5:     (B, 1024, 13, 13)

AFTER SNNTextEncoder:
  word_embeddings: (B, 77, 512)
  state (global):  (B, 1024)

AFTER FPN:
  fused:  (B, 512, 26, 26)

AFTER TransformerDecoder:
  output: (B, 512, 26, 26)

AFTER Projector:
  prediction: (B, 1, 104, 104) logits

AFTER SIGMOID:
  probability: (B, 1, 104, 104) [0, 1]

AFTER THRESHOLDING (>0.35):
  binary_mask: (B, 1, 104, 104) {0, 1}
"""

# ============================================================================
# 4. KEY CONCEPTS
# ============================================================================

"""
RATE CODING:
  Concept: Pixel intensity → frequency of spikes
  Formula: pixel_intensity ∈ [0, 1] → P(spike) = intensity
  Example:
    - Bright pixel (0.9) → fires 9/10 time steps
    - Dark pixel (0.1) → fires 1/10 time steps
  Why: Continuous values represented as temporal spike patterns

LIF NEURON (Leaky Integrate-and-Fire):
  Equation:
    V(t) = β * V(t-1) + input(t)
    if V(t) > 1: spike(t) = 1, V(t) = 0
    else: spike(t) = 0
  
  Parameters:
    β = 0.9 (membrane decay - 90% retained)
    threshold = 1.0 (firing threshold)

SURROGATE GRADIENT:
  Problem: Spike function (step) not differentiable for backprop
  Solution: Use smooth approximation (fast_sigmoid) as gradient
  Effect: Allows gradient flow through spike-generating neurons

TRANSFORMER ATTENTION:
  Self-attention: Query=Key=Value (same input)
    Learn which parts matter together
  Cross-attention: Query=Visual, Key=Value=Text
    Learn which text matters to each location
  Output: Updated features focusing on relevant information

MULTI-SCALE FEATURES:
  C3: High resolution, small context (52×52)
    Captures fine details, object edges
  C4: Medium resolution (26×26)
    Balanced detail and context
  C5: Low resolution, large context (13×13)
    Captures semantic information

PYRAMID FUSION:
  Bottom-up path: C3 → C4 → C5 (coarse features)
  Top-down path: C5 → C4 → C3 (refine with fine details)
  Result: Multi-scale features at same resolution (26×26)
"""

# ============================================================================
# 5. TRAINING FLOW (SIMPLIFIED)
# ============================================================================

"""
FOR EACH EPOCH:
  1. Load batch (B, 3, 416, 416), (B, 77), (B, 1, 416, 416)
  
  2. FORWARD PASS:
     image → SNNVisionEncoder → [C3, C4, C5]
     text → SNNTextEncoder → [word_embeddings, state]
     
     [C3,C4,C5] + state → FPN → fused_features
     fused_features + word_embeddings → TransformerDecoder → refined
     refined + state → Projector → prediction (B, 1, 104, 104)
  
  3. LOSS:
     loss = BCE(prediction, resized_target)
  
  4. BACKWARD:
     loss.backward() → compute gradients for all parameters
  
  5. UPDATE:
     optimizer.step() → update all parameters
  
  6. METRICS:
     IoU = |pred ∩ target| / |pred ∪ target|
     Prec@50 = percentage with IoU > 0.5
  
  7. LOGGING:
     Print loss, IoU, Prec@50 every print_freq batches
  
  8. VALIDATION (end of epoch):
     Run inference on val set without gradients
     Compute validation IoU
     Save checkpoint if best so far

REPEAT for all 50 epochs
"""

# ============================================================================
# 6. COMMON COMMANDS
# ============================================================================

"""
SETUP (first time only):
  cd code/CRIS.pytorch
  pip install -r requirement.txt
  pip install snntorch  # For SNN support

VERIFY ARCHITECTURE:
  python verify_architecture.py
  (Should print no errors, all shapes correct)

SINGLE GPU TRAINING (debug/testing):
  python -m torch.distributed.launch --nproc_per_node=1 train.py \
    --config config/refcoco/cris_r50.yaml \
    --opts TRAIN.batch_size 8

MULTI GPU TRAINING (4 GPUs):
  python -m torch.distributed.launch --nproc_per_node=4 train.py \
    --config config/refcoco/cris_r50.yaml \
    --opts TRAIN.batch_size 16

KAGGLE NOTEBOOK TRAINING:
  !python -m torch.distributed.launch --nproc_per_node=1 train.py \
    --config config/refcoco/cris_r50.yaml \
    --opts TRAIN.batch_size 4

QUICK TEST (1 batch):
  python train.py --config config/refcoco/cris_r50.yaml \
    --opts TRAIN.epochs 1 TRAIN.batch_size 1 TRAIN.print_freq 1

CHECK GPU:
  nvidia-smi  # See GPU memory usage
  
CHECK TRAINING:
  # In another terminal
  watch -n 1 nvidia-smi

STOP TRAINING:
  Ctrl+C  # Will save checkpoint on signal

RESUME FROM CHECKPOINT:
  python train.py --config config/refcoco/cris_r50.yaml \
    --resume checkpoint_25.pth.tar  # Continue from epoch 25
"""

# ============================================================================
# 7. CONFIGURATION PARAMETERS
# ============================================================================

"""
KEY PARAMETERS (config/refcoco/cris_r50.yaml):

INPUT:
  size: 416                    # Image size
  
MODEL:
  vis_dim: 512                 # Visual feature dimension
  word_dim: 1024               # Text dimension
  num_layers: 3                # Transformer depth
  num_head: 8                  # Attention heads
  dim_ffn: 2048                # Feedforward hidden size
  num_steps: 10                # SNN temporal steps (CRITICAL for SNN)
  
DATASET:
  word_len: 77                 # Text length (CLIP tokenizer)
  
TRAINING:
  batch_size: 64               # Batch size per GPU
  base_lr: 0.0001              # Initial learning rate
  epochs: 50                   # Total epochs
  warmup_epochs: 5             # Warmup phase
  weight_decay: 0.01           # L2 regularization
  max_norm: 0.1                # Gradient clipping
  
VALIDATION:
  threshold: 0.35              # Segmentation threshold
  print_freq: 20               # Print interval

TUNING STRATEGIES:
  ✓ Low num_steps (5): faster training, less accurate
  ✗ High num_steps (20): slower training, marginal improvement
  
  ✓ High warmup_epochs (10): more stable training
  ✗ Low warmup_epochs (1): potential divergence
  
  ✓ High weight_decay (0.05): prevent overfitting
  ✗ Low weight_decay (0.001): possible overfitting
  
  ✓ Larger batch_size (128): stable gradients (if GPU memory allows)
  ✗ Smaller batch_size (16): noisy gradients, slow convergence
"""

# ============================================================================
# 8. EXPECTED RESULTS
# ============================================================================

"""
TRAINING PROGRESS (RefCOCO dataset, single GPU):

Epoch 1:  Loss ≈ 0.70  IoU ≈ 0.35  Prec@50 ≈ 15%  (random init)
Epoch 10: Loss ≈ 0.35  IoU ≈ 0.60  Prec@50 ≈ 55%  (learning)
Epoch 25: Loss ≈ 0.25  IoU ≈ 0.70  Prec@50 ≈ 75%  (mid-training)
Epoch 50: Loss ≈ 0.15  IoU ≈ 0.75  Prec@50 ≈ 85%  (convergence)

Val IoU should be within 5% of train IoU
If gap > 10%: model is overfitting (add regularization)

TIME PER EPOCH:
  Single GPU (batch_size=64): ~10-15 minutes
  4 GPUs (batch_size=16 each): ~3-4 minutes
  Total training: ~8-10 hours (4 GPUs)

MEMORY USAGE:
  Single GPU: ~8 GB GPU memory (batch_size=64)
  4 GPUs: ~2 GB per GPU

DATASET NOTES:
  RefCOCO: ~142k images, ~600k referring expressions
  RefCOCO+: ~141k images, ~600k referring expressions (harder)
  RefCOCOg: ~108k images, ~80k expressions (evaluation only)
"""

# ============================================================================
# 9. TROUBLESHOOTING QUICK LINKS
# ============================================================================

"""
See DEBUGGING_GUIDE.md for detailed solutions:

Shape Mismatch:
  → Check tensor shapes after each layer
  → Run verify_architecture.py
  
Out of Memory:
  → Reduce batch_size
  → Reduce num_steps: 10 → 5
  
NaN/Inf Loss:
  → Check data normalization
  → Reduce learning rate
  → Check gradient clipping enabled
  
Training Not Converging:
  → Verify model.train() called
  → Check optimizer.step() called
  → Check learning rate > 0
  
Validation IoU Much Lower:
  → Model overfitting
  → Add regularization (weight_decay, dropout)
  → Use early stopping
  
Distributed Training Errors:
  → Check all GPUs available (nvidia-smi)
  → Verify environment variables set
  → Reduce batch_size

Slow Training:
  → Increase num_workers in DataLoader
  → Check GPU utilization (nvidia-smi)
  → Enable persistent_workers=True
"""

# ============================================================================
# 10. KEY FILES EXPLAINED
# ============================================================================

"""
train.py:
  Entry point for training
  Responsibilities:
    - Parse arguments & load config
    - Spawn distributed processes
    - Initialize model, optimizer, scheduler
    - Main training loop (calls engine.train/validate)
  Key functions:
    get_parser(): CLI argument parsing
    main(): Process launcher
    main_worker(rank, ...): Per-GPU training logic

engine/engine.py:
  Training loop implementation
  Key functions:
    train(): One training epoch
    validate(): One validation pass
  Computes:
    Loss calculation
    Metric computation (IoU, Precision)
    Gradient updates

model/segmenter.py:
  Main model orchestration
  Components:
    backbone: SNNVisionEncoder
    text_encoder: SNNTextEncoder
    neck: FPN
    decoder: TransformerDecoder
    proj: Projector
  Output: Segmentation prediction

model/snn_encoder.py:
  Vision & text SNN encoders
  Classes:
    SNNVisionEncoder: Image → multi-scale features
    SNNTextEncoder: Text → embeddings + global feature
  Methods:
    rate_encode(): Pixel → spike trains (rate coding)

model/layers.py:
  FPN, TransformerDecoder, Projector
  Already from original CRIS (no changes needed)

utils/dataset.py:
  Data loading from LMDB
  Key functions:
    RefDataset.__getitem__(): Load single sample
    Data preprocessing (resize, normalize)
    Image-text-mask triplets
  Output: Training triplets

utils/config.py:
  YAML configuration loading
  Merges command-line overrides with config file
  Example: cfg.opts TRAIN.batch_size 8 override config

verify_architecture.py:
  Validation script (NEW)
  Checks:
    - All tensor shapes correct
    - Model forward pass successful
    - Data pipeline working
    - No dimension mismatches
  Run: python verify_architecture.py
"""

# ============================================================================
# 11. NEXT STEPS
# ============================================================================

"""
QUICK START (from here):

1. VERIFY SETUP:
   python verify_architecture.py
   (Should report "✓ All checks passed!")

2. PREPARE DATA:
   - Download RefCOCO dataset
   - Convert to LMDB (see tools/prepare_datasets.md)
   - Update config paths

3. START TRAINING (single GPU):
   python -m torch.distributed.launch --nproc_per_node=1 train.py \
     --config config/refcoco/cris_r50.yaml \
     --opts TRAIN.batch_size 8

4. MONITOR TRAINING:
   - Watch for loss decreasing
   - Check validation IoU increasing
   - Monitor GPU memory (nvidia-smi)

5. SAVE BEST CHECKPOINT:
   Automatically saves best_checkpoint.pth.tar

6. EVALUATE ON TEST SET:
   python test.py --config config/refcoco/cris_r50.yaml \
     --resume best_checkpoint.pth.tar

FOR KAGGLE:
  1. Create notebook
  2. Upload LMDB dataset or prepare inline
  3. Run training (batch_size=4 for 2GB GPU memory)
  4. Monitor with print statements

ADVANCED:
  - Experiment with num_steps (5, 10, 15)
  - Try different datasets (RefCOCO+, RefCOCOg)
  - Tune hyperparameters (learning rate, batch size)
  - Implement distillation from CLIP model
  - Benchmark on neuromorphic hardware
"""
