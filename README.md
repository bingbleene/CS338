# CRIS-SNN: Spiking Neural Network Referring Image Segmentation

Energy-efficient referring image segmentation using Spiking Neural Networks (SNNs) instead of CNNs.

## Overview

This is a modified version of CRIS (CLIP-Driven Referring Image Segmentation) that replaces:
- ❌ CLIP vision/text encoders (CNN-based)
- ✅ **SNN encoders** (energy-efficient, temporal processing)

## Key Features

### 🧠 Spiking Neural Network Architecture
- **Rate Coding**: Ảnh được chuyển thành spike trains (temporal sequences)
- **LIF Neurons**: Leaky Integrate-and-Fire neurons cho temporal dynamics
- **Surrogate Gradients**: Fast sigmoid surrogate gradients cho backprop
- **Multi-scale Output**: Maintains 3-scale pyramid (C3, C4, C5) like CLIP

### ⚡ Performance Characteristics
| Metric | Status |
|--------|--------|
| Architecture | ✅ 100% compatible with CRIS |
| Memory | May be higher (spike trains) |
| Speed | May be slower (temporal steps) |
| Energy | Should be lower (SNNs) |
| Accuracy | TBD (experimental) |

## Installation

```bash
# Install dependencies
pip install -r requirement.txt

# Key packages:
# - snntorch (for SNN operations)
# - torch >= 1.10
# - torchvision
# - wandb (for logging)
```

## Project Structure

```
cris_snn/
├── model/
│   ├── snn_encoder.py      ← SNN encoders (NEW!)
│   ├── segmenter.py        ← CRIS model with SNN
│   ├── layers.py           ← FPN, Decoder, Projector
│   └── clip.py             ← Original CLIP (kept for reference)
├── engine/
│   └── engine.py           ← Training/validation logic
├── utils/
│   ├── dataset.py          ← Data loading from LMDB
│   ├── config.py           ← Config management
│   └── misc.py             ← Utilities
├── tools/
│   ├── data_process.py     ← Prepare datasets
│   ├── folder2lmdb.py      ← Convert to LMDB format
│   └── refer.py            ← REFER API
├── config/
│   └── refcoco/cris_r50.yaml  ← Config with num_steps=10
├── data_loader.py          ← Data loading examples
├── train.py                ← Training script
├── test.py                 ← Testing script
├── FIXES_APPLIED.md        ← Detailed explanation of changes
├── verify_architecture.py  ← Validation script (RUN THIS!)
└── requirement.txt         ← Dependencies
```

## Quick Start

### 1️⃣ Verify Architecture
```bash
python verify_architecture.py
```
This checks:
- ✅ SNN encoders output correct shapes
- ✅ Model forward pass works
- ✅ Parameter groups are correct
- ✅ Data pipeline is ready

### 2️⃣ Prepare Datasets

Follow instructions in `tools/prepare_datasets.md`:
```bash
# Extract and process REFER datasets
python tools/data_process.py \
    --data_root /path/to/refer \
    --output_dir datasets/ \
    --dataset refcoco \
    --generate_mask

# Convert to LMDB format
python tools/folder2lmdb.py \
    -j datasets/anns/refcoco/train.json \
    -i /path/to/coco/images/train2014 \
    -m datasets/masks/refcoco \
    -o datasets/lmdb/refcoco \
    -s train
```

### 3️⃣ Train Model

```bash
# Single GPU (Kaggle)
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config config/refcoco/cris_r50.yaml \
    --opts TRAIN.batch_size 4 TRAIN.epochs 30
```

### 4️⃣ Evaluate Model

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
    --config config/refcoco/cris_r50.yaml \
    --opts TEST.visualize True
```

## Architecture Details

### SNNVisionEncoder
Converts images → spike trains → multi-scale features

```
Input: (batch, 3, 416, 416)
  ↓ Rate Coding [Internally]
  Spike trains: (batch, 3, 416, 416, num_steps)
  ↓ Temporal Processing (num_steps=10 iterations)
  ├─ Stem: 3→64, size 208
  ├─ Layer1: 64→256, size 52  → C3 output
  ├─ Layer2: 256→512, size 26 → C4 output  
  └─ Layer3: 512→1024, size 13 → C5 output
  ↓
Output: [(B,256,52,52), (B,512,26,26), (B,1024,13,13)]
```

### SNNTextEncoder
Text → embeddings → spike trains → text feature + global state

```
Input: (batch, 77)  [tokenized text]
  ↓ Embedding + Positional Encoding
  (batch, 77, 512)
  ↓ Temporal Processing
  ├─ LIF neurons process embeddings
  ├─ Self-attention over time
  └─ Extract last spike state
  ↓
Outputs:
  - word_embeddings: (batch, 77, 512)  → TransformerDecoder
  - state: (batch, 1024)              → FPN fusion
```

### Full Pipeline (Same as CRIS!)
```
Image + Text
  ↓
SNNVisionEncoder → (C3, C4, C5)
SNNTextEncoder   → (word, state)
  ↓
FPN(C3,C4,C5,state) → fused features
  ↓
TransformerDecoder(fused, word)
  ↓
Projector → mask
  ↓
Loss: Binary Cross Entropy
```

## Configuration

Main config file: `config/refcoco/cris_r50.yaml`

**Key SNN parameters:**
```yaml
TRAIN:
  input_size: 416           # Input image size
  word_len: 17              # Text sequence length
  num_steps: 10             # Temporal steps for SNN (NEW!)
  vis_dim: 512              # Vision feature dimension
  word_dim: 1024            # Text feature dimension
```

**Training settings:**
```yaml
  batch_size: 64
  base_lr: 0.0001
  epochs: 50
  milestones: [35]
  weight_decay: 0.0
```

## Data Flow

### Input Format
```python
img:   (batch, 3, 416, 416) normalized to [0, 1]
text:  (batch, 77) token IDs
mask:  (batch, 1, 416, 416) binary segmentation mask
```

### SNNVisionEncoder.rate_encode() [Automatic]
```python
# Rate coding: pixel intensity → firing probability
spike_trains = (rand() < pixel_intensity)  # Stochastic!
# Results: (batch, 3, 416, 416, num_steps)
```

### Processing
- Each temporal step processes spike inputs
- Membrane potentials accumulate over time  
- Final output is used for downstream tasks

## Important Notes

### ⚠️ Memory Considerations
- Spike trains consume extra memory: `batch × channels × height × width × num_steps`
- Example: batch=2, 3×416×416×10 = 10.4MB extra per batch
- **Solution**: Reduce batch_size if OOM errors

### ⚠️ Training Time
- More temporal steps = more computation
- num_steps=10: ~10x more forward passes
- **Solution**: Start with num_steps=5, increase gradually

### ⚠️ Stochastic Behavior
- Rate coding is stochastic (random spike generation)
- Set `manual_seed` for reproducibility
- Different runs may have slight variance

## Troubleshooting

### Issue: CUDA out of memory
```bash
# Reduce batch size
python train.py --config config/refcoco/cris_r50.yaml \
    --opts TRAIN.batch_size 2
```

### Issue: Slow training
```bash
# Reduce temporal steps
python train.py --config config/refcoco/cris_r50.yaml \
    --opts TRAIN.num_steps 5
```

### Issue: Low accuracy
```bash
# Increase num_steps for better temporal processing
python train.py --config config/refcoco/cris_r50.yaml \
    --opts TRAIN.num_steps 15
```

## Validation

Run validation script before training:
```bash
python verify_architecture.py
```

Expected output:
```
✅ All checks passed! Ready for training.
```

## References

### Original CRIS Paper
- "CRIS: CLIP-Driven Referring Image Segmentation" (CVPR 2022)
- https://arxiv.org/abs/2111.15174

### snnTorch Documentation
- https://snntorch.readthedocs.io/

### Spiking Neural Networks
- Understanding rate coding and temporal processing
- https://snntorch.readthedocs.io/tutorials/index.html

## Citation

If you use this work, please cite:
```bibtex
@inproceedings{wang2021cris,
  title={CRIS: CLIP-Driven Referring Image Segmentation},
  author={Wang, Zhaoqing and Lu, Yu and Li, Qiang and Tao, Xunqiang and Guo, Yandong and Gong, Mingming and Liu, Tongliang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

## Status

**✅ Architecture**: Fixed and validated
**✅ Multi-scale outputs**: Correct
**✅ Dimension matching**: Verified  
**✅ Training pipeline**: Ready
**⏳ Performance**: Experimental

---

**Last Updated**: May 9, 2026
**Version**: 1.0-snn (Fixed)
**Status**: Ready for training on Kaggle GPU
