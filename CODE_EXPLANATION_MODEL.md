"""
================================================================================
DETAILED CODE EXPLANATION - MODEL ARCHITECTURE
================================================================================

Phần giải thích model/segmenter.py, model/snn_encoder.py, model/clip.py
"""

# ============================================================================
# 1. MODEL/SEGMENTER.PY - CRIS_SNN VERSION
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.snn_encoder import SNNVisionEncoder, SNNTextEncoder
from .layers import FPN, Projector, TransformerDecoder


class CRIS(nn.Module):
    """CRIS model with SNN encoders"""
    
    def __init__(self, cfg):
        super().__init__()
        
        # === SNN VISION ENCODER ===
        # Convert images to multi-scale features using spiking neural networks
        self.backbone = SNNVisionEncoder(
            input_size=cfg.input_size,      # 416x416 input images
            num_steps=cfg.num_steps)        # 10 temporal steps
        
        # === SNN TEXT ENCODER ===
        # Convert text tokens to embeddings and global text feature
        self.text_encoder = SNNTextEncoder(
            vocab_size=49408,               # CLIP tokenizer vocabulary
            embed_dim=cfg.vis_dim,          # 512 embedding dimension
            num_steps=cfg.num_steps)        # 10 temporal steps
        
        # === MULTI-MODAL FPN ===
        # Fuse vision features (C3, C4, C5) with text features
        self.neck = FPN(
            in_channels=cfg.fpn_in,         # [512, 1024, 1024] input channels
            out_channels=cfg.fpn_out)       # [256, 512, 1024] output channels
        
        # === TRANSFORMER DECODER ===
        # Cross-attention between vision features and text embeddings
        self.decoder = TransformerDecoder(
            num_layers=cfg.num_layers,      # 3 transformer layers
            d_model=cfg.vis_dim,            # 512 feature dimension
            nhead=cfg.num_head,             # 8 attention heads
            dim_ffn=cfg.dim_ffn,            # 2048 feedforward dimension
            dropout=cfg.dropout,            # 0.1 dropout rate
            return_intermediate=cfg.intermediate)  # False (return only last)
        
        # === PROJECTOR ===
        # Project text features to pixel-level predictions
        self.proj = Projector(
            cfg.word_dim,                   # 1024 text dimension
            cfg.vis_dim // 2,               # 256 projection dimension
            3)                              # 3x3 kernel size
    
    def forward(self, img, word, mask=None):
        """
        Forward pass of CRIS model
        
        Args:
            img:   (batch, 3, 416, 416) - input images
            word:  (batch, 77) - tokenized text
            mask:  (batch, 1, 416, 416) - ground truth segmentation masks
        
        Returns:
            If training:
                (pred, mask, loss) - predictions, target, loss value
            If eval:
                pred - binary segmentation predictions
        """
        
        # === CREATE PADDING MASK ===
        # Mask for attention (zeros in text are padded tokens)
        # (batch, 77) → (batch, 77) boolean mask
        # True where token==0 (padding), False elsewhere
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()
        
        # === ENCODE VISION (CNN-like backbone) ===
        # Input: (batch, 3, 416, 416)
        # Process through SNN layers:
        #   - Rate encode to spike trains (internally)
        #   - Process through 10 temporal steps
        #   - Extract multi-scale features
        # Output: vis = [(batch, 256, 52, 52),    # C3 small receptive field
        #               (batch, 512, 26, 26),    # C4 medium receptive field
        #               (batch, 1024, 13, 13)]   # C5 large receptive field
        vis = self.backbone(img)
        
        # === ENCODE TEXT ===
        # Input: (batch, 77) token IDs
        # Process through SNN text layers:
        #   - Token embedding: (batch, 77, 512)
        #   - Add positional encoding
        #   - Process through 10 temporal steps with LIF neurons
        # Outputs:
        #   - word: (batch, 77, 512) - text embeddings for cross-attention
        #   - state: (batch, 1024) - global text feature for FPN fusion
        word, state = self.text_encoder(word)
        
        # === FPN (Feature Pyramid Network) ===
        # Fuse multi-scale vision features with text
        # Inputs:
        #   - vis: [(batch, 256, 52, 52), (batch, 512, 26, 26), (batch, 1024, 13, 13)]
        #   - state: (batch, 1024)
        # Process:
        #   1. Text projection: state → text spatial feature
        #   2. Feature fusion at different scales
        #   3. Upsampling and concatenation
        # Output: (batch, 512, 26, 26)
        fq = self.neck(vis, state)
        
        # Get spatial dimensions for later reshaping
        b, c, h, w = fq.size()  # b=batch, c=512, h=26, w=26
        
        # === TRANSFORMER DECODER ===
        # Cross-attention between fused features and text
        # Input: fq (batch, 512, 26, 26) reshaped to (HW=676, batch, 512)
        # Process:
        #   1. Self-attention on visual features
        #   2. Cross-attention between vision and text
        #   3. Feed-forward network
        # Output: (batch, 512, HW=676) feature map
        fq = self.decoder(fq, word, pad_mask)
        
        # Reshape back to spatial format
        # (batch, 512, HW) → (batch, 512, 26, 26)
        fq = fq.reshape(b, c, h, w)
        
        # === PROJECTOR ===
        # Project to pixel-level predictions
        # Inputs:
        #   - fq: (batch, 512, 26, 26) visual features
        #   - state: (batch, 1024) text feature
        # Process:
        #   1. Upsample visual features: 26x26 → 104x104 (4x)
        #   2. Text-to-pixel convolution
        # Output: (batch, 1, 104, 104) segmentation mask
        pred = self.proj(fq, state)
        
        # === LOSS CALCULATION ===
        if self.training:
            # Resize target mask to match prediction size
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:],
                                     mode='nearest').detach()
            
            # Binary cross-entropy loss for segmentation
            loss = F.binary_cross_entropy_with_logits(pred, mask)
            
            # Return predictions, target, and loss for logging
            return pred.detach(), mask, loss
        else:
            # Evaluation mode: return predictions only
            return pred.detach()


# ============================================================================
# 2. MODEL/SNN_ENCODER.PY - VISION ENCODER
# ============================================================================

class SNNVisionEncoder(nn.Module):
    """SNN-based vision encoder for referring expression segmentation"""
    
    def __init__(self, input_size=416, num_steps=10, beta=0.9):
        super().__init__()
        self.num_steps = num_steps           # Number of temporal steps
        self.input_size = input_size         # 416 (input image size)
        
        # === STEM LAYERS ===
        # Convert image from 3 channels to 64 feature channels
        # Input: (batch, 3, 416, 416) → Output: (batch, 64, 208, 208)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.lif_stem = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # === LAYER 1 - C3 OUTPUT ===
        # Process 64→256 channels
        # Input: (batch, 64, 208, 208) → Output: (batch, 256, 52, 52)
        self.layer1_conv = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_bn = nn.BatchNorm2d(256)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        
        # === LAYER 2 - C4 OUTPUT ===
        # Process 256→512 channels with downsampling
        # Input: (batch, 256, 52, 52) → Output: (batch, 512, 26, 26)
        self.layer2_conv = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer2_bn = nn.BatchNorm2d(512)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        
        # === LAYER 3 - C5 OUTPUT ===
        # Process 512→1024 channels with downsampling
        # Input: (batch, 512, 26, 26) → Output: (batch, 1024, 13, 13)
        self.layer3_conv = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer3_bn = nn.BatchNorm2d(1024)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
    
    def forward(self, x):
        """
        Process image through SNN layers
        
        Args:
            x: (batch, 3, 416, 416) normalized image
        
        Returns:
            [(batch, 256, 52, 52),    # C3 - small scale features
             (batch, 512, 26, 26),    # C4 - medium scale features
             (batch, 1024, 13, 13)]   # C5 - large scale features
        """
        
        # === RATE ENCODING ===
        # Convert pixel intensity to spike trains (stochastic)
        # (batch, 3, 416, 416) → (batch, 3, 416, 416, num_steps)
        # For each pixel intensity I in [0, 1]:
        #   - spike_train = (random() < I) → binary (0 or 1)
        spike_input = self.rate_encode(x)
        
        # === INITIALIZE MEMBRANE POTENTIALS ===
        # Each LIF neuron has a membrane potential that accumulates over time
        mem_stem = self.lif_stem.init_leaky()      # Initialize V=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        # === TEMPORAL PROCESSING ===
        # Process through multiple time steps (simulating neural dynamics)
        c3_out = None
        c4_out = None
        c5_out = None
        
        for step in range(self.num_steps):
            # Extract spike input for current time step
            # (batch, 3, 416, 416, num_steps) → (batch, 3, 416, 416)
            x_step = spike_input[..., step]
            
            # === STEM PASS ===
            # Input: (batch, 3, 416, 416)
            # Conv1: 3→64 channels
            x_step = self.conv1(x_step)                 # (batch, 64, 208, 208)
            x_step = self.bn1(x_step)                   # Batch normalize
            # LIF neuron: integrate spike input into membrane potential
            # Output spike if membrane > threshold
            _, mem_stem = self.lif_stem(x_step, mem_stem)
            x_step = self.maxpool(_)                    # Max pool
            
            # === LAYER 1 PASS (C3) ===
            x_step = F.avg_pool2d(x_step, kernel_size=4, stride=4)  # (batch, 64, 52, 52)
            x_step = self.layer1_conv(x_step)           # (batch, 256, 52, 52)
            x_step = self.layer1_bn(x_step)
            spk1, mem1 = self.lif1(x_step, mem1)        # LIF neuron
            c3_out = spk1                               # Save C3 output
            
            # === LAYER 2 PASS (C4) ===
            x_step = self.layer2_conv(spk1)             # (batch, 512, 26, 26)
            x_step = self.layer2_bn(x_step)
            spk2, mem2 = self.lif2(x_step, mem2)        # LIF neuron
            c4_out = spk2                               # Save C4 output
            
            # === LAYER 3 PASS (C5) ===
            x_step = self.layer3_conv(spk2)             # (batch, 1024, 13, 13)
            x_step = self.layer3_bn(x_step)
            spk3, mem3 = self.lif3(x_step, mem3)        # LIF neuron
            c5_out = spk3                               # Save C5 output
        
        # === RETURN MULTI-SCALE FEATURES ===
        # Return final spike outputs from each layer
        return [c3_out, c4_out, c5_out]
    
    def rate_encode(self, x):
        """
        Rate coding: convert pixel intensity to spike trains
        
        Higher pixel intensity = higher spike rate (more frequent spikes)
        
        Args:
            x: (batch, 3, H, W) pixel values in [0, 1]
        
        Returns:
            (batch, 3, H, W, num_steps) binary spike trains
        """
        # Expand to temporal dimension
        rates = x.unsqueeze(-1).repeat(1, 1, 1, 1, self.num_steps)  # (B, 3, H, W, T)
        
        # Generate spikes: sample from Bernoulli with parameter=pixel_intensity
        # If pixel_intensity=0.8: 80% of steps fire, 20% don't
        spikes = torch.rand_like(rates) < rates    # (B, 3, H, W, T) boolean
        
        return spikes.float()


# ============================================================================
# 3. MODEL/SNN_ENCODER.PY - TEXT ENCODER
# ============================================================================

class SNNTextEncoder(nn.Module):
    """SNN-based text encoder"""
    
    def __init__(self, vocab_size=49408, embed_dim=512, num_steps=10, beta=0.9):
        super().__init__()
        self.num_steps = num_steps
        self.embed_dim = embed_dim          # 512
        self.output_dim = 1024              # Match CLIP output for FPN
        
        # === EMBEDDING LAYER ===
        # Convert token IDs to dense vectors
        # Input: token ID (0-49407) → Output: 512-dim vector
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # === POSITIONAL ENCODING ===
        # Add position information (first token is important, last is less)
        # Learnable parameter: (77, 512)
        self.positional_embedding = nn.Parameter(
            torch.randn(77, embed_dim) / embed_dim**0.5)
        
        # === LIF NEURONS ===
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        
        # === ATTENTION LAYER ===
        # Self-attention to capture relationships between tokens
        self.attn1 = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        
        # === OUTPUT PROJECTION ===
        # Project embeddings to match FPN text feature dimension
        self.fc_out = nn.Linear(embed_dim, self.output_dim)  # 512→1024
    
    def forward(self, text):
        """
        Encode text to embeddings and global feature
        
        Args:
            text: (batch, 77) token IDs
        
        Returns:
            word_embeddings: (batch, 77, 512) - for transformer decoder
            state: (batch, 1024) - for FPN fusion
        """
        
        # === EMBED TOKENS ===
        # (batch, 77) → (batch, 77, 512)
        x = self.token_embedding(text)
        
        # === ADD POSITIONAL ENCODING ===
        # Add position info: (batch, 77, 512) + (77, 512) broadcast
        x = x + self.positional_embedding[:text.size(1)]
        
        # === INITIALIZE MEMBRANE POTENTIALS ===
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # === TEMPORAL PROCESSING ===
        for step in range(self.num_steps):
            # === SELF-ATTENTION ===
            # Learn relationships between tokens
            # Input: (batch, 77, 512)
            # Q=K=V, each attends to all positions
            # Output: (batch, 77, 512)
            attn_out, _ = self.attn1(x, x, x)
            
            # === LIF PROCESSING ===
            # Spike-based processing of attention output
            spk1, mem1 = self.lif1(attn_out, mem1)
            spk2, mem2 = self.lif2(spk1, mem2)
            
            # Update embeddings for next iteration
            x = spk2
        
        # === EXTRACT OUTPUTS ===
        word_embeddings = spk2                          # (batch, 77, 512)
        
        # Global text feature: average pool + linear projection
        state = spk2.mean(dim=1)                        # (batch, 512)
        state = self.fc_out(state)                      # (batch, 1024)
        
        return word_embeddings, state


# ============================================================================
# 4. KEY CONCEPTS - LIF NEURON (Leaky Integrate-and-Fire)
# ============================================================================

"""
LIF Neuron Dynamics:

At each time step:
1. INTEGRATE: accumulate weighted input into membrane potential
   V(t) = β * V(t-1) + x(t)
   
2. FIRE: if V(t) > threshold, output spike and reset
   spike(t) = 1 if V(t) > 1, else 0
   V(t) = V(t) - spike(t)  # Reset
   
3. LEAKY: membrane potential decays over time (β = 0.9 means 90% retention)

Beta parameter (decay factor):
- β = 0.0: no memory (amnesia) - only current input matters
- β = 0.9: high memory - neuron remembers past inputs
- β = 1.0: perfect integration - never forgets (can overflow)

Surrogate Gradient:
- Spike function (step function) is non-differentiable
- Use surrogate gradient (fast_sigmoid): smooth approximation for backprop
- Allows gradient flow while keeping spike output binary
"""


# ============================================================================
# 5. COMPARISON: CRIS.pytorch (CLIP) vs CRIS_SNN
# ============================================================================

"""
CRIS.pytorch CLIP ENCODER:
├── Load pretrained CLIP weights from .pt file
├── ResNet-50 backbone → (C3, C4, C5)
├── Deterministic: same input → same output every time
└── Fast: no temporal processing

CRIS_SNN SNN ENCODER:
├── Initialize random weights (no pretrained)
├── SNN layers with LIF neurons → (C3, C4, C5)
├── Stochastic: rate coding adds randomness
└── Slower: num_steps=10 means 10x forward passes

OUTPUT FORMAT (IDENTICAL):
- Both return: [(B,256,52,52), (B,512,26,26), (B,1024,13,13)]
- Both text returns: embeddings (B,77,512) + state (B,1024)

TRAINING:
- CLIP: already trained, can use lower learning rate or freeze layers
- SNN: random init, needs higher learning rate to converge
"""
