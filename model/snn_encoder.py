import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF

class SNNVisionEncoder(nn.Module):
    def __init__(self, input_size=416, num_steps=10, beta=0.9):
        super().__init__()
        self.num_steps = num_steps
        self.input_size = input_size

        # Stem layers (input: 3x416x416)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.lif_stem = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer1: 64 -> 256 channels (stride=1, size=56x56)
        self.layer1_conv = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_bn = nn.BatchNorm2d(256)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        # Layer2: 256 -> 512 channels (stride=2, size=26x26)
        self.layer2_conv = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer2_bn = nn.BatchNorm2d(512)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        # Layer3: 512 -> 1024 channels (stride=2, size=13x13)
        self.layer3_conv = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer3_bn = nn.BatchNorm2d(1024)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        """
        x: (batch, 3, 416, 416)
        Returns: [(batch, 256, 52, 52), (batch, 512, 26, 26), (batch, 1024, 13, 13)]
                 representing C3, C4, C5 scales
        """
        # Encode to spike trains using rate coding
        spike_input = self.rate_encode(x)  # (batch, 3, 416, 416, num_steps)

        # Initialize membrane potentials
        mem_stem = self.lif_stem.init_leaky()
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # Store intermediate activations across time steps
        c3_out = None  # 256x52x52
        c4_out = None  # 512x26x26
        c5_out = None  # 1024x13x13

        # Process through time steps
        for step in range(self.num_steps):
            x_step = spike_input[..., step]  # (batch, 3, 416, 416)

            # Stem: 3 -> 64 channels, size 208x208
            x_step = self.maxpool(self.lif_stem(self.bn1(self.conv1(x_step)), mem_stem)[0])

            # Layer1: 64 -> 256 channels, size 52x52 (after avg_pool stride=4)
            x_step = F.avg_pool2d(x_step, kernel_size=4, stride=4)
            spk1, mem1 = self.lif1(self.layer1_bn(self.layer1_conv(x_step)), mem1)
            c3_out = spk1

            # Layer2: 256 -> 512 channels, size 26x26
            spk2, mem2 = self.lif2(self.layer2_bn(self.layer2_conv(spk1)), mem2)
            c4_out = spk2

            # Layer3: 512 -> 1024 channels, size 13x13
            spk3, mem3 = self.lif3(self.layer3_bn(self.layer3_conv(spk2)), mem3)
            c5_out = spk3

        # Return 3 scales like CLIP: C3, C4, C5
        return [c3_out, c4_out, c5_out]

    def rate_encode(self, x):
        # Rate coding: convert pixel intensity to firing rate
        # x: (batch, 3, H, W) normalized to [0,1]
        # Return: (batch, 3, H, W, num_steps)
        rates = x.unsqueeze(-1).repeat(1, 1, 1, 1, self.num_steps)
        spikes = torch.rand_like(rates) < rates
        return spikes.float()

class SNNTextEncoder(nn.Module):
    def __init__(self, vocab_size=49408, embed_dim=512, num_steps=10, beta=0.9):
        super().__init__()
        self.num_steps = num_steps
        self.embed_dim = embed_dim
        self.output_dim = 1024  # Match CLIP output

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.randn(77, embed_dim) / embed_dim**0.5)

        # SNN transformer-like layers (simplified)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.attn1 = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        
        # Project to match CLIP output dimension (1024)
        self.fc_out = nn.Linear(embed_dim, self.output_dim)

    def forward(self, text):
        """
        text: (batch, seq_len)
        Returns: (word, state)
          - word: (batch, seq_len, embed_dim) - text embeddings for transformer decoder
          - state: (batch, 1024) - text global feature for FPN fusion
        """
        x = self.token_embedding(text) + self.positional_embedding[:text.size(1)]
        
        # Initialize membrane
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Process through time
        for step in range(self.num_steps):
            # Self-attention
            attn_out, _ = self.attn1(x, x, x)
            spk1, mem1 = self.lif1(attn_out, mem1)
            spk2, mem2 = self.lif2(spk1, mem2)
            x = spk2

        # Use spike output as word embeddings for decoder
        word_embeddings = spk2  # (batch, seq_len, embed_dim)
        
        # Project spike output to get state vector for FPN
        state = spk2.mean(dim=1)  # (batch, embed_dim)
        state = self.fc_out(state)  # (batch, 1024) - match FPN expectation
        
        return word_embeddings, state