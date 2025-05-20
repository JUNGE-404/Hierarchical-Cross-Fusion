import torch
import torch.nn as nn

import math
import random
import numpy as np
import torch.nn.functional as F
import numpy as np
from pathlib import Path

class HCFModel(nn.Module):
    def __init__(
        self,
        dim,
        num_layers,
        hidden_dim: int=4096,
        num_heads: int=8, 
        gate_dim: int=256,
        pooling_mode: str='mean',
    ):
        super().__init__()
        self.num_layers = num_layers
        self.pooling = pooling_mode

        # decay weighted
        self.alpha = 1 / (1 + math.exp(-random.random()))
        self.beta = 1 / (1 + math.exp(-random.random()))
        
        # projection
        self.proj_low = nn.Linear(dim, hidden_dim)
        self.proj_high = nn.Linear(dim, hidden_dim)

        # cross-attention block
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # gate network
        self.gate = nn.Sequential(
            nn.Linear(4 * hidden_dim, gate_dim),
            nn.ReLU(),
            nn.Linear(gate_dim, 4),
            nn.Softmax(dim=-1)
        )

    # feature extraction
    def _aggragate_layers(self, hidden_states):
        mid = int(self.num_layers / 2)

        # Split into lower and upper halves
        lower_half = torch.stack(hidden_states[:mid], dim=0)
        upper_half = torch.stack(hidden_states[mid:], dim=0)

        # Compute decay weights for each layer
        layer_indices = torch.arange(self.num_layers + 1, device=lower_half.device)
        distances = mid - torch.abs(layer_indices - mid)
        lower_weights = (self.alpha ** distances[:mid]).view(-1, 1, 1, 1)  # Apply decay formula for lower layers
        # print(f"lower_weights: {lower_weights.shape}")
        upper_weights = (self.beta ** distances[mid:]).view(-1, 1, 1, 1)  # Apply decay formula for upper layers
        # print(f"upper_weights: {upper_weights.shape}")

        # Apply weighted sum across lower and upper halves separately
        lower_aggregated = torch.sum(lower_half * lower_weights, dim=0).to(dtype=torch.bfloat16)
        upper_aggregated = torch.sum(upper_half * upper_weights, dim=0).to(dtype=torch.bfloat16)
        
        return lower_aggregated, upper_aggregated

    def forward(
        self, 
        hidden_states,
    ):
        """
        Args:
            hidden_states: Tuple[Tensor], shape:(batch, seq_len, hidden_dim), len:num_layers
        Returns:
            embedding: Tensor, shape:(batch, hidden_dim)
        """
        h_low, h_high = self._aggragate_layers(hidden_states)
        h_low = self.proj_low(h_low)
        h_high = self.proj_high(h_high)
        high2low, _ = self.cross_attn(query=h_high, key=h_low, value=h_low)
        low2high, _ = self.cross_attn(query=h_low, key=h_high, value=h_high)
        
        gate_input = torch.cat([h_low, h_high, low2high, high2low], dim=-1)
        gate_weights = self.gate(gate_input)
        fused_output = (gate_weights[..., 0:1] * h_low + 
                        gate_weights[..., 1:2] * low2high +
                        gate_weights[..., 2:3] * h_high + 
                        gate_weights[..., 3:4] * high2low)
        
        final_embedding = fused_output.mean(dim=1)
        return final_embedding