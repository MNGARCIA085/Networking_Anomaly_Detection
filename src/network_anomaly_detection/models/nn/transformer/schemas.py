# ============================================================
# Configuration
# ============================================================

from dataclasses import dataclass

@dataclass
class TransformerAEConfig:
    input_dim: int
    seq_len: int
    d_model: int = 64
    nhead: int = 4
    dim_feedforward: int = 128
    num_encoder_layers: int = 2
    dropout: float = 0.1