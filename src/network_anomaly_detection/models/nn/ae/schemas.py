from dataclasses import dataclass


@dataclass
class AEConfig:
    input_dim: int
    encoder_dims: list[int] = (8, 4)
    decoder_dims: list[int] = (8,)