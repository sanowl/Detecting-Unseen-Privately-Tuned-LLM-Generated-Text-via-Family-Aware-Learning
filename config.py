from dataclasses import dataclass

@dataclass
class PhantomHunterConfig:
    """Configuration for PhantomHunter model"""
    feature_dim: int = 128
    num_cnn_layers: int = 3
    num_transformer_layers: int = 2
    num_attention_heads: int = 4
    contrastive_temperature: float = 0.07
    lambda1: float = 1.0  # Family classification loss weight
    lambda2: float = 1.0  # Binary detection loss weight
    lambda3: float = 0.5  # Contrastive loss weight
    dropout: float = 0.1
    max_sequence_length: int = 512 