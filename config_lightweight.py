from dataclasses import dataclass

@dataclass
class PhantomHunterLightweightConfig:
    """Lightweight configuration for PhantomHunter model - optimized for low-resource testing"""
    # Reduced feature dimensions
    feature_dim: int = 64  # Reduced from 128
    num_cnn_layers: int = 2  # Reduced from 3
    num_transformer_layers: int = 1  # Reduced from 2
    num_attention_heads: int = 2  # Reduced from 4
    
    # Loss function parameters
    contrastive_temperature: float = 0.07
    lambda1: float = 1.0  # Family classification loss weight
    lambda2: float = 1.0  # Binary detection loss weight
    lambda3: float = 0.5  # Contrastive loss weight
    
    # Regularization
    dropout: float = 0.1
    
    # Sequence processing - significantly reduced for speed
    max_sequence_length: int = 128  # Reduced from 512
    
    # New parameters for lightweight mode
    use_cpu_only: bool = True
    enable_model_offloading: bool = True
    use_mock_models: bool = False  # For testing without downloading models
    reduced_vocab_size: int = 1000  # For mock testing 