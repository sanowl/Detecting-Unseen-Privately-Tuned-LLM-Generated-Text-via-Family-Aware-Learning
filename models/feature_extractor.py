import torch
import torch.nn as nn
from config import PhantomHunterConfig

class BaseProbabilityFeatureExtractor(nn.Module):
    """Step 1: Base Probability Feature Extraction"""
    
    def __init__(self, config: PhantomHunterConfig, num_base_models: int):
        super().__init__()
        self.config = config
        self.num_base_models = num_base_models
        
        # CNN layers for processing probability features
        self.cnn_layers = nn.ModuleList()
        in_channels = num_base_models
        
        for i in range(config.num_cnn_layers):
            out_channels = config.feature_dim if i == config.num_cnn_layers - 1 else in_channels * 2
            self.cnn_layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ))
            self.cnn_layers.append(nn.ReLU())
            self.cnn_layers.append(nn.MaxPool1d(kernel_size=2, stride=1, padding=1))
            self.cnn_layers.append(nn.Dropout(config.dropout))
            in_channels = out_channels
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.feature_dim,
            nhead=config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_transformer_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(config.feature_dim, config.feature_dim)
        self.layer_norm = nn.LayerNorm(config.feature_dim)
        
    def forward(self, probability_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probability_features: [batch_size, num_base_models, sequence_length]
        Returns:
            features: [batch_size, num_base_models, feature_dim]
        """
        batch_size, num_models, seq_len = probability_features.shape
        
        # Apply CNN layers
        x = probability_features  # [batch_size, num_models, seq_len]
        for layer in self.cnn_layers:
            if isinstance(layer, nn.Conv1d):
                x = layer(x)
            elif isinstance(layer, nn.MaxPool1d):
                # Ensure we don't reduce sequence length too much
                if x.size(-1) > 1:
                    x = layer(x)
            else:
                x = layer(x)
        
        # Reshape for transformer: [batch_size, seq_len, feature_dim]
        x = x.transpose(1, 2)  # [batch_size, seq_len, feature_dim]
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling across sequence dimension
        x = x.mean(dim=1)  # [batch_size, feature_dim]
        
        # Expand to match number of base models for mixture of experts
        x = x.unsqueeze(1).expand(batch_size, num_models, -1)  # [batch_size, num_models, feature_dim]
        
        # Apply output projection and layer norm
        x = self.output_projection(x)
        x = self.layer_norm(x)
        
        return x 