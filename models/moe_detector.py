import torch
import torch.nn as nn
from config import PhantomHunterConfig

class MixtureOfExpertsDetector(nn.Module):
    """Step 3: Mixture-of-Experts-Based Detection"""
    
    def __init__(self, config: PhantomHunterConfig, num_base_models: int):
        super().__init__()
        self.config = config
        self.num_base_models = num_base_models
        
        # Expert detectors (one per family)
        self.expert_detectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.feature_dim, config.feature_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.feature_dim // 2, 2)  # Binary classification
            )
            for _ in range(num_base_models)
        ])
        
    def forward(self, features: torch.Tensor, family_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, feature_dim]
            family_weights: [batch_size, num_base_models]
        Returns:
            detection_logits: [batch_size, 2]
        """
        batch_size = features.size(0)
        
        # Get predictions from each expert
        expert_predictions = []
        for expert in self.expert_detectors:
            pred = expert(features)  # [batch_size, 2]
            expert_predictions.append(pred)
        
        expert_predictions = torch.stack(expert_predictions, dim=1)  # [batch_size, num_experts, 2]
        
        # Weight expert predictions by family weights
        family_weights = family_weights.unsqueeze(-1)  # [batch_size, num_experts, 1]
        weighted_predictions = expert_predictions * family_weights  # [batch_size, num_experts, 2]
        
        # Sum weighted predictions
        final_predictions = weighted_predictions.sum(dim=1)  # [batch_size, 2]
        
        return final_predictions 