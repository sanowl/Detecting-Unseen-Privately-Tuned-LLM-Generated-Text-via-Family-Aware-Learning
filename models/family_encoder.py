import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from config import PhantomHunterConfig

class ContrastiveFamilyEncoder(nn.Module):
    """Step 2: Contrastive Family-Aware Learning"""
    
    def __init__(self, config: PhantomHunterConfig, num_base_models: int):
        super().__init__()
        self.config = config
        self.num_base_models = num_base_models
        
        # Family classifier
        self.family_classifier = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feature_dim // 2, num_base_models)
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [batch_size, num_base_models, feature_dim]
        Returns:
            family_predictions: [batch_size, num_base_models]
            pooled_features: [batch_size, feature_dim]
        """
        # Global average pooling across base models
        pooled_features = features.mean(dim=1)  # [batch_size, feature_dim]
        
        # Family classification
        family_predictions = self.family_classifier(pooled_features)  # [batch_size, num_base_models]
        family_predictions = F.softmax(family_predictions, dim=-1)
        
        return family_predictions, pooled_features
    
    def compute_contrastive_loss(self, features: torch.Tensor, family_labels: torch.Tensor, 
                                augmented_features: torch.Tensor, augmented_family_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute SimCLR-style contrastive loss for family-aware learning
        """
        batch_size = features.size(0)
        device = features.device
        
        # Pool features
        pooled_features = features.mean(dim=1)  # [batch_size, feature_dim]
        augmented_pooled_features = augmented_features.mean(dim=1)  # [batch_size, feature_dim]
        
        # Normalize features
        pooled_features = F.normalize(pooled_features, dim=-1)
        augmented_pooled_features = F.normalize(augmented_pooled_features, dim=-1)
        
        # Concatenate original and augmented features
        all_features = torch.cat([pooled_features, augmented_pooled_features], dim=0)  # [2*batch_size, feature_dim]
        all_labels = torch.cat([family_labels, augmented_family_labels], dim=0)  # [2*batch_size]
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(all_features, all_features.T) / self.config.contrastive_temperature
        
        # Create positive mask (same family)
        labels_expanded = all_labels.unsqueeze(1).expand(-1, 2 * batch_size)
        positive_mask = (labels_expanded == labels_expanded.T).float()
        
        # Remove self-similarity
        identity_mask = torch.eye(2 * batch_size, device=device)
        positive_mask = positive_mask - identity_mask
        
        # Compute contrastive loss
        exp_similarities = torch.exp(similarity_matrix)
        
        # Positive similarities
        positive_similarities = exp_similarities * positive_mask
        positive_sum = positive_similarities.sum(dim=1, keepdim=True)
        
        # All similarities (excluding self)
        all_similarities = exp_similarities * (1 - identity_mask)
        all_sum = all_similarities.sum(dim=1, keepdim=True)
        
        # Avoid division by zero
        positive_sum = torch.clamp(positive_sum, min=1e-8)
        all_sum = torch.clamp(all_sum, min=1e-8)
        
        # Contrastive loss
        contrastive_loss = -torch.log(positive_sum / all_sum)
        contrastive_loss = contrastive_loss[positive_sum.squeeze() > 1e-8]  # Only compute loss where we have positives
        
        return contrastive_loss.mean() if len(contrastive_loss) > 0 else torch.tensor(0.0, device=device) 