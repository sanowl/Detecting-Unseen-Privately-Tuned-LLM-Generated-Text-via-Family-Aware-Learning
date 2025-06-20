import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
import logging

from config import PhantomHunterConfig
from .feature_extractor import BaseProbabilityFeatureExtractor
from .family_encoder import ContrastiveFamilyEncoder
from .moe_detector import MixtureOfExpertsDetector

class PhantomHunter(nn.Module):
    """Complete PhantomHunter model"""
    
    def __init__(self, config: PhantomHunterConfig, base_model_names: List[str]):
        super().__init__()
        self.config = config
        self.base_model_names = base_model_names
        self.num_base_models = len(base_model_names)
        
        # Load base models for probability extraction
        self.base_models = {}
        self.tokenizers = {}
        
        for model_name in base_model_names:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                model.eval()
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                self.tokenizers[model_name] = tokenizer
                self.base_models[model_name] = model
                
            except Exception as e:
                logging.warning(f"Could not load model {model_name}: {e}")
        
        # Model components
        self.feature_extractor = BaseProbabilityFeatureExtractor(config, self.num_base_models)
        self.family_encoder = ContrastiveFamilyEncoder(config, self.num_base_models)
        self.moe_detector = MixtureOfExpertsDetector(config, self.num_base_models)
        
    def extract_base_probabilities(self, texts: List[str]) -> torch.Tensor:
        """Extract probability features from base models"""
        batch_size = len(texts)
        probability_features = []
        
        with torch.no_grad():
            for model_name in self.base_model_names:
                if model_name not in self.base_models:
                    # If model not available, use random probabilities as fallback
                    prob_features = torch.randn(batch_size, self.config.max_sequence_length)
                    probability_features.append(prob_features)
                    continue
                
                tokenizer = self.tokenizers[model_name]
                model = self.base_models[model_name]
                
                # Tokenize texts
                encoded = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_sequence_length
                )
                
                input_ids = encoded["input_ids"].to(model.device)
                attention_mask = encoded["attention_mask"].to(model.device)
                
                # Get model outputs
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # [batch_size, seq_len, vocab_size]
                
                # Convert logits to probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Get probability of next token at each position
                prob_features = []
                for b in range(batch_size):
                    seq_probs = []
                    seq_len = attention_mask[b].sum().item()
                    
                    for pos in range(min(seq_len - 1, self.config.max_sequence_length - 1)):
                        if pos + 1 < input_ids.size(1):
                            next_token_id = input_ids[b, pos + 1]
                            prob = probs[b, pos, next_token_id].item()
                            seq_probs.append(prob)
                    
                    # Pad or truncate to max_sequence_length
                    while len(seq_probs) < self.config.max_sequence_length:
                        seq_probs.append(0.0)
                    seq_probs = seq_probs[:self.config.max_sequence_length]
                    
                    prob_features.append(seq_probs)
                
                prob_tensor = torch.tensor(prob_features, dtype=torch.float32)
                probability_features.append(prob_tensor)
        
        # Stack probability features from all base models
        probability_features = torch.stack(probability_features, dim=1)  # [batch_size, num_models, seq_len]
        
        return probability_features
    
    def forward(self, texts: List[str], family_labels: Optional[torch.Tensor] = None,
                augmented_texts: Optional[List[str]] = None, 
                augmented_family_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of PhantomHunter
        
        Args:
            texts: List of input texts
            family_labels: Ground truth family labels for contrastive learning
            augmented_texts: Augmented texts for contrastive learning
            augmented_family_labels: Family labels for augmented texts
        
        Returns:
            Dictionary with model outputs
        """
        # Step 1: Extract base probability features
        probability_features = self.extract_base_probabilities(texts)
        base_features = self.feature_extractor(probability_features)
        
        # Step 2: Family-aware learning
        family_predictions, pooled_features = self.family_encoder(base_features)
        
        # Step 3: Binary detection with mixture of experts
        detection_logits = self.moe_detector(pooled_features, family_predictions)
        
        outputs = {
            "detection_logits": detection_logits,
            "family_predictions": family_predictions,
            "base_features": base_features,
            "pooled_features": pooled_features
        }
        
        # Compute contrastive loss if augmented data is provided
        if augmented_texts is not None and family_labels is not None and augmented_family_labels is not None:
            aug_probability_features = self.extract_base_probabilities(augmented_texts)
            aug_base_features = self.feature_extractor(aug_probability_features)
            
            contrastive_loss = self.family_encoder.compute_contrastive_loss(
                base_features, family_labels, aug_base_features, augmented_family_labels
            )
            outputs["contrastive_loss"] = contrastive_loss
        
        return outputs
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], binary_labels: torch.Tensor,
                    family_labels: torch.Tensor, family_labels_aug: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute total loss according to Equation (5) in the paper:
        L = λ1(LF + L~F) + λ2LB + λ3LC
        """
        device = binary_labels.device
        
        # Binary detection loss (LB)
        detection_loss = F.cross_entropy(outputs["detection_logits"], binary_labels)
        
        # Family classification loss (LF)
        family_loss = F.cross_entropy(outputs["family_predictions"], family_labels)
        
        # Augmented family classification loss (L~F)
        if family_labels_aug is not None:
            family_loss_aug = F.cross_entropy(outputs["family_predictions"], family_labels_aug)
        else:
            family_loss_aug = torch.tensor(0.0, device=device)
        
        # Contrastive loss (LC)
        contrastive_loss = outputs.get("contrastive_loss", torch.tensor(0.0, device=device))
        
        # Total loss
        total_loss = (
            self.config.lambda1 * (family_loss + family_loss_aug) +
            self.config.lambda2 * detection_loss +
            self.config.lambda3 * contrastive_loss
        )
        
        return total_loss 