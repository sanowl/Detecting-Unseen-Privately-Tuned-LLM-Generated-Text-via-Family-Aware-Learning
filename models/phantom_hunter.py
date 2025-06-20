"""
PhantomHunter: AI-Generated Text Detection with Family-Aware Learning
Includes adversarial robustness, explainability, watermark detection, and source attribution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Tuple, Union
import logging
import math
import warnings

from config import PhantomHunterConfig
from .feature_extractor import BaseProbabilityFeatureExtractor
from .family_encoder import ContrastiveFamilyEncoder
from .moe_detector import MixtureOfExpertsDetector

class WatermarkDetector(nn.Module):
    """Watermark detection module for identifying embedded watermarks in text"""
    
    def __init__(self, vocab_size: int = 50257, embedding_dim: int = 768):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Watermark pattern detector
        self.pattern_detector = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Statistical watermark detector
        self.statistical_detector = StatisticalWatermarkDetector()
        
        # Frequency analysis detector
        self.frequency_detector = FrequencyWatermarkDetector()
    
    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Detect watermarks in text embeddings"""
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # Pattern-based detection
        pooled_embeddings = self._pool_embeddings(embeddings, attention_mask)
        pattern_scores = self.pattern_detector(pooled_embeddings)
        
        # Statistical detection
        statistical_scores = self.statistical_detector(embeddings, attention_mask)
        
        # Frequency analysis
        frequency_scores = self.frequency_detector(embeddings, attention_mask)
        
        # Combine detection methods
        combined_scores = (pattern_scores + statistical_scores + frequency_scores) / 3
        
        return {
            'watermark_scores': combined_scores,
            'pattern_scores': pattern_scores,
            'statistical_scores': statistical_scores,
            'frequency_scores': frequency_scores,
            'watermark_detected': combined_scores > 0.5
        }
    
    def _pool_embeddings(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool embeddings with attention mask"""
        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
        pooled = torch.sum(masked_embeddings, dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
        return pooled

class StatisticalWatermarkDetector(nn.Module):
    """Statistical watermark detection using entropy and frequency analysis"""
    
    def __init__(self):
        super().__init__()
        self.entropy_threshold = 4.5
        self.frequency_threshold = 0.1
    
    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Detect statistical watermarks"""
        batch_size = embeddings.shape[0]
        
        # Compute token-level entropy
        token_entropies = self._compute_token_entropy(embeddings)
        
        # Compute frequency patterns
        frequency_patterns = self._analyze_frequency_patterns(embeddings)
        
        # Detect anomalies
        entropy_anomalies = (token_entropies < self.entropy_threshold).float()
        frequency_anomalies = (frequency_patterns > self.frequency_threshold).float()
        
        # Combine detections
        statistical_scores = (entropy_anomalies + frequency_anomalies) / 2
        
        return statistical_scores.mean(dim=1, keepdim=True)
    
    def _compute_token_entropy(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute entropy for each token position"""
        # Normalize embeddings
        normalized = F.normalize(embeddings, dim=-1)
        
        # Compute similarity distributions
        similarities = torch.matmul(normalized, normalized.transpose(-2, -1))
        
        # Convert to probabilities
        probs = F.softmax(similarities, dim=-1)
        
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        return entropy
    
    def _analyze_frequency_patterns(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Analyze frequency patterns that might indicate watermarks"""
        # Apply FFT to detect periodic patterns
        fft_embeddings = torch.fft.fft(embeddings, dim=1)
        magnitude_spectrum = torch.abs(fft_embeddings)
        
        # Look for dominant frequencies
        max_magnitude = torch.max(magnitude_spectrum, dim=-1)[0]
        mean_magnitude = torch.mean(magnitude_spectrum, dim=-1)
        
        # Ratio indicates potential watermark patterns
        frequency_ratio = max_magnitude / (mean_magnitude + 1e-10)
        
        return frequency_ratio

class FrequencyWatermarkDetector(nn.Module):
    """Frequency-domain watermark detection"""
    
    def __init__(self):
        super().__init__()
        self.detector = nn.Sequential(
            nn.Conv1d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Detect frequency-domain watermarks"""
        # Transpose for conv1d (batch, channels, length)
        conv_input = embeddings.transpose(1, 2)
        
        # Apply convolutional detection
        scores = self.detector(conv_input)
        
        return scores

class SourceAttributionModule(nn.Module):
    """Source attribution for identifying specific model sources"""
    
    def __init__(self, hidden_dim: int = 768, num_sources: int = 50):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_sources = num_sources
        
        # Hierarchical classification
        self.family_classifier = nn.Linear(hidden_dim, 10)  # GPT, BERT, T5, etc.
        self.model_classifier = nn.Linear(hidden_dim + 10, 25)  # Specific models within families
        self.version_classifier = nn.Linear(hidden_dim + 35, num_sources)  # Specific versions
        
        # Style embedding for fine-grained attribution
        self.style_encoder = StyleEncoder(hidden_dim)
        
        # Uncertainty estimation
        self.uncertainty_estimator = UncertaintyEstimator(hidden_dim)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Hierarchical source attribution"""
        
        # Family-level classification
        family_logits = self.family_classifier(features)
        family_probs = F.softmax(family_logits, dim=-1)
        
        # Model-level classification
        family_features = torch.cat([features, family_probs], dim=-1)
        model_logits = self.model_classifier(family_features)
        model_probs = F.softmax(model_logits, dim=-1)
        
        # Version-level classification
        combined_features = torch.cat([features, family_probs, model_probs], dim=-1)
        version_logits = self.version_classifier(combined_features)
        version_probs = F.softmax(version_logits, dim=-1)
        
        # Style analysis
        style_embeddings = self.style_encoder(features)
        
        # Uncertainty estimation
        uncertainty_scores = self.uncertainty_estimator(features)
        
        return {
            'family_logits': family_logits,
            'model_logits': model_logits,
            'version_logits': version_logits,
            'family_probs': family_probs,
            'model_probs': model_probs,
            'version_probs': version_probs,
            'style_embeddings': style_embeddings,
            'uncertainty_scores': uncertainty_scores,
            'predicted_source': torch.argmax(version_logits, dim=-1)
        }

class StyleEncoder(nn.Module):
    """Encode writing style features for attribution"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        self.style_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        # Style components
        self.syntax_encoder = nn.Linear(128, 32)
        self.semantic_encoder = nn.Linear(128, 32)
        self.lexical_encoder = nn.Linear(128, 32)
        self.discourse_encoder = nn.Linear(128, 32)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Encode multiple style dimensions"""
        base_style = self.style_layers(features)
        
        syntax_style = self.syntax_encoder(base_style)
        semantic_style = self.semantic_encoder(base_style)
        lexical_style = self.lexical_encoder(base_style)
        discourse_style = self.discourse_encoder(base_style)
        
        # Combine style dimensions
        style_embedding = torch.cat([
            syntax_style, semantic_style, lexical_style, discourse_style
        ], dim=-1)
        
        return style_embedding

class UncertaintyEstimator(nn.Module):
    """Estimate prediction uncertainty for reliability assessment"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        # Aleatoric uncertainty (data uncertainty)
        self.aleatoric_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Softplus()
        )
        
        # Epistemic uncertainty (model uncertainty)
        self.epistemic_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Softplus()
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Estimate different types of uncertainty"""
        aleatoric_uncertainty = self.aleatoric_head(features)
        epistemic_uncertainty = self.epistemic_head(features)
        
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
        
        return {
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
            'total_uncertainty': total_uncertainty,
            'confidence': 1.0 / (1.0 + total_uncertainty)
        }

class PhantomHunter(nn.Module):
    """
    Comprehensive AI-generated text detection with sophisticated features:
    - Family-aware learning with contrastive training
    - Adversarial robustness
    - Explainable AI with feature attribution
    - Watermark detection
    - Source attribution
    - Uncertainty quantification
    """
    
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
        
        # Extended features
        self.watermark_detector = WatermarkDetector(
            vocab_size=config.vocab_size,
            embedding_dim=config.hidden_dim
        )
        
        self.source_attribution = SourceAttributionModule(
            hidden_dim=config.hidden_dim,
            num_sources=config.num_sources
        )
        
        # Feature fusion and final classification
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        self.detection_head = nn.Linear(config.hidden_dim, config.num_classes)
        
        # Loss weights
        self.detection_weight = config.detection_weight
        self.family_weight = config.family_weight
        self.watermark_weight = config.watermark_weight
        self.attribution_weight = config.attribution_weight
        
        # Temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
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