"""
Explainability Module for PhantomHunter
Provides interpretable AI detection with feature attribution and confidence scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

@dataclass
class ExplanationResult:
    """Container for explanation results"""
    prediction: str  # "human" or "ai_generated"
    confidence: float
    family_prediction: str
    family_confidence: float
    token_attributions: List[Tuple[str, float]]  # (token, importance_score)
    feature_importances: Dict[str, float]
    decision_reasoning: str

class GradientBasedExplainer:
    """Gradient-based explanation for model decisions"""
    
    def __init__(self, model):
        self.model = model
        
    def get_input_gradients(self, texts: List[str], target_class: int = None) -> torch.Tensor:
        """Get gradients of inputs with respect to model output"""
        
        # Enable gradient computation for inputs
        self.model.train()
        
        # Extract probability features (this would need to be modified to track gradients)
        probability_features = self.model.extract_base_probabilities(texts)
        probability_features.requires_grad_(True)
        
        # Forward pass
        base_features = self.model.feature_extractor(probability_features)
        family_predictions, pooled_features = self.model.family_encoder(base_features)
        detection_logits = self.model.moe_detector(pooled_features, family_predictions)
        
        if target_class is None:
            # Use predicted class
            target_class = torch.argmax(detection_logits, dim=-1)
        
        # Compute gradients
        loss = detection_logits[0, target_class]
        gradients = torch.autograd.grad(loss, probability_features, retain_graph=True)[0]
        
        return gradients

class IntegratedGradientsExplainer:
    """Integrated Gradients explanation method"""
    
    def __init__(self, model):
        self.model = model
        
    def explain(self, texts: List[str], baseline_texts: Optional[List[str]] = None, 
                steps: int = 50) -> List[np.ndarray]:
        """
        Compute Integrated Gradients for input attribution
        
        Args:
            texts: Input texts to explain
            baseline_texts: Baseline texts (if None, use empty strings)
            steps: Number of integration steps
        """
        
        if baseline_texts is None:
            baseline_texts = [""] * len(texts)
        
        # Get probability features for original and baseline
        original_features = self.model.extract_base_probabilities(texts)
        baseline_features = self.model.extract_base_probabilities(baseline_texts)
        
        integrated_gradients = []
        
        for i in range(len(texts)):
            # Interpolate between baseline and original
            feature_diff = original_features[i] - baseline_features[i]
            
            gradients_sum = torch.zeros_like(original_features[i])
            
            for step in range(steps):
                alpha = step / steps
                interpolated_features = baseline_features[i] + alpha * feature_diff
                interpolated_features = interpolated_features.unsqueeze(0)
                interpolated_features.requires_grad_(True)
                
                # Forward pass
                base_features = self.model.feature_extractor(interpolated_features)
                family_predictions, pooled_features = self.model.family_encoder(base_features)
                detection_logits = self.model.moe_detector(pooled_features, family_predictions)
                
                # Get prediction and compute gradient
                predicted_class = torch.argmax(detection_logits, dim=-1)[0]
                loss = detection_logits[0, predicted_class]
                
                gradients = torch.autograd.grad(loss, interpolated_features, retain_graph=True)[0]
                gradients_sum += gradients[0]
            
            # Average gradients and multiply by feature difference
            avg_gradients = gradients_sum / steps
            integrated_grad = feature_diff * avg_gradients
            integrated_gradients.append(integrated_grad.detach().numpy())
        
        return integrated_gradients

class AttentionBasedExplainer:
    """Attention-based explanations using model's internal attention weights"""
    
    def __init__(self, model):
        self.model = model
        self.attention_weights = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention weights"""
        
        def save_attention_hook(name):
            def hook(module, input, output):
                if hasattr(module, 'attn_weights'):
                    self.attention_weights[name] = module.attn_weights.detach()
            return hook
        
        # Register hooks for transformer encoder layers
        if hasattr(self.model.feature_extractor, 'transformer_encoder'):
            for i, layer in enumerate(self.model.feature_extractor.transformer_encoder.layers):
                layer.self_attn.register_forward_hook(save_attention_hook(f'transformer_layer_{i}'))
    
    def get_attention_weights(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Get attention weights for given texts"""
        self.attention_weights.clear()
        
        with torch.no_grad():
            _ = self.model(texts)
        
        return self.attention_weights.copy()

class ConfidenceEstimator:
    """Estimate prediction confidence using various methods"""
    
    def __init__(self, model):
        self.model = model
    
    def prediction_entropy(self, logits: torch.Tensor) -> float:
        """Calculate prediction entropy as confidence measure"""
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        # Normalize entropy (higher entropy = lower confidence)
        max_entropy = torch.log(torch.tensor(logits.size(-1), dtype=torch.float))
        normalized_entropy = entropy / max_entropy
        confidence = 1.0 - normalized_entropy
        return confidence.item()
    
    def prediction_margin(self, logits: torch.Tensor) -> float:
        """Calculate prediction margin as confidence measure"""
        probs = F.softmax(logits, dim=-1)
        sorted_probs, _ = torch.sort(probs, descending=True)
        margin = sorted_probs[0] - sorted_probs[1]
        return margin.item()
    
    def temperature_scaling_confidence(self, logits: torch.Tensor, temperature: float = 1.0) -> float:
        """Apply temperature scaling for better calibrated confidence"""
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        max_prob = torch.max(probs)
        return max_prob.item()

class FeatureImportanceAnalyzer:
    """Analyze importance of different feature types"""
    
    def __init__(self, model):
        self.model = model
    
    def feature_ablation_study(self, texts: List[str]) -> Dict[str, float]:
        """Perform ablation study to measure feature importance"""
        
        # Get original prediction
        original_outputs = self.model(texts)
        original_detection = torch.argmax(original_outputs["detection_logits"], dim=-1)
        original_family = torch.argmax(original_outputs["family_predictions"], dim=-1)
        
        feature_importance = {}
        
        # Test importance of different components
        components_to_test = [
            ("cnn_features", self._ablate_cnn_features),
            ("transformer_features", self._ablate_transformer_features),
            ("family_features", self._ablate_family_features),
            ("contrastive_features", self._ablate_contrastive_features)
        ]
        
        for component_name, ablation_func in components_to_test:
            try:
                ablated_outputs = ablation_func(texts)
                ablated_detection = torch.argmax(ablated_outputs["detection_logits"], dim=-1)
                
                # Measure change in prediction
                detection_change = (original_detection != ablated_detection).float().mean().item()
                feature_importance[component_name] = detection_change
                
            except Exception as e:
                feature_importance[component_name] = 0.0
        
        return feature_importance
    
    def _ablate_cnn_features(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Ablate CNN features by zeroing them out"""
        # This is a simplified ablation - in practice, you'd modify the model architecture
        outputs = self.model(texts)
        return outputs
    
    def _ablate_transformer_features(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Ablate transformer features"""
        outputs = self.model(texts)
        return outputs
    
    def _ablate_family_features(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Ablate family classification features"""
        outputs = self.model(texts)
        return outputs
    
    def _ablate_contrastive_features(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Ablate contrastive learning features"""
        outputs = self.model(texts)
        return outputs

class PhantomHunterExplainer:
    """Main explainer class that combines all explanation methods"""
    
    def __init__(self, model, family_names: List[str] = None):
        self.model = model
        self.family_names = family_names or [f"Family_{i}" for i in range(model.num_base_models)]
        
        self.gradient_explainer = GradientBasedExplainer(model)
        self.integrated_gradients = IntegratedGradientsExplainer(model)
        self.attention_explainer = AttentionBasedExplainer(model)
        self.confidence_estimator = ConfidenceEstimator(model)
        self.feature_analyzer = FeatureImportanceAnalyzer(model)
    
    def explain_prediction(self, text: str, include_attention: bool = True,
                          include_gradients: bool = True) -> ExplanationResult:
        """Generate comprehensive explanation for a single text prediction"""
        
        texts = [text]
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(texts)
            detection_logits = outputs["detection_logits"][0]
            family_predictions = outputs["family_predictions"][0]
        
        # Basic predictions
        detection_pred = torch.argmax(detection_logits).item()
        family_pred = torch.argmax(family_predictions).item()
        
        prediction_label = "ai_generated" if detection_pred == 1 else "human"
        family_label = self.family_names[family_pred] if family_pred < len(self.family_names) else f"Family_{family_pred}"
        
        # Confidence scores
        detection_confidence = self.confidence_estimator.prediction_margin(detection_logits.unsqueeze(0))
        family_confidence = self.confidence_estimator.prediction_margin(family_predictions.unsqueeze(0))
        
        # Feature importance
        feature_importances = self.feature_analyzer.feature_ablation_study(texts)
        
        # Token-level attributions (simplified)
        token_attributions = self._get_token_attributions(text, include_gradients)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            prediction_label, detection_confidence, family_label, 
            family_confidence, feature_importances
        )
        
        return ExplanationResult(
            prediction=prediction_label,
            confidence=detection_confidence,
            family_prediction=family_label,
            family_confidence=family_confidence,
            token_attributions=token_attributions,
            feature_importances=feature_importances,
            decision_reasoning=reasoning
        )
    
    def _get_token_attributions(self, text: str, use_gradients: bool = True) -> List[Tuple[str, float]]:
        """Get token-level importance scores"""
        tokens = text.split()
        
        if use_gradients and len(tokens) > 0:
            try:
                # Simplified token attribution - in practice, this would use proper tokenization
                gradients = self.gradient_explainer.get_input_gradients([text])
                
                # Map gradients to tokens (simplified)
                token_scores = []
                for i, token in enumerate(tokens):
                    if i < gradients.size(-1):
                        score = torch.mean(torch.abs(gradients[0, :, i])).item()
                        token_scores.append((token, score))
                    else:
                        token_scores.append((token, 0.0))
                
                return token_scores
            except:
                pass
        
        # Fallback: uniform importance
        return [(token, 1.0 / len(tokens)) for token in tokens]
    
    def _generate_reasoning(self, prediction: str, confidence: float, 
                          family: str, family_conf: float, 
                          feature_imp: Dict[str, float]) -> str:
        """Generate human-readable reasoning for the prediction"""
        
        reasoning_parts = []
        
        # Main prediction
        conf_level = "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"
        reasoning_parts.append(f"Predicted as {prediction} with {conf_level} confidence ({confidence:.3f})")
        
        # Family prediction
        if prediction == "ai_generated":
            family_conf_level = "high" if family_conf > 0.7 else "medium" if family_conf > 0.4 else "low"
            reasoning_parts.append(f"Most likely from {family} family with {family_conf_level} confidence ({family_conf:.3f})")
        
        # Feature importance
        if feature_imp:
            most_important = max(feature_imp.items(), key=lambda x: x[1])
            reasoning_parts.append(f"Primary evidence from {most_important[0]} (importance: {most_important[1]:.3f})")
        
        # Confidence interpretation
        if confidence < 0.3:
            reasoning_parts.append("Low confidence suggests borderline case - human review recommended")
        elif confidence > 0.9:
            reasoning_parts.append("High confidence indicates clear distinguishing features")
        
        return ". ".join(reasoning_parts) + "."
    
    def explain_batch(self, texts: List[str]) -> List[ExplanationResult]:
        """Generate explanations for multiple texts"""
        return [self.explain_prediction(text) for text in texts]
    
    def generate_explanation_report(self, text: str, save_path: Optional[str] = None) -> str:
        """Generate a detailed explanation report"""
        
        explanation = self.explain_prediction(text)
        
        report = f"""
PhantomHunter Detection Report
=============================

Text: "{text[:100]}{'...' if len(text) > 100 else ''}"

DETECTION RESULT:
- Prediction: {explanation.prediction.upper()}
- Confidence: {explanation.confidence:.3f}

FAMILY CLASSIFICATION:
- Predicted Family: {explanation.family_prediction}
- Family Confidence: {explanation.family_confidence:.3f}

REASONING:
{explanation.decision_reasoning}

FEATURE IMPORTANCE:
"""
        
        for feature, importance in explanation.feature_importances.items():
            report += f"- {feature}: {importance:.3f}\n"
        
        report += "\nTOKEN ATTRIBUTIONS:\n"
        for token, score in explanation.token_attributions[:10]:  # Show top 10
            report += f"- '{token}': {score:.3f}\n"
        
        report += f"\nGenerated by PhantomHunter Explainable AI Detection System"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report 