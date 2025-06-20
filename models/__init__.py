"""
Models package for PhantomHunter
Exports all essential components for comprehensive AI-generated text detection
"""

from .phantom_hunter import (
    PhantomHunter,
    WatermarkDetector,
    StatisticalWatermarkDetector,
    FrequencyWatermarkDetector,
    SourceAttributionModule,
    StyleEncoder,
    UncertaintyEstimator
)

from .feature_extractor import BaseProbabilityFeatureExtractor
from .family_encoder import ContrastiveFamilyEncoder
from .moe_detector import MixtureOfExpertsDetector

from .adversarial import (
    TextAttacks,
    AdversarialTraining,
    ConsistencyRegularizer,
    DiversityRegularizer
)

from .explainability import (
    PhantomHunterExplainer,
    GradientBasedExplainer,
    IntegratedGradientsExplainer,
    AttentionBasedExplainer,
    ConfidenceEstimator,
    FeatureImportanceAnalyzer
)

__all__ = [
    # Core model
    'PhantomHunter',
    
    # Feature components
    'BaseProbabilityFeatureExtractor',
    'ContrastiveFamilyEncoder', 
    'MixtureOfExpertsDetector',
    
    # Watermark detection
    'WatermarkDetector',
    'StatisticalWatermarkDetector',
    'FrequencyWatermarkDetector',
    
    # Source attribution
    'SourceAttributionModule',
    'StyleEncoder',
    'UncertaintyEstimator',
    
    # Adversarial robustness
    'TextAttacks',
    'AdversarialTraining',
    'ConsistencyRegularizer',
    'DiversityRegularizer',
    
    # Explainability
    'PhantomHunterExplainer',
    'GradientBasedExplainer',
    'IntegratedGradientsExplainer',
    'AttentionBasedExplainer',
    'ConfidenceEstimator',
    'FeatureImportanceAnalyzer'
] 