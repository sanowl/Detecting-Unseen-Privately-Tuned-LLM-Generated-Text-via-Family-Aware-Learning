"""
PhantomHunter: AI-Generated Text Detection via Family-Aware Learning

A sophisticated neural network model for detecting AI-generated text using 
family-aware learning and mixture-of-experts architecture.
"""

from .config import PhantomHunterConfig
from .models import PhantomHunter, BaseProbabilityFeatureExtractor, ContrastiveFamilyEncoder, MixtureOfExpertsDetector
from .dataset import PhantomHunterDataset
from .trainer import PhantomHunterTrainer

__version__ = "1.0.0"
__author__ = "PhantomHunter Team"

__all__ = [
    'PhantomHunterConfig',
    'PhantomHunter',
    'BaseProbabilityFeatureExtractor',
    'ContrastiveFamilyEncoder',
    'MixtureOfExpertsDetector',
    'PhantomHunterDataset',
    'PhantomHunterTrainer'
] 