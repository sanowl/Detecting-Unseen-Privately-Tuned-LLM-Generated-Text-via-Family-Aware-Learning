"""
Utility functions for PhantomHunter model
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import os
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )

def save_model(model: torch.nn.Module, save_path: str, 
               config: Optional[Dict[str, Any]] = None):
    """Save model checkpoint"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config
    }
    
    torch.save(checkpoint, save_path)
    logging.info(f"Model saved to {save_path}")

def load_model(model: torch.nn.Module, load_path: str) -> Dict[str, Any]:
    """Load model checkpoint"""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found at {load_path}")
    
    checkpoint = torch.load(load_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logging.info(f"Model loaded from {load_path}")
    return checkpoint.get('config', {})

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def print_model_summary(model: torch.nn.Module, input_shape: Optional[tuple] = None):
    """Print a summary of the model architecture"""
    param_counts = count_parameters(model)
    
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Total parameters: {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,}")
    print(f"Non-trainable parameters: {param_counts['non_trainable']:,}")
    print("=" * 60)
    
    if input_shape:
        print(f"Input shape: {input_shape}")
    
    print("\nModel architecture:")
    print(model)
    print("=" * 60) 