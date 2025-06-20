"""
Configuration for PhantomHunter AI-Generated Text Detection
Comprehensive configuration supporting all sophisticated features
"""

class PhantomHunterConfig:
    """Complete configuration for PhantomHunter model"""
    
    def __init__(self):
        # Core model parameters
        self.hidden_dim = 768
        self.num_classes = 2  # Human vs AI-generated
        self.num_families = 8  # Number of model families (GPT, Claude, etc.)
        self.num_sources = 50  # Number of specific model sources
        self.vocab_size = 50257
        self.max_sequence_length = 512
        self.dropout = 0.1
        
        # Feature extraction parameters
        self.feature_dim = 768
        self.num_attention_heads = 12
        self.num_layers = 6
        self.intermediate_size = 3072
        
        # Family-aware learning parameters
        self.family_temperature = 0.1
        self.contrastive_margin = 0.5
        self.family_weight_decay = 1e-4
        
        # Mixture of Experts parameters
        self.num_experts = 8
        self.expert_dim = 256
        self.top_k_experts = 2
        self.expert_dropout = 0.1
        self.load_balancing_weight = 0.01
        
        # Loss weights
        self.detection_weight = 1.0
        self.family_weight = 0.3
        self.watermark_weight = 0.2
        self.attribution_weight = 0.2
        self.contrastive_weight = 0.4
        self.consistency_weight = 0.1
        
        # Training parameters
        self.learning_rate = 2e-5
        self.batch_size = 16
        self.num_epochs = 10
        self.warmup_steps = 1000
        self.weight_decay = 0.01
        self.gradient_clipping = 1.0
        
        # Adversarial training parameters
        self.adversarial_training = True
        self.adversarial_probability = 0.3
        self.adversarial_epsilon = 0.1
        self.adversarial_steps = 3
        self.adversarial_step_size = 0.01
        
        # Watermark detection parameters
        self.watermark_detection = True
        self.watermark_threshold = 0.5
        self.statistical_threshold = 4.5
        self.frequency_threshold = 0.1
        
        # Source attribution parameters
        self.source_attribution = True
        self.hierarchical_classification = True
        self.style_encoding_dim = 128
        self.uncertainty_estimation = True
        
        # Explainability parameters
        self.explainability = True
        self.attribution_methods = ['integrated_gradients', 'gradient_shap', 'attention']
        self.baseline_strategy = 'zero'
        self.num_attribution_steps = 50
        
        # Model ensemble parameters
        self.ensemble_models = True
        self.num_ensemble_members = 3
        self.ensemble_weights = [0.4, 0.3, 0.3]
        
        # Calibration parameters
        self.temperature_scaling = True
        self.platt_scaling = False
        self.isotonic_regression = False
        
        # Robustness evaluation parameters
        self.robustness_evaluation = True
        self.attack_types = ['textfooler', 'bert_attack', 'deepwordbug', 'back_translation']
        self.max_attack_iterations = 20
        self.attack_success_threshold = 0.1
        
        # Data augmentation parameters
        self.data_augmentation = True
        self.augmentation_probability = 0.2
        self.augmentation_methods = ['paraphrase', 'synonym_replacement', 'random_insertion']
        
        # Regularization parameters
        self.label_smoothing = 0.1
        self.mixup_alpha = 0.2
        self.cutoff_probability = 0.1
        
        # Optimization parameters
        self.optimizer = 'adamw'
        self.scheduler = 'cosine_annealing'
        self.min_learning_rate = 1e-7
        self.patience = 3
        self.factor = 0.5
        
        # Evaluation parameters
        self.evaluation_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr']
        self.confidence_intervals = True
        self.bootstrap_samples = 1000
        
        # Hardware and efficiency parameters
        self.device = 'cuda' if self._cuda_available() else 'cpu'
        self.mixed_precision = True
        self.gradient_checkpointing = False
        self.model_parallelism = False
        self.data_parallelism = True
        
        # Logging and monitoring parameters
        self.logging_level = 'INFO'
        self.log_interval = 100
        self.save_interval = 1000
        self.early_stopping = True
        self.monitor_metric = 'val_f1'
        
        # Model checkpointing parameters
        self.save_best_model = True
        self.save_last_model = True
        self.checkpoint_dir = './checkpoints'
        self.model_name = 'phantom_hunter'
        
        # Inference parameters
        self.batch_inference = True
        self.inference_batch_size = 32
        self.return_probabilities = True
        self.return_explanations = False
        self.return_attributions = False
        
        # Base model configurations
        self.base_models = {
            'gpt2': {
                'model_name': 'gpt2',
                'tokenizer': 'gpt2',
                'max_length': 512,
                'device_map': 'auto'
            },
            'bert': {
                'model_name': 'bert-base-uncased',
                'tokenizer': 'bert-base-uncased',
                'max_length': 512,
                'device_map': 'auto'
            },
            'roberta': {
                'model_name': 'roberta-base',
                'tokenizer': 'roberta-base',
                'max_length': 512,
                'device_map': 'auto'
            }
        }
        
        # Mock model settings for testing
        self.use_mock_models = False
        self.mock_model_size = 'small'  # 'tiny', 'small', 'medium'
        self.mock_sequence_length = 128
        
        # Research experiment parameters
        self.experiment_name = 'phantom_hunter_comprehensive'
        self.seed = 42
        self.reproducible = True
        self.track_gradients = False
        self.profile_memory = False
        
        # Feature flags
        self.enable_watermark_detection = True
        self.enable_source_attribution = True
        self.enable_adversarial_training = True
        self.enable_explainability = True
        self.enable_uncertainty_quantification = True
        self.enable_style_analysis = True
        
        # Performance optimization
        self.compile_model = False  # PyTorch 2.0 compilation
        self.channels_last = False
        self.jit_trace = False
        self.quantization = False
        
        # Loss configuration
        self.focal_loss = False
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        self.class_weights = None
        self.weighted_sampling = False
        
        # Curriculum learning parameters
        self.curriculum_learning = False
        self.curriculum_strategy = 'length_based'
        self.curriculum_pace = 'linear'
        self.curriculum_epochs = 5
        
        # Multi-task learning parameters
        self.multitask_learning = True
        self.task_weights = {
            'detection': 1.0,
            'family': 0.3,
            'watermark': 0.2,
            'attribution': 0.2
        }
        
        # Domain adaptation parameters
        self.domain_adaptation = False
        self.domain_adversarial_weight = 0.1
        self.gradient_reversal_lambda = 1.0
        
        # Active learning parameters
        self.active_learning = False
        self.uncertainty_sampling = True
        self.diversity_sampling = False
        self.query_strategy = 'entropy'
        
        # Federated learning parameters
        self.federated_learning = False
        self.num_clients = 10
        self.local_epochs = 5
        self.communication_rounds = 50
        
    def _cuda_available(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def update_config(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown configuration parameter: {key}")
    
    def get_lightweight_config(self):
        """Get lightweight configuration for testing"""
        lightweight_config = PhantomHunterConfig()
        
        # Reduce model complexity
        lightweight_config.hidden_dim = 256
        lightweight_config.num_layers = 3
        lightweight_config.num_attention_heads = 4
        lightweight_config.intermediate_size = 1024
        lightweight_config.max_sequence_length = 128
        
        # Reduce expert complexity
        lightweight_config.num_experts = 4
        lightweight_config.expert_dim = 128
        lightweight_config.top_k_experts = 2
        
        # Use mock models
        lightweight_config.use_mock_models = True
        lightweight_config.mock_model_size = 'small'
        
        # Reduce training complexity
        lightweight_config.batch_size = 8
        lightweight_config.num_epochs = 3
        lightweight_config.warmup_steps = 100
        
        # Disable expensive features for testing
        lightweight_config.adversarial_training = False
        lightweight_config.gradient_checkpointing = False
        lightweight_config.mixed_precision = False
        
        return lightweight_config
    
    def validate_config(self):
        """Validate configuration parameters"""
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.num_classes >= 2, "num_classes must be at least 2"
        assert 0 <= self.dropout <= 1, "dropout must be between 0 and 1"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        
        # Validate loss weights sum
        total_weight = (self.detection_weight + self.family_weight + 
                       self.watermark_weight + self.attribution_weight)
        if abs(total_weight - 1.6) > 0.1:  # Allow some tolerance
            print(f"Warning: Loss weights sum to {total_weight}, consider normalizing")
        
        return True
    
    def to_dict(self):
        """Convert configuration to dictionary"""
        return {key: value for key, value in self.__dict__.items() 
                if not key.startswith('_')}
    
    def save_config(self, filepath: str):
        """Save configuration to file"""
        import json
        config_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    @classmethod
    def load_config(cls, filepath: str):
        """Load configuration from file"""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        config.update_config(**config_dict)
        return config 