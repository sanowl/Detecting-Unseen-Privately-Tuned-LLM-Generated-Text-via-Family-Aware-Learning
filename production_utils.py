"""
Production Utilities for PhantomHunter
Comprehensive production-ready features including monitoring, caching, error handling,
performance optimization, and deployment utilities
"""

import torch
import numpy as np
import time
import logging
import psutil
import gc
import json
import pickle
import hashlib
import asyncio
from functools import wraps, lru_cache
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import warnings
import traceback
from datetime import datetime, timedelta
import queue

# Monitoring and metrics
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import prometheus_client
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Performance monitoring
@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    inference_time: float
    memory_usage: float
    gpu_memory_usage: float
    cpu_usage: float
    batch_size: int
    sequence_length: int
    model_size: int
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class PerformanceMonitor:
    """Monitor system performance and model metrics"""
    
    def __init__(self, enable_gpu_monitoring: bool = True):
        self.enable_gpu_monitoring = enable_gpu_monitoring and torch.cuda.is_available()
        self.metrics_history = []
        self.alert_thresholds = {
            'memory_usage': 0.85,  # 85% RAM usage
            'gpu_memory_usage': 0.90,  # 90% GPU memory
            'inference_time': 5.0,  # 5 seconds per inference
            'cpu_usage': 0.80  # 80% CPU usage
        }
        
        # Initialize metrics collectors
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        self.inference_time_histogram = prometheus_client.Histogram(
            'phantom_hunter_inference_seconds',
            'Time spent on inference',
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self.memory_usage_gauge = prometheus_client.Gauge(
            'phantom_hunter_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.prediction_counter = prometheus_client.Counter(
            'phantom_hunter_predictions_total',
            'Total number of predictions made',
            ['prediction_type', 'confidence_level']
        )
    
    @contextmanager
    def monitor_inference(self, batch_size: int = 1, sequence_length: int = 512):
        """Context manager for monitoring inference performance"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_gpu_memory = self._get_gpu_memory_usage()
        start_cpu = psutil.cpu_percent()
        
        try:
            yield
        finally:
            end_time = time.time()
            inference_time = end_time - start_time
            
            metrics = PerformanceMetrics(
                inference_time=inference_time,
                memory_usage=self._get_memory_usage(),
                gpu_memory_usage=self._get_gpu_memory_usage(),
                cpu_usage=psutil.cpu_percent(),
                batch_size=batch_size,
                sequence_length=sequence_length,
                model_size=self._get_model_size(),
                timestamp=datetime.now()
            )
            
            self._record_metrics(metrics)
            self._check_alerts(metrics)
    
    def _get_memory_usage(self) -> float:
        """Get current RAM usage as percentage"""
        return psutil.virtual_memory().percent / 100.0
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage as percentage"""
        if not self.enable_gpu_monitoring:
            return 0.0
        
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            return allocated_memory / total_memory
        except Exception:
            return 0.0
    
    def _get_model_size(self) -> int:
        """Estimate current model size in memory"""
        return sum(p.numel() * p.element_size() for p in torch.nn.Module().parameters())
    
    def _record_metrics(self, metrics: PerformanceMetrics):
        """Record metrics to various backends"""
        self.metrics_history.append(metrics)
        
        # Log to standard logging
        logging.info(f"Performance metrics: {metrics}")
        
        # Record to Prometheus if available
        if PROMETHEUS_AVAILABLE:
            self.inference_time_histogram.observe(metrics.inference_time)
            self.memory_usage_gauge.set(metrics.memory_usage * 100)
        
        # Record to Weights & Biases if available
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                'inference_time': metrics.inference_time,
                'memory_usage': metrics.memory_usage,
                'gpu_memory_usage': metrics.gpu_memory_usage,
                'cpu_usage': metrics.cpu_usage
            })
    
    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check if any metrics exceed alert thresholds"""
        alerts = []
        
        if metrics.memory_usage > self.alert_thresholds['memory_usage']:
            alerts.append(f"High memory usage: {metrics.memory_usage:.2%}")
        
        if metrics.gpu_memory_usage > self.alert_thresholds['gpu_memory_usage']:
            alerts.append(f"High GPU memory usage: {metrics.gpu_memory_usage:.2%}")
        
        if metrics.inference_time > self.alert_thresholds['inference_time']:
            alerts.append(f"Slow inference: {metrics.inference_time:.2f}s")
        
        if metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
            alerts.append(f"High CPU usage: {metrics.cpu_usage:.2%}")
        
        if alerts:
            logging.warning(f"Performance alerts: {'; '.join(alerts)}")

class IntelligentCache:
    """Intelligent caching system with TTL and memory management"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        self._lock = threading.RLock()
    
    def _generate_key(self, texts: List[str], config_hash: str = "") -> str:
        """Generate cache key from inputs"""
        text_str = "\n".join(texts)
        content = f"{text_str}{config_hash}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self._lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            if self._is_expired(key):
                self._remove_item(key)
                return None
            
            # Update access time
            self.access_times[key] = time.time()
            return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache"""
        with self._lock:
            # Remove expired items
            self._cleanup_expired()
            
            # Ensure cache size limit
            while len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Add new item
            current_time = time.time()
            self.cache[key] = value
            self.access_times[key] = current_time
            self.creation_times[key] = current_time
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache item is expired"""
        if key not in self.creation_times:
            return True
        
        creation_time = self.creation_times[key]
        return time.time() - creation_time > self.ttl_seconds
    
    def _cleanup_expired(self) -> None:
        """Remove all expired items"""
        expired_keys = [k for k in self.cache.keys() if self._is_expired(k)]
        for key in expired_keys:
            self._remove_item(key)
    
    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_item(lru_key)
    
    def _remove_item(self, key: str) -> None:
        """Remove item from all cache structures"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.creation_times.pop(key, None)
    
    def clear(self) -> None:
        """Clear entire cache"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.creation_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': getattr(self, '_hits', 0) / max(getattr(self, '_requests', 1), 1),
                'memory_usage': sum(len(pickle.dumps(v)) for v in self.cache.values())
            }

def robust_error_handler(max_retries: int = 3, backoff_factor: float = 1.5):
    """Decorator for robust error handling with retries and backoff"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except torch.cuda.OutOfMemoryError as e:
                    logging.error(f"GPU OOM error in {func.__name__}: {e}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    if attempt == max_retries:
                        raise RuntimeError(f"GPU out of memory after {max_retries} retries") from e
                    
                    time.sleep(backoff_factor ** attempt)
                    last_exception = e
                
                except Exception as e:
                    logging.error(f"Error in {func.__name__} (attempt {attempt + 1}): {e}")
                    logging.error(f"Traceback: {traceback.format_exc()}")
                    
                    if attempt == max_retries:
                        raise RuntimeError(f"Function failed after {max_retries} retries") from e
                    
                    time.sleep(backoff_factor ** attempt)
                    last_exception = e
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator

class BatchProcessor:
    """Intelligent batch processing with dynamic sizing"""
    
    def __init__(self, initial_batch_size: int = 8, max_batch_size: int = 64):
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = initial_batch_size
        self.performance_history = []
        self.adjustment_threshold = 5  # Number of batches before adjustment
    
    def adaptive_batch_size(self, texts: List[str]) -> int:
        """Determine optimal batch size based on text lengths and performance"""
        
        # Adjust based on text lengths
        avg_length = np.mean([len(text.split()) for text in texts])
        
        if avg_length > 300:  # Long texts
            suggested_size = min(self.current_batch_size // 2, 4)
        elif avg_length < 50:  # Short texts
            suggested_size = min(self.current_batch_size * 2, self.max_batch_size)
        else:
            suggested_size = self.current_batch_size
        
        # Adjust based on recent performance
        if len(self.performance_history) >= self.adjustment_threshold:
            recent_times = self.performance_history[-self.adjustment_threshold:]
            avg_time = np.mean(recent_times)
            
            if avg_time > 2.0:  # Too slow
                suggested_size = max(suggested_size // 2, 1)
            elif avg_time < 0.5 and self.current_batch_size < self.max_batch_size:  # Can go faster
                suggested_size = min(suggested_size * 2, self.max_batch_size)
        
        self.current_batch_size = suggested_size
        return suggested_size
    
    def process_batches(self, texts: List[str], process_func: Callable, **kwargs) -> List[Any]:
        """Process texts in adaptive batches"""
        results = []
        
        for i in range(0, len(texts), self.current_batch_size):
            batch_texts = texts[i:i + self.current_batch_size]
            batch_size = self.adaptive_batch_size(batch_texts)
            
            # Re-batch if size changed
            if batch_size != len(batch_texts) and i + batch_size < len(texts):
                batch_texts = texts[i:i + batch_size]
            
            start_time = time.time()
            try:
                batch_results = process_func(batch_texts, **kwargs)
                results.extend(batch_results)
                
                # Record performance
                batch_time = time.time() - start_time
                self.performance_history.append(batch_time)
                
                # Keep only recent history
                if len(self.performance_history) > 20:
                    self.performance_history = self.performance_history[-20:]
                    
            except Exception as e:
                logging.error(f"Batch processing failed: {e}")
                # Try processing individually as fallback
                for text in batch_texts:
                    try:
                        result = process_func([text], **kwargs)
                        results.extend(result)
                    except Exception as inner_e:
                        logging.error(f"Individual processing failed: {inner_e}")
                        results.append(None)  # Placeholder for failed item
        
        return results

class HealthChecker:
    """System health monitoring and alerts"""
    
    def __init__(self):
        self.checks = {}
        self.last_check_times = {}
        self.check_intervals = {}
        
    def register_check(self, name: str, check_func: Callable, interval: int = 60):
        """Register a health check function"""
        self.checks[name] = check_func
        self.check_intervals[name] = interval
        self.last_check_times[name] = 0
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        current_time = time.time()
        
        for name, check_func in self.checks.items():
            # Skip if not time for this check
            if current_time - self.last_check_times[name] < self.check_intervals[name]:
                continue
            
            try:
                result = check_func()
                results[name] = {
                    'status': 'healthy' if result else 'unhealthy',
                    'details': result,
                    'timestamp': datetime.now().isoformat()
                }
                self.last_check_times[name] = current_time
                
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'details': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                logging.error(f"Health check {name} failed: {e}")
        
        return results
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        return {
            'memory': {
                'used_percent': psutil.virtual_memory().percent,
                'available_gb': psutil.virtual_memory().available / (1024**3)
            },
            'cpu': {
                'percent': psutil.cpu_percent(interval=1),
                'cores': psutil.cpu_count()
            },
            'disk': {
                'used_percent': psutil.disk_usage('/').percent,
                'free_gb': psutil.disk_usage('/').free / (1024**3)
            },
            'gpu': self._get_gpu_status(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_gpu_status(self) -> Dict[str, Any]:
        """Get GPU status information"""
        if not torch.cuda.is_available():
            return {'available': False}
        
        try:
            return {
                'available': True,
                'count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'memory_allocated': torch.cuda.memory_allocated() / (1024**3),
                'memory_cached': torch.cuda.memory_reserved() / (1024**3)
            }
        except Exception as e:
            return {'available': True, 'error': str(e)}

class ModelOptimizer:
    """Model optimization utilities for production deployment"""
    
    @staticmethod
    def optimize_for_inference(model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for inference"""
        # Set to evaluation mode
        model.eval()
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        # Try to compile with PyTorch 2.0 if available
        try:
            model = torch.compile(model, mode="max-autotune")
            logging.info("Model compiled with PyTorch 2.0")
        except Exception as e:
            logging.warning(f"Could not compile model: {e}")
        
        return model
    
    @staticmethod
    def convert_to_half_precision(model: torch.nn.Module) -> torch.nn.Module:
        """Convert model to half precision for memory efficiency"""
        try:
            model = model.half()
            logging.info("Model converted to half precision")
        except Exception as e:
            logging.warning(f"Could not convert to half precision: {e}")
        
        return model
    
    @staticmethod
    def quantize_model(model: torch.nn.Module) -> torch.nn.Module:
        """Apply dynamic quantization to model"""
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            logging.info("Model quantized successfully")
            return quantized_model
        except Exception as e:
            logging.warning(f"Could not quantize model: {e}")
            return model

class ConfigValidator:
    """Validate and sanitize configuration for production"""
    
    @staticmethod
    def validate_production_config(config) -> Dict[str, List[str]]:
        """Validate configuration for production deployment"""
        issues = {
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check critical settings
        if config.batch_size > 32:
            issues['warnings'].append("Large batch size may cause memory issues")
        
        if config.max_sequence_length > 1024:
            issues['warnings'].append("Long sequences increase memory usage and latency")
        
        if not config.mixed_precision and torch.cuda.is_available():
            issues['recommendations'].append("Enable mixed precision for better performance")
        
        if config.gradient_checkpointing:
            issues['warnings'].append("Gradient checkpointing enabled - not needed for inference")
        
        # Security checks
        if hasattr(config, 'api_key') and not config.api_key:
            issues['errors'].append("API key not configured for production")
        
        # Performance checks
        if not config.enable_watermark_detection and not config.enable_source_attribution:
            issues['warnings'].append("All advanced features disabled - consider enabling some")
        
        return issues
    
    @staticmethod
    def get_production_recommendations(config) -> Dict[str, Any]:
        """Get recommendations for production deployment"""
        return {
            'memory_optimization': {
                'use_mixed_precision': True,
                'optimize_batch_size': True,
                'enable_model_caching': True
            },
            'monitoring': {
                'enable_performance_monitoring': True,
                'enable_health_checks': True,
                'log_predictions': True
            },
            'security': {
                'enable_input_validation': True,
                'set_rate_limits': True,
                'enable_authentication': True
            },
            'scaling': {
                'use_async_processing': True,
                'enable_batch_processing': True,
                'consider_model_ensemble': True
            }
        }

# Initialize global instances
performance_monitor = PerformanceMonitor()
intelligent_cache = IntelligentCache()
health_checker = HealthChecker()
batch_processor = BatchProcessor()

def setup_production_logging():
    """Setup comprehensive logging for production"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    
    # File handler with rotation
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        'phantom_hunter.log',
        maxBytes=100*1024*1024,  # 100MB
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )
    
    # Suppress noisy third-party loggers
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)

# Async utilities for high-throughput serving
class AsyncPhantomHunter:
    """Async wrapper for PhantomHunter to handle concurrent requests"""
    
    def __init__(self, model, max_concurrent_requests: int = 10):
        self.model = model
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_requests)
        self.request_queue = asyncio.Queue(maxsize=100)
        
    async def predict_async(self, texts: List[str]) -> Dict[str, Any]:
        """Async prediction with request queuing"""
        loop = asyncio.get_event_loop()
        
        # Run prediction in thread pool
        future = loop.run_in_executor(
            self.executor,
            self._predict_sync,
            texts
        )
        
        return await future
    
    def _predict_sync(self, texts: List[str]) -> Dict[str, Any]:
        """Synchronous prediction wrapper"""
        with performance_monitor.monitor_inference(
            batch_size=len(texts),
            sequence_length=max(len(text.split()) for text in texts)
        ):
            return self.model.predict(texts)
    
    async def batch_predict_async(self, all_texts: List[str], batch_size: int = 8) -> List[Dict[str, Any]]:
        """Async batch prediction with concurrent processing"""
        tasks = []
        
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i + batch_size]
            task = self.predict_async(batch_texts)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Batch prediction failed: {result}")
                final_results.append({'error': str(result)})
            else:
                final_results.append(result)
        
        return final_results 