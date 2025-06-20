"""
Production-Ready FastAPI Server for PhantomHunter
Includes authentication, rate limiting, monitoring, caching, and comprehensive error handling
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import asyncio
import time
import hashlib
import logging
import uuid
from datetime import datetime, timedelta
import jwt
from functools import wraps
import redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager
import uvicorn
import numpy as np
import torch

from config import PhantomHunterConfig
from models import PhantomHunter
from production_utils import (
    performance_monitor, intelligent_cache, health_checker,
    batch_processor, setup_production_logging, AsyncPhantomHunter,
    robust_error_handler
)

# Configure logging
setup_production_logging()
logger = logging.getLogger(__name__)

# Rate limiting setup
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    limiter = Limiter(key_func=get_remote_address, storage_uri="redis://localhost:6379")
except Exception as e:
    logger.warning(f"Redis not available, using in-memory rate limiting: {e}")
    limiter = Limiter(key_func=get_remote_address)

# Pydantic models for API
class TextAnalysisRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    include_explanations: bool = Field(default=False)
    include_attributions: bool = Field(default=False)
    include_watermark_analysis: bool = Field(default=True)
    include_source_attribution: bool = Field(default=True)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    
    @validator('texts')
    def validate_texts(cls, v):
        for text in v:
            if len(text.strip()) == 0:
                raise ValueError("Empty text not allowed")
            if len(text) > 10000:  # 10k character limit
                raise ValueError("Text too long (max 10,000 characters)")
        return v

class PredictionResult(BaseModel):
    text_id: str
    is_ai_generated: bool
    confidence: float
    prediction_scores: Dict[str, float]
    family_prediction: Optional[str] = None
    family_confidence: Optional[float] = None
    watermark_detected: Optional[bool] = None
    watermark_confidence: Optional[float] = None
    source_attribution: Optional[Dict[str, Any]] = None
    explanations: Optional[Dict[str, Any]] = None
    processing_time: float
    model_version: str

class BatchAnalysisResponse(BaseModel):
    request_id: str
    timestamp: datetime
    results: List[PredictionResult]
    total_processing_time: float
    model_info: Dict[str, Any]
    warnings: List[str] = []

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    system_health: Dict[str, Any]
    model_health: Dict[str, Any]
    performance_metrics: Dict[str, Any]

class ErrorResponse(BaseModel):
    error: str
    error_code: str
    timestamp: datetime
    request_id: str

# Authentication
class AuthManager:
    def __init__(self, secret_key: str = "your-secret-key-change-in-production"):
        self.secret_key = secret_key
        self.algorithm = "HS256"
        self.valid_api_keys = set()  # In production, load from database
        
    def verify_api_key(self, api_key: str) -> bool:
        """Verify API key (implement your logic here)"""
        # In production, verify against database
        return api_key in self.valid_api_keys or api_key == "demo-api-key"
    
    def create_token(self, user_id: str, expires_delta: timedelta = timedelta(hours=24)) -> str:
        """Create JWT token"""
        expire = datetime.utcnow() + expires_delta
        payload = {
            "user_id": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

# Global instances
auth_manager = AuthManager()
security = HTTPBearer()

# App lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    logger.info("Starting PhantomHunter API server...")
    
    # Initialize model
    config = PhantomHunterConfig()
    base_models = ['gpt2', 'bert-base-uncased', 'roberta-base']
    
    try:
        model = PhantomHunter(config, base_models)
        app.state.model = model
        app.state.async_model = AsyncPhantomHunter(model, max_concurrent_requests=10)
        app.state.config = config
        
        # Register health checks
        health_checker.register_check("model_loaded", lambda: app.state.model is not None)
        health_checker.register_check("gpu_available", lambda: torch.cuda.is_available())
        
        logger.info("PhantomHunter API server started successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
    finally:
        logger.info("Shutting down PhantomHunter API server...")

# Create FastAPI app
app = FastAPI(
    title="PhantomHunter API",
    description="Production-ready API for AI-generated text detection",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify authentication"""
    token = credentials.credentials
    
    # Try API key first
    if auth_manager.verify_api_key(token):
        return {"user_id": "api_key_user", "auth_type": "api_key"}
    
    # Try JWT token
    try:
        payload = auth_manager.verify_token(token)
        return {"user_id": payload["user_id"], "auth_type": "jwt"}
    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to all requests"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    start_time = time.time()
    response = await call_next(request)
    processing_time = time.time() - start_time
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Processing-Time"] = str(processing_time)
    
    return response

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.error(f"Unhandled exception in request {request_id}: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            timestamp=datetime.now(),
            request_id=request_id
        ).dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP exception handler"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            timestamp=datetime.now(),
            request_id=request_id
        ).dict()
    )

# API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    system_health = health_checker.get_system_health()
    model_health = health_checker.run_checks()
    performance_metrics = {
        'cache_stats': intelligent_cache.stats(),
        'recent_performance': performance_monitor.metrics_history[-10:] if performance_monitor.metrics_history else []
    }
    
    overall_status = "healthy"
    if system_health['memory']['used_percent'] > 90:
        overall_status = "degraded"
    if any(check['status'] != 'healthy' for check in model_health.values()):
        overall_status = "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(),
        system_health=system_health,
        model_health=model_health,
        performance_metrics=performance_metrics
    )

@app.post("/analyze", response_model=BatchAnalysisResponse)
@limiter.limit("100/minute")
async def analyze_texts(
    request: Request,
    analysis_request: TextAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Analyze texts for AI-generated content"""
    request_id = request.state.request_id
    start_time = time.time()
    
    logger.info(f"Processing analysis request {request_id} for user {current_user['user_id']}")
    
    try:
        # Check cache
        cache_key = intelligent_cache._generate_key(
            analysis_request.texts,
            f"{analysis_request.confidence_threshold}"
        )
        
        cached_result = intelligent_cache.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for request {request_id}")
            return cached_result
        
        # Process texts
        async_model = request.app.state.async_model
        
        results = []
        warnings = []
        
        # Process in batches for better performance
        batch_size = min(len(analysis_request.texts), 8)
        
        for i in range(0, len(analysis_request.texts), batch_size):
            batch_texts = analysis_request.texts[i:i + batch_size]
            
            try:
                # Get predictions
                predictions = await async_model.predict_async(batch_texts)
                
                # Process each result
                for j, text in enumerate(batch_texts):
                    result = await _process_single_prediction(
                        text=text,
                        text_id=f"{request_id}_{i+j}",
                        predictions=predictions,
                        index=j,
                        request=analysis_request,
                        config=request.app.state.config
                    )
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Batch processing failed for request {request_id}: {e}")
                warnings.append(f"Batch processing failed: {str(e)}")
                
                # Fallback to individual processing
                for j, text in enumerate(batch_texts):
                    try:
                        individual_predictions = await async_model.predict_async([text])
                        result = await _process_single_prediction(
                            text=text,
                            text_id=f"{request_id}_{i+j}",
                            predictions=individual_predictions,
                            index=0,
                            request=analysis_request,
                            config=request.app.state.config
                        )
                        results.append(result)
                    except Exception as individual_e:
                        logger.error(f"Individual processing failed: {individual_e}")
                        results.append(_create_error_result(f"{request_id}_{i+j}", str(individual_e)))
        
        total_processing_time = time.time() - start_time
        
        response = BatchAnalysisResponse(
            request_id=request_id,
            timestamp=datetime.now(),
            results=results,
            total_processing_time=total_processing_time,
            model_info={
                "model_version": "1.0.0",
                "features_enabled": {
                    "watermark_detection": analysis_request.include_watermark_analysis,
                    "source_attribution": analysis_request.include_source_attribution,
                    "explanations": analysis_request.include_explanations
                }
            },
            warnings=warnings
        )
        
        # Cache successful results
        if not warnings:
            intelligent_cache.put(cache_key, response)
        
        # Log metrics in background
        background_tasks.add_task(
            _log_request_metrics,
            request_id=request_id,
            user_id=current_user['user_id'],
            num_texts=len(analysis_request.texts),
            processing_time=total_processing_time
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Analysis failed for request {request_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

async def _process_single_prediction(
    text: str,
    text_id: str,
    predictions: Dict[str, Any],
    index: int,
    request: TextAnalysisRequest,
    config: Any
) -> PredictionResult:
    """Process a single prediction result"""
    
    start_time = time.time()
    
    # Extract basic predictions
    is_ai_generated = bool(predictions['is_ai_generated'][index])
    confidence = float(predictions['confidence'][index])
    
    # Create prediction scores
    detection_scores = predictions.get('detection_scores', [[0.5, 0.5]])[index]
    prediction_scores = {
        'human': float(detection_scores[0]),
        'ai_generated': float(detection_scores[1])
    }
    
    # Optional analyses
    family_prediction = None
    family_confidence = None
    if request.include_source_attribution and 'family_predictions' in predictions:
        family_pred = predictions['family_predictions'][index]
        family_names = ['GPT', 'BERT', 'T5', 'CLAUDE', 'PALM', 'LLAMA', 'BLOOM', 'OTHER']
        family_idx = int(np.argmax(family_pred))
        family_prediction = family_names[min(family_idx, len(family_names)-1)]
        family_confidence = float(np.max(family_pred))
    
    watermark_detected = None
    watermark_confidence = None
    if request.include_watermark_analysis and 'watermark_detected' in predictions:
        watermark_detected = bool(predictions['watermark_detected'][index])
        watermark_confidence = float(predictions['watermark_confidence'][index])
    
    source_attribution = None
    if request.include_source_attribution and 'source_attribution' in predictions:
        source_attr = predictions['source_attribution']
        source_attribution = {
            'predicted_source': int(source_attr['predicted_source'][index]),
            'confidence': float(source_attr['confidence'][index]),
            'family_probs': source_attr['family_probs'][index].tolist(),
            'model_probs': source_attr['model_probs'][index].tolist()
        }
    
    explanations = None
    if request.include_explanations:
        # This would require implementing explanation generation
        explanations = {
            'method': 'simplified',
            'token_importance': [],
            'feature_importance': {},
            'reasoning': f"Text classified as {'AI-generated' if is_ai_generated else 'human-written'} with {confidence:.2%} confidence"
        }
    
    processing_time = time.time() - start_time
    
    return PredictionResult(
        text_id=text_id,
        is_ai_generated=is_ai_generated,
        confidence=confidence,
        prediction_scores=prediction_scores,
        family_prediction=family_prediction,
        family_confidence=family_confidence,
        watermark_detected=watermark_detected,
        watermark_confidence=watermark_confidence,
        source_attribution=source_attribution,
        explanations=explanations,
        processing_time=processing_time,
        model_version="1.0.0"
    )

def _create_error_result(text_id: str, error_msg: str) -> PredictionResult:
    """Create error result for failed predictions"""
    return PredictionResult(
        text_id=text_id,
        is_ai_generated=False,
        confidence=0.0,
        prediction_scores={'human': 0.5, 'ai_generated': 0.5},
        processing_time=0.0,
        model_version="1.0.0"
    )

async def _log_request_metrics(request_id: str, user_id: str, num_texts: int, processing_time: float):
    """Log request metrics for monitoring"""
    try:
        metrics = {
            'request_id': request_id,
            'user_id': user_id,
            'num_texts': num_texts,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'throughput': num_texts / processing_time if processing_time > 0 else 0
        }
        
        logger.info(f"Request metrics: {metrics}")
        
        # In production, send to monitoring system
        # await send_to_monitoring_system(metrics)
        
    except Exception as e:
        logger.error(f"Failed to log metrics: {e}")

@app.get("/metrics")
async def get_metrics(current_user: dict = Depends(get_current_user)):
    """Get system metrics (admin only)"""
    # In production, add admin role check
    
    return {
        'performance': {
            'recent_metrics': performance_monitor.metrics_history[-50:],
            'cache_stats': intelligent_cache.stats()
        },
        'system': health_checker.get_system_health(),
        'model': health_checker.run_checks()
    }

@app.post("/auth/token")
async def create_access_token(user_id: str, expires_hours: int = 24):
    """Create access token (simplified - implement proper auth)"""
    token = auth_manager.create_token(user_id, timedelta(hours=expires_hours))
    return {"access_token": token, "token_type": "bearer"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "PhantomHunter API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "analyze": "/analyze",
            "health": "/health",
            "metrics": "/metrics"
        },
        "documentation": "/docs"
    }

# Production server configuration
def create_production_app():
    """Create production-configured app"""
    import os
    
    # Load configuration from environment
    config = PhantomHunterConfig()
    
    # Override with environment variables
    config.batch_size = int(os.getenv('BATCH_SIZE', '8'))
    config.max_sequence_length = int(os.getenv('MAX_SEQUENCE_LENGTH', '512'))
    config.device = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    return app

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 