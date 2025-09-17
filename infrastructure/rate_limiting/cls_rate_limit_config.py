"""
Rate limit configuration and initialization for different providers and models.
"""

import logging
from infrastructure.rate_limiting.cls_enhanced_rate_limit_tracker import enhanced_rate_limit_tracker, LimitType

logger = logging.getLogger(__name__)

class RateLimitConfig:
    """Manages rate limit configuration for different models and providers."""
    
    # Known model configurations
    MODEL_CONFIGS = {
        # Gemini Models - Free Tier (conservative estimates)
        'gemini-1.5-pro': {
            LimitType.RPM: 2,
            LimitType.TPM: 32000,
            LimitType.RPD: 50
        },
        'gemini-1.5-flash': {
            LimitType.RPM: 15,
            LimitType.TPM: 1000000,
            LimitType.RPD: 1500
        },
        'gemini-2.5-flash': {
            LimitType.RPM: 15,
            LimitType.TPM: 1000000,
            LimitType.RPD: 1500
        },
        'gemini-1.5-pro-latest': {
            LimitType.RPM: 2,
            LimitType.TPM: 32000,
            LimitType.RPD: 50
        },
        'gemini-1.5-flash-latest': {
            LimitType.RPM: 15,
            LimitType.TPM: 1000000,
            LimitType.RPD: 1500
        },
        # Additional Gemini model variants
        'gemini-2.5-flash-preview-05-20': {
            LimitType.RPM: 15,
            LimitType.TPM: 1000000,
            LimitType.RPD: 1500
        },
        'gemini-flash': {
            LimitType.RPM: 15,
            LimitType.TPM: 1000000,
            LimitType.RPD: 1500
        },
        'gemini-pro': {
            LimitType.RPM: 2,
            LimitType.TPM: 32000,
            LimitType.RPD: 50
        },
        
        # Gemini Models - Paid Tier 1 (with billing enabled)
        'gemini-1.5-pro-paid': {
            LimitType.RPM: 360,
            LimitType.TPM: 4000000,
            LimitType.RPD: 10000
        },
        'gemini-1.5-flash-paid': {
            LimitType.RPM: 1000,
            LimitType.TPM: 4000000,
            LimitType.RPD: 10000
        },
        
        # Ollama Models (typically local, high limits)
        'ollama-default': {
            LimitType.RPM: 100,
            LimitType.TPM: 100000
        },
        
        # Generic defaults for unknown models
        'unknown-free': {
            LimitType.RPM: 10,
            LimitType.TPM: 10000,
            LimitType.RPD: 100
        },
        'unknown-paid': {
            LimitType.RPM: 100,
            LimitType.TPM: 100000,
            LimitType.RPD: 1000
        }
    }
    
    @classmethod
    def initialize_model_limits(cls, model_key: str, is_paid: bool = False, provider: str = "unknown"):
        """
        Initialize rate limits for a model if not already configured.
        
        Args:
            model_key: The model identifier
            is_paid: Whether the user has a paid account
            provider: The provider name (for fallback defaults)
        """
        # Check if already configured
        status = enhanced_rate_limit_tracker.get_status_summary(model_key)
        if status["status"] != "no_limits_configured":
            logger.debug(f"Rate limits already configured for {model_key}")
            return
        
        # Try to find specific configuration
        config = None
        
        # Exact match first
        if model_key in cls.MODEL_CONFIGS:
            config = cls.MODEL_CONFIGS[model_key]
        # Try with paid suffix if paid account
        elif is_paid and f"{model_key}-paid" in cls.MODEL_CONFIGS:
            config = cls.MODEL_CONFIGS[f"{model_key}-paid"]
        # Pattern matching for Gemini models
        elif provider.lower() in ["gemini", "google"]:
            if "flash" in model_key.lower():
                config = cls.MODEL_CONFIGS['gemini-flash']  # Use flash defaults
                logger.debug(f"Using Gemini Flash defaults for {model_key}")
            elif "pro" in model_key.lower():
                config = cls.MODEL_CONFIGS['gemini-pro']  # Use pro defaults  
                logger.debug(f"Using Gemini Pro defaults for {model_key}")
            else:
                # Default to flash for unknown Gemini models
                config = cls.MODEL_CONFIGS['gemini-flash']
                logger.debug(f"Using Gemini Flash defaults for unknown Gemini model: {model_key}")
        # Provider defaults
        elif provider.lower() == "ollama":
            config = cls.MODEL_CONFIGS['ollama-default']
        # Generic fallbacks
        elif is_paid:
            config = cls.MODEL_CONFIGS['unknown-paid']
        else:
            config = cls.MODEL_CONFIGS['unknown-free']
        
        if config:
            enhanced_rate_limit_tracker.set_model_limits(model_key, config)
            logger.info(f"Initialized rate limits for {model_key}: {config}")
        else:
            logger.warning(f"No rate limit configuration found for {model_key}")
    
    @classmethod
    def configure_gemini_defaults(cls, is_paid: bool = False):
        """Configure default limits for all Gemini models."""
        gemini_models = [
            'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-2.5-flash',
            'gemini-1.5-pro-latest', 'gemini-1.5-flash-latest',
            'gemini-pro', 'gemini-flash'  # Legacy names
        ]
        
        for model in gemini_models:
            cls.initialize_model_limits(model, is_paid=is_paid, provider="gemini")
    
    @classmethod
    def configure_ollama_defaults(cls):
        """Configure default limits for Ollama models."""
        # Ollama models are typically local, so we use generous defaults
        # Individual models will be configured as they're encountered
        pass
    
    @classmethod
    def auto_detect_tier(cls, model_key: str, error_message: str = "") -> bool:
        """
        Try to auto-detect if user has paid tier based on error messages.
        
        Args:
            model_key: The model that had an error
            error_message: The error message from the API
            
        Returns:
            True if paid tier is detected, False otherwise
        """
        # This is a heuristic approach - could be improved with more data
        error_lower = error_message.lower()
        
        # High RPM in error suggests paid tier
        if "300 rpm" in error_lower or "1000 rpm" in error_lower:
            return True
        
        # High TPM suggests paid tier
        if "4000000 tpm" in error_lower or "4m tpm" in error_lower:
            return True
        
        # High daily limits suggest paid tier
        if "10000 rpd" in error_lower or "10k rpd" in error_lower:
            return True
        
        # Default to free tier
        return False
    
    @classmethod
    def update_limits_from_error(cls, model_key: str, error_message: str):
        """
        Update model limits based on information in error message.
        
        Args:
            model_key: The model that had an error
            error_message: The error message from the API
        """
        # Auto-detect tier and reconfigure if needed
        is_paid = cls.auto_detect_tier(model_key, error_message)
        
        # Get current status
        status = enhanced_rate_limit_tracker.get_status_summary(model_key)
        
        # If no limits configured or if we detected a tier upgrade
        if status["status"] == "no_limits_configured" or is_paid:
            cls.initialize_model_limits(model_key, is_paid=is_paid, provider="gemini")
            logger.info(f"Updated {model_key} limits based on error - paid tier: {is_paid}")

# Initialize common models on import
def initialize_common_models():
    """Initialize rate limits for commonly used models."""
    config = RateLimitConfig()
    
    # Configure Gemini defaults (assume free tier initially)
    config.configure_gemini_defaults(is_paid=False)
    
    # Configure Ollama defaults
    config.configure_ollama_defaults()

# Auto-initialize on import
try:
    initialize_common_models()
    logger.info("Initialized common model rate limits")
except Exception as e:
    logger.warning(f"Failed to initialize common model rate limits: {e}")