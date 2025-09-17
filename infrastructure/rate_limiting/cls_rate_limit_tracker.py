import os
import json
import time
import logging
from typing import Dict, Any, Optional
from core.globals import g

logger = logging.getLogger(__name__)

class RateLimitTracker:
    """
    A class for tracking rate limits for models.
    Maintains a file with model rate limit information including:
    - Model name
    - Try again seconds (cooldown period)
    - Entry time (when the rate limit was hit)
    """
    
    _instance: Optional["RateLimitTracker"] = None
    _rate_limits: Dict[str, Dict[str, Any]] = {}
    
    def __new__(cls) -> "RateLimitTracker":
        """
        Create a singleton instance of RateLimitTracker.
        
        Returns:
            RateLimitTracker: The singleton instance.
        """
        if cls._instance is None:
            cls._instance = super(RateLimitTracker, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """Initialize the RateLimitTracker instance by loading existing rate limits."""
        self._rate_limits = self._load_rate_limits()
    
    def _load_rate_limits(self) -> Dict[str, Dict[str, Any]]:
        """
        Load rate limits from the persistent storage file.
        
        Returns:
            Dict[str, Dict[str, Any]]: Rate limits data keyed by model name.
        """
        try:
            if os.path.exists(g.MODEL_RATE_LIMITS_PATH):
                with open(g.MODEL_RATE_LIMITS_PATH, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to load model rate limits: {e}")
            return {}
    
    def _save_rate_limits(self) -> None:
        """Save the current rate limits to the persistent storage file."""
        try:
            os.makedirs(os.path.dirname(g.MODEL_RATE_LIMITS_PATH), exist_ok=True)
            with open(g.MODEL_RATE_LIMITS_PATH, 'w') as f:
                json.dump(self._rate_limits, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model rate limits: {e}")
    
    def update_rate_limit(self, model: str, try_again_seconds: float) -> None:
        """
        Update the rate limit information for a model.
        
        Args:
            model (str): The model identifier.
            try_again_seconds (float): The cooldown period in seconds.
        """
        self._rate_limits[model] = {
            "try_again_seconds": try_again_seconds,
            "entry_time": time.time()
        }
        self._save_rate_limits()
        logger.debug(f"Rate limited {model}: {try_again_seconds}s cooldown")
    
    def is_rate_limited(self, model: str) -> bool:
        """
        Check if a model is currently rate limited.
        
        Args:
            model (str): The model identifier.
            
        Returns:
            bool: True if the model is rate limited, False otherwise.
        """
        if model not in self._rate_limits:
            return False
        
        rate_limit_info = self._rate_limits[model]
        current_time = time.time()
        entry_time = rate_limit_info["entry_time"]
        try_again_seconds = rate_limit_info["try_again_seconds"]
        
        # Check if enough time has passed since the rate limit was hit
        if current_time >= entry_time + try_again_seconds:
            # Rate limit has expired, remove it
            del self._rate_limits[model]
            self._save_rate_limits()
            return False
        
        # Still rate limited
        return True
    
    def get_remaining_time(self, model: str) -> float:
        """
        Get the remaining time in seconds until the rate limit expires.
        
        Args:
            model (str): The model identifier.
            
        Returns:
            float: The remaining time in seconds. Returns 0 if the model is not rate limited.
        """
        if not self.is_rate_limited(model):
            return 0
        
        rate_limit_info = self._rate_limits[model]
        current_time = time.time()
        entry_time = rate_limit_info["entry_time"]
        try_again_seconds = rate_limit_info["try_again_seconds"]
        
        return max(0, (entry_time + try_again_seconds) - current_time)

# Create singleton instance
rate_limit_tracker = RateLimitTracker() 