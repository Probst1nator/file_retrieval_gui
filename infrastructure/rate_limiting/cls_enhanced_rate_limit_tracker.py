import os
import json
import time
import random
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from core.globals import g

logger = logging.getLogger(__name__)

class LimitType(Enum):
    """Types of rate limits that can be tracked."""
    RPM = "requests_per_minute"      # Requests Per Minute
    TPM = "tokens_per_minute"        # Tokens Per Minute  
    RPD = "requests_per_day"         # Requests Per Day
    IPM = "images_per_minute"        # Images Per Minute
    GENERAL = "general_cooldown"     # General cooldown period

@dataclass
class RateLimit:
    """Represents a rate limit with its current state."""
    limit_type: LimitType
    limit_value: int                 # Maximum allowed (e.g., 300 for 300 RPM)
    current_usage: int = 0           # Current usage in this window
    window_start: float = 0          # When this window started (timestamp)
    window_duration: int = 60        # Window duration in seconds
    cooldown_until: float = 0        # If rate limited, cooldown until this timestamp
    retry_delay: float = 0           # Suggested retry delay from API
    
    def is_rate_limited(self) -> bool:
        """Check if currently in cooldown period."""
        return time.time() < self.cooldown_until
    
    def get_remaining_cooldown(self) -> float:
        """Get remaining cooldown time in seconds."""
        return max(0, self.cooldown_until - time.time())
    
    def is_window_expired(self) -> bool:
        """Check if the current tracking window has expired."""
        return time.time() - self.window_start >= self.window_duration
    
    def reset_window(self):
        """Reset the usage window."""
        self.current_usage = 0
        self.window_start = time.time()
    
    def can_make_request(self, cost: int = 1) -> bool:
        """Check if a request with given cost can be made."""
        if self.is_rate_limited():
            return False
        
        if self.is_window_expired():
            self.reset_window()
        
        return self.current_usage + cost <= self.limit_value
    
    def record_usage(self, cost: int = 1):
        """Record usage of this rate limit."""
        if self.is_window_expired():
            self.reset_window()
        self.current_usage += cost
    
    def apply_cooldown(self, cooldown_seconds: float, retry_delay: float = 0):
        """Apply a cooldown period."""
        self.cooldown_until = time.time() + cooldown_seconds
        self.retry_delay = retry_delay

class ExponentialBackoffManager:
    """Manages exponential backoff for rate limited requests."""
    
    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0, max_retries: int = 5):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.retry_counts: Dict[str, int] = {}
    
    def get_delay(self, model_key: str) -> float:
        """Get the delay for the next retry."""
        retry_count = self.retry_counts.get(model_key, 0)
        if retry_count >= self.max_retries:
            return -1  # No more retries
        
        # Exponential backoff: base_delay * (2 ^ retry_count) + jitter
        delay = min(self.base_delay * (2 ** retry_count), self.max_delay)
        jitter = random.uniform(0, delay * 0.1)  # 10% jitter
        return delay + jitter
    
    def record_retry(self, model_key: str):
        """Record a retry attempt."""
        self.retry_counts[model_key] = self.retry_counts.get(model_key, 0) + 1
    
    def reset_retries(self, model_key: str):
        """Reset retry count for successful request."""
        self.retry_counts.pop(model_key, None)
    
    def get_retry_count(self, model_key: str) -> int:
        """Get current retry count."""
        return self.retry_counts.get(model_key, 0)

class EnhancedRateLimitTracker:
    """
    Enhanced rate limit tracker supporting multi-dimensional limits
    (RPM, TPM, RPD, IPM) with exponential backoff and provider-specific parsing.
    """
    
    _instance: Optional["EnhancedRateLimitTracker"] = None
    
    def __new__(cls) -> "EnhancedRateLimitTracker":
        if cls._instance is None:
            cls._instance = super(EnhancedRateLimitTracker, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the enhanced rate limit tracker."""
        self.model_limits: Dict[str, Dict[LimitType, RateLimit]] = {}
        self.backoff_manager = ExponentialBackoffManager()
        self._load_limits()
    
    def _get_storage_path(self) -> str:
        """Get the storage path for rate limits."""
        return g.MODEL_RATE_LIMITS_PATH.replace('.json', '_enhanced.json')
    
    def _load_limits(self):
        """Load rate limits from persistent storage."""
        try:
            storage_path = self._get_storage_path()
            if os.path.exists(storage_path):
                with open(storage_path, 'r') as f:
                    data = json.load(f)
                
                # Convert loaded data back to RateLimit objects
                for model_key, limits_data in data.items():
                    self.model_limits[model_key] = {}
                    for limit_type_str, limit_data in limits_data.items():
                        limit_type = LimitType(limit_type_str)
                        rate_limit = RateLimit(
                            limit_type=limit_type,
                            limit_value=limit_data['limit_value'],
                            current_usage=limit_data.get('current_usage', 0),
                            window_start=limit_data.get('window_start', time.time()),
                            window_duration=limit_data.get('window_duration', 60),
                            cooldown_until=limit_data.get('cooldown_until', 0),
                            retry_delay=limit_data.get('retry_delay', 0)
                        )
                        self.model_limits[model_key][limit_type] = rate_limit
        except Exception as e:
            logger.error(f"Failed to load enhanced rate limits: {e}")
            self.model_limits = {}
    
    def _save_limits(self):
        """Save rate limits to persistent storage."""
        try:
            storage_path = self._get_storage_path()
            os.makedirs(os.path.dirname(storage_path), exist_ok=True)
            
            # Convert RateLimit objects to dictionaries
            data = {}
            for model_key, limits in self.model_limits.items():
                data[model_key] = {}
                for limit_type, rate_limit in limits.items():
                    data[model_key][limit_type.value] = asdict(rate_limit)
                    # Convert enum back to string for JSON serialization
                    data[model_key][limit_type.value]['limit_type'] = limit_type.value
            
            with open(storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save enhanced rate limits: {e}")
    
    def set_model_limits(self, model_key: str, limits_config: Dict[LimitType, int]):
        """
        Set rate limits for a model.
        
        Args:
            model_key: Model identifier
            limits_config: Dict mapping LimitType to limit values
        """
        if model_key not in self.model_limits:
            self.model_limits[model_key] = {}
        
        for limit_type, limit_value in limits_config.items():
            if limit_type in self.model_limits[model_key]:
                # Update existing limit value but preserve current state
                self.model_limits[model_key][limit_type].limit_value = limit_value
            else:
                # Create new rate limit
                window_duration = 60  # Default to 1 minute
                if limit_type == LimitType.RPD:
                    window_duration = 24 * 60 * 60  # 24 hours for daily limits
                
                self.model_limits[model_key][limit_type] = RateLimit(
                    limit_type=limit_type,
                    limit_value=limit_value,
                    window_duration=window_duration,
                    window_start=time.time()
                )
        
        self._save_limits()
    
    def can_make_request(self, model_key: str, token_cost: int = 0, image_count: int = 0) -> Tuple[bool, str]:
        """
        Check if a request can be made given the costs.
        
        Args:
            model_key: Model identifier
            token_cost: Number of tokens this request will consume
            image_count: Number of images this request will process
        
        Returns:
            Tuple of (can_make_request, reason_if_cannot)
        """
        if model_key not in self.model_limits:
            return True, ""
        
        limits = self.model_limits[model_key]
        
        # Check each applicable limit
        checks = [
            (LimitType.RPM, 1),           # Always costs 1 request
            (LimitType.RPD, 1),           # Always costs 1 request
            (LimitType.TPM, token_cost),  # Cost in tokens
            (LimitType.IPM, image_count), # Cost in images
            (LimitType.GENERAL, 0)        # General cooldown check
        ]
        
        for limit_type, cost in checks:
            if limit_type in limits:
                rate_limit = limits[limit_type]
                
                # Special handling for general cooldown
                if limit_type == LimitType.GENERAL:
                    if rate_limit.is_rate_limited():
                        remaining = rate_limit.get_remaining_cooldown()
                        return False, f"General cooldown active for {remaining:.1f}s"
                    continue
                
                # Skip if no cost for this limit type
                if cost == 0:
                    continue
                
                if not rate_limit.can_make_request(cost):
                    if rate_limit.is_rate_limited():
                        remaining = rate_limit.get_remaining_cooldown()
                        return False, f"{limit_type.value} cooldown active for {remaining:.1f}s"
                    else:
                        return False, f"{limit_type.value} limit would be exceeded ({rate_limit.current_usage + cost} > {rate_limit.limit_value})"
        
        return True, ""
    
    def record_request(self, model_key: str, token_cost: int = 0, image_count: int = 0):
        """
        Record a successful request and its costs.
        
        Args:
            model_key: Model identifier
            token_cost: Number of tokens consumed
            image_count: Number of images processed
        """
        if model_key not in self.model_limits:
            return
        
        limits = self.model_limits[model_key]
        
        # Record usage for applicable limits
        usage_records = [
            (LimitType.RPM, 1),
            (LimitType.RPD, 1),
            (LimitType.TPM, token_cost),
            (LimitType.IPM, image_count)
        ]
        
        for limit_type, cost in usage_records:
            if limit_type in limits and cost > 0:
                limits[limit_type].record_usage(cost)
        
        # Reset backoff on successful request
        self.backoff_manager.reset_retries(model_key)
        self._save_limits()
    
    def apply_rate_limit(self, model_key: str, limit_type: LimitType, cooldown_seconds: float, retry_delay: float = 0):
        """
        Apply a rate limit cooldown to a specific limit type.
        
        Args:
            model_key: Model identifier
            limit_type: Type of limit that was exceeded
            cooldown_seconds: How long to wait before trying again
            retry_delay: Suggested retry delay from API
        """
        if model_key not in self.model_limits:
            self.model_limits[model_key] = {}
        
        if limit_type not in self.model_limits[model_key]:
            # Create a default rate limit if it doesn't exist
            self.model_limits[model_key][limit_type] = RateLimit(
                limit_type=limit_type,
                limit_value=999999,  # Unknown limit
                window_start=time.time()
            )
        
        self.model_limits[model_key][limit_type].apply_cooldown(cooldown_seconds, retry_delay)
        self.backoff_manager.record_retry(model_key)
        self._save_limits()
    
    def get_suggested_retry_delay(self, model_key: str) -> float:
        """
        Get suggested retry delay combining API suggestions and exponential backoff.
        
        Returns:
            Delay in seconds, or -1 if max retries exceeded
        """
        # Check if any rate limits have specific retry delays
        if model_key in self.model_limits:
            max_api_delay = 0
            for rate_limit in self.model_limits[model_key].values():
                if rate_limit.is_rate_limited() and rate_limit.retry_delay > 0:
                    max_api_delay = max(max_api_delay, rate_limit.retry_delay)
            
            if max_api_delay > 0:
                return max_api_delay
        
        # Fall back to exponential backoff
        return self.backoff_manager.get_delay(model_key)
    
    def get_status_summary(self, model_key: str) -> Dict[str, Any]:
        """Get a summary of current rate limit status for a model."""
        if model_key not in self.model_limits:
            return {"status": "no_limits_configured"}
        
        summary = {
            "status": "ok",
            "limits": {},
            "retry_count": self.backoff_manager.get_retry_count(model_key)
        }
        
        for limit_type, rate_limit in self.model_limits[model_key].items():
            summary["limits"][limit_type.value] = {
                "limit": rate_limit.limit_value,
                "usage": rate_limit.current_usage,
                "window_remaining": max(0, rate_limit.window_duration - (time.time() - rate_limit.window_start)),
                "is_rate_limited": rate_limit.is_rate_limited(),
                "cooldown_remaining": rate_limit.get_remaining_cooldown()
            }
            
            if rate_limit.is_rate_limited():
                summary["status"] = "rate_limited"
        
        return summary
    
    def reset_model_limits(self, model_key: str):
        """Reset all rate limits and backoff state for a specific model."""
        if model_key in self.model_limits:
            del self.model_limits[model_key]
        self.backoff_manager.reset_retries(model_key)
        self._save_limits()
        logger.info(f"Reset rate limits for {model_key}")
    
    def clear_all_cooldowns(self, model_key: str):
        """Clear all active cooldowns for a model without resetting limits."""
        if model_key in self.model_limits:
            for rate_limit in self.model_limits[model_key].values():
                rate_limit.cooldown_until = 0
                rate_limit.retry_delay = 0
            self.backoff_manager.reset_retries(model_key)
            self._save_limits()
            logger.info(f"Cleared cooldowns for {model_key}")

# Create singleton instance
enhanced_rate_limit_tracker = EnhancedRateLimitTracker()