import re
import logging
from typing import Dict, Tuple, Optional
from abc import ABC, abstractmethod
from infrastructure.rate_limiting.cls_enhanced_rate_limit_tracker import LimitType

logger = logging.getLogger(__name__)

class RateLimitParser(ABC):
    """Abstract base class for provider-specific rate limit error parsing."""
    
    @abstractmethod
    def parse_error(self, error_message: str) -> Tuple[LimitType, float, Dict[str, any]]:
        """
        Parse a rate limit error message.
        
        Args:
            error_message: The error message from the API
            
        Returns:
            Tuple of (limit_type, cooldown_seconds, extra_info)
        """
        pass
    
    @abstractmethod
    def get_default_limits(self) -> Dict[LimitType, int]:
        """Get default rate limits for this provider."""
        pass

class GeminiRateLimitParser(RateLimitParser):
    """Parser for Google Gemini API rate limit errors."""
    
    def parse_error(self, error_message: str) -> Tuple[LimitType, float, Dict[str, any]]:
        """
        Parse Gemini API rate limit error messages.
        
        Gemini returns errors like:
        - "429 Quota exceeded for requests per minute (RPM)"
        - "429 Resource has been exhausted (e.g. check quota)"
        - "retry_delay { seconds: 60 }"
        """
        error_lower = error_message.lower()
        extra_info = {}
        
        # Extract retry delay if present
        retry_delay = 60  # Default
        retry_matches = re.findall(r"retry_delay\s*\{\s*seconds:\s*(\d+)", error_message)
        if retry_matches:
            retry_delay = int(retry_matches[0])
            extra_info['api_suggested_delay'] = retry_delay
        
        # Determine limit type based on error message
        if "requests per minute" in error_lower or "rpm" in error_lower:
            return LimitType.RPM, retry_delay, extra_info
        elif "tokens per minute" in error_lower or "tpm" in error_lower:
            return LimitType.TPM, retry_delay, extra_info
        elif "requests per day" in error_lower or "rpd" in error_lower:
            return LimitType.RPD, retry_delay, extra_info
        elif "images per minute" in error_lower or "ipm" in error_lower:
            return LimitType.IPM, retry_delay, extra_info
        elif "quota" in error_lower and "exceed" in error_lower:
            # Generic quota exceeded - likely RPM
            return LimitType.RPM, retry_delay, extra_info
        else:
            # Unknown type, use general cooldown
            return LimitType.GENERAL, retry_delay, extra_info
    
    def get_default_limits(self) -> Dict[LimitType, int]:
        """
        Get default Gemini API limits.
        These are conservative estimates for free tier.
        """
        return {
            LimitType.RPM: 15,      # Conservative estimate for free tier
            LimitType.TPM: 32000,   # Free tier TPM
            LimitType.RPD: 25       # Free tier daily requests
        }
    
    def parse_tier_from_error(self, error_message: str) -> Optional[str]:
        """Try to determine user's tier from error message."""
        # This would require more sophisticated parsing based on actual error messages
        # For now, assume free tier
        return "free"

class OllamaRateLimitParser(RateLimitParser):
    """Parser for Ollama rate limit errors (usually simpler)."""
    
    def parse_error(self, error_message: str) -> Tuple[LimitType, float, Dict[str, any]]:
        """
        Parse Ollama rate limit errors.
        
        Ollama typically has simpler rate limiting, often just connection limits.
        """
        error_lower = error_message.lower()
        extra_info = {}
        
        if "too many requests" in error_lower:
            return LimitType.RPM, 30.0, extra_info  # 30 second cooldown
        elif "rate limit" in error_lower:
            # Extract any numeric delays
            delay_matches = re.findall(r"(\d+)\s*second", error_message)
            delay = int(delay_matches[0]) if delay_matches else 60
            return LimitType.GENERAL, float(delay), extra_info
        else:
            # Generic rate limit
            return LimitType.GENERAL, 60.0, extra_info
    
    def get_default_limits(self) -> Dict[LimitType, int]:
        """
        Ollama limits are typically much higher or unlimited locally.
        These are conservative estimates for hosted instances.
        """
        return {
            LimitType.RPM: 100,   # Usually much higher for local
            LimitType.TPM: 50000  # Token limits are typically generous
        }

class GenericRateLimitParser(RateLimitParser):
    """Generic fallback parser for unknown providers."""
    
    def parse_error(self, error_message: str) -> Tuple[LimitType, float, Dict[str, any]]:
        """Generic parsing for unknown providers."""
        error_lower = error_message.lower()
        extra_info = {}
        
        # Look for common patterns
        if "minute" in error_lower:
            return LimitType.RPM, 60.0, extra_info
        elif "hour" in error_lower:
            return LimitType.GENERAL, 3600.0, extra_info
        elif "day" in error_lower:
            return LimitType.RPD, 300.0, extra_info  # 5 minute cooldown for daily limits
        else:
            # Default to general cooldown
            return LimitType.GENERAL, 60.0, extra_info
    
    def get_default_limits(self) -> Dict[LimitType, int]:
        """Conservative generic limits."""
        return {
            LimitType.RPM: 60,
            LimitType.TPM: 10000,
            LimitType.RPD: 100
        }

class RateLimitParserFactory:
    """Factory for creating appropriate rate limit parsers."""
    
    _parsers = {
        'google': GeminiRateLimitParser,
        'gemini': GeminiRateLimitParser,
        'ollama': OllamaRateLimitParser,
        'openai': GenericRateLimitParser,  # Could be specialized later
        'anthropic': GenericRateLimitParser,  # Could be specialized later
        'groq': GenericRateLimitParser,   # Could be specialized later
    }
    
    @classmethod
    def get_parser(cls, provider_name: str) -> RateLimitParser:
        """
        Get appropriate parser for a provider.
        
        Args:
            provider_name: Name of the provider (case-insensitive)
            
        Returns:
            RateLimitParser instance
        """
        provider_key = provider_name.lower()
        
        # Try exact match first
        if provider_key in cls._parsers:
            return cls._parsers[provider_key]()
        
        # Try partial matches
        for key, parser_class in cls._parsers.items():
            if key in provider_key:
                return parser_class()
        
        # Default to generic parser
        return GenericRateLimitParser()
    
    @classmethod
    def register_parser(cls, provider_name: str, parser_class: type):
        """Register a new parser for a provider."""
        cls._parsers[provider_name.lower()] = parser_class