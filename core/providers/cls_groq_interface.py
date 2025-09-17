import os
from typing import Optional, Tuple, Any, TYPE_CHECKING, List, Dict
from groq import Groq
from termcolor import colored
from core.chat import Chat
from py_classes.unified_interfaces import AIProviderInterface
from infrastructure.rate_limiting.cls_rate_limit_tracker import rate_limit_tracker
import socket
import logging
import requests
from core.globals import g

# Only used for type annotations
if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Define custom exception classes to differentiate error types
class TimeoutException(Exception):
    """Exception raised when a request to Groq API times out."""
    pass

class RateLimitException(Exception):
    """Exception raised when Groq API rate limits are exceeded."""
    pass

class GroqAPI(AIProviderInterface):
    """
    Implementation of the AIProviderInterface for the Groq API.
    """

    @staticmethod
    def generate_response(chat: Chat, model_key: str, temperature: float = 0, silent_reason: str = "", thinking_budget: Optional[int] = None) -> Any:
        """
        Generates a response using the Groq API.
        Args:
            chat (Chat): The chat object containing messages or a string prompt.
            model_key (str): The model identifier.
            temperature (float): The temperature setting for the model.
            silent_reason (str): Reason for silence if applicable.
        Returns:
            Any: A stream object that yields response chunks.
            
        Raises:
            RateLimitException: If the model is rate limited.
            Exception: For other issues, to be handled by the router.
        """
        # Check if the model is rate limited
        if rate_limit_tracker.is_rate_limited(model_key):
            remaining_time = rate_limit_tracker.get_remaining_time(model_key)
            rate_limit_reason = f"rate limited (wait {remaining_time:.1f}s)"
            
            if not silent_reason:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(f"Groq-Api: {colored('<', 'yellow')}{colored(model_key, 'yellow')}{colored('>', 'yellow')} is {colored(rate_limit_reason, 'yellow')}", force_print=True, prefix=prefix)
            
            # Raise a rate limit exception to be handled by the router
            raise RateLimitException(f"Model {model_key} is rate limited. Try again in {remaining_time:.1f} seconds")

        # Configure the client (let any error here bubble up to the router)
        client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        
        # Informational logging (not error handling)
        if silent_reason:
            temp_str = "" if temperature == 0 or temperature == None else f" at temperature {temperature}"
            prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
            g.debug_log(f"Groq-Api: {colored('<', 'green')}{colored(model_key, 'green')}{colored('>', 'green')} is {colored('silently', 'green')} generating response{temp_str}...", force_print=True, prefix=prefix)
        else:
            temp_str = "" if temperature == 0 or temperature == None else f" at temperature {temperature}"
            prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
            g.debug_log(f"Groq-Api: {colored('<', 'green')}{colored(model_key, 'green')}{colored('>', 'green')} is generating response{temp_str}...", "green", force_print=True, prefix=prefix)

        # Let any errors here bubble up to the router for centralized handling
        return client.chat.completions.create(
            model=model_key,
            messages=chat.to_groq(),
            temperature=temperature,
            stream=True
        )

    @staticmethod
    def get_available_models(chat: Optional[Chat] = None) -> List[Dict[str, Any]]:
        """
        Discovers and returns available Groq models.
        
        This method queries the Groq API to get a real-time list of available models.
        Can be used for dynamic model discovery in the future.
        Currently, the LLM router uses a static list of known Groq models for 
        performance reasons, but this method provides the foundation for dynamic discovery.
        
        Args:
            chat (Optional[Chat]): The chat object for debug printing.
            
        Returns:
            List[Dict[str, Any]]: List of available model information dictionaries.
                Each dictionary contains:
                - id: Model identifier (e.g., "llama-3.3-70b-versatile")
                - object: Object type (typically "model")
                - created: Creation timestamp
                - owned_by: Model developer/owner
                - root: Root model identifier
                - parent: Parent model (if applicable)
                - context_window: Maximum context window size
        """
        try:
            # Check if API key is available
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                error_msg = "Groq API key not found. Set the GROQ_API_KEY environment variable."
                if chat:
                    prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                    g.debug_log(f"Groq-Api: {error_msg}", "red", is_error=True, prefix=prefix)
                logger.error(error_msg)
                return []
            
            # Print status message
            prefix = chat.get_debug_title_prefix() if chat and hasattr(chat, 'get_debug_title_prefix') else ""
            if chat:
                g.debug_log(f"Groq-Api: {colored('<', 'green')}{colored('model-discovery', 'green')}{colored('>', 'green')} discovering available models...", force_print=True, prefix=prefix)
            
            # Make request to Groq models endpoint
            url = "https://api.groq.com/openai/v1/models"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            models_data = response.json()
            models = []
            
            if "data" in models_data:
                for model in models_data["data"]:
                    model_info = {
                        'id': model.get('id', ''),
                        'object': model.get('object', ''),
                        'created': model.get('created', 0),
                        'owned_by': model.get('owned_by', ''),
                        'root': model.get('root', ''),
                        'parent': model.get('parent'),
                        'context_window': model.get('context_window'),
                        'max_model_len': model.get('max_model_len'),  # Alternative context window field
                        'description': model.get('description', ''),
                        'pricing': model.get('pricing', {}),
                        'supported_modalities': model.get('supported_modalities', [])
                    }
                    models.append(model_info)
                
                if chat:
                    g.debug_log(f"Groq-Api: {colored('<', 'green')}{colored('model-discovery', 'green')}{colored('>', 'green')} found {len(models)} available models", force_print=True, prefix=prefix)
                
                return models
            else:
                error_msg = "No 'data' field found in Groq API response"
                if chat:
                    g.debug_log(f"Groq-Api: {error_msg}", "yellow", prefix=prefix)
                logger.warning(error_msg)
                return []
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Groq API model discovery request error: {e}"
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(error_msg, "red", is_error=True, prefix=prefix)
            logger.error(error_msg)
            return []
        except ValueError as e:
            error_msg = f"Groq API model discovery JSON parsing error: {e}"
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(error_msg, "red", is_error=True, prefix=prefix)
            logger.error(error_msg)
            return []
        except Exception as e:
            error_msg = f"Groq API model discovery error: {e}"
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(error_msg, "red", is_error=True, prefix=prefix)
            logger.error(error_msg)
            return []

    @staticmethod
    def transcribe_audio(filepath: str, model: str = "whisper-large-v3-turbo", language: Optional[str] = None, silent_reason: str = False, chat: Optional['Chat'] = None) -> Optional[Tuple[str, str]]:
        """
        Transcribes an audio file using Groq's Whisper implementation.
        
        Args:
            filepath (str): The path to the audio file.
            model (str): The Whisper model to use.
            language (str): The language of the audio (default: "auto" for auto-detection).
            silent_reason (str): Reason for silence if applicable.
            chat (Optional[Chat]): Chat object for debug printing with title.
            
        Returns:
            Optional[Tuple[str, str]]: A tuple containing the transcribed text and detected language.
            
        Raises:
            RateLimitException: If the model is rate limited.
            TimeoutException: If the request times out.
            Exception: For other errors.
        """
        # Check if the model is rate limited
        if rate_limit_tracker.is_rate_limited(model):
            remaining_time = rate_limit_tracker.get_remaining_time(model)
            rate_limit_reason = f"rate limited (wait {remaining_time:.1f}s)"
            
            if not silent_reason and chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(f"Groq-Api: {colored('<', 'yellow')}{colored(model, 'yellow')}{colored('>', 'yellow')} is {colored(rate_limit_reason, 'yellow')}", force_print=True, prefix=prefix)
            
            # Raise a silent rate limit exception
            raise RateLimitException(f"Model {model} is rate limited. Try again in {remaining_time:.1f} seconds")
        
        try:
            client = Groq(api_key=os.getenv('GROQ_API_KEY'), timeout=3.0, max_retries=0)
            
            if not silent_reason and chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(f"Groq-Api: Transcribing audio using {colored('<', 'green')}{colored(model, 'green')}{colored('>', 'green')}...", force_print=True, prefix=prefix)
            
            with open(filepath, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(filepath, file.read()),
                    model=model,
                    response_format="verbose_json",
                    language=language
                )
            
            if not silent_reason and chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(colored("Transcription complete.", 'green'), force_print=True, prefix=prefix)
            
            return transcription.text, transcription.language

        except (socket.timeout, socket.error, TimeoutError) as e:
            # Handle timeout-related errors silently without printing
            raise TimeoutException(f"❌ Request timed out: {e}")
        except Exception as e:
            # Check if this is a rate limit error (429)
            error_str = str(e)
            if "Error code: 429" in error_str or "rate_limit_exceeded" in error_str:
                try:
                    try_again_seconds = int(error_str.split("Please try again in")[1].split("ms")[0].strip()) / 1000
                except (IndexError, ValueError):
                    # If parsing fails, use a default value
                    try_again_seconds = 60.0  # Default to 60 seconds if parsing fails
                    logger.warning(f"Failed to parse rate limit wait time, using default of {try_again_seconds}s")
                
                # Update the rate limit tracker
                rate_limit_tracker.update_rate_limit(model, try_again_seconds)
                
                # Handle rate limit errors silently
                raise RateLimitException(f"❌ Rate limit exceeded: {e}")
            
            # Handle other errors without printing
            error_msg = f"Groq-Api audio transcription error: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

    # The token counting method is commented out in your original code,
    # so I'm leaving it as is.
    
