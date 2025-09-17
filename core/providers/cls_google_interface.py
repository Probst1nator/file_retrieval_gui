import os
from typing import Optional, List, Dict, Any, Union
import logging
from termcolor import colored
import google.generativeai as genai
from google.generativeai import types
from core.chat import Chat, Role
from py_classes.unified_interfaces import AIProviderInterface
from infrastructure.rate_limiting.cls_enhanced_rate_limit_tracker import enhanced_rate_limit_tracker
from infrastructure.rate_limiting.cls_rate_limit_parsers import RateLimitParserFactory
from infrastructure.rate_limiting.cls_rate_limit_config import RateLimitConfig
from core.globals import g
# Import audio utility functions
from shared.utils_audio import save_binary_file, convert_to_wav
import mimetypes
# Import exceptions from Groq interface to avoid duplication
from core.providers.cls_groq_interface import RateLimitException

logger = logging.getLogger(__name__)

# Flag to track if API has been configured
_gemini_api_configured = False


class GoogleAPI(AIProviderInterface):
    """
    Implementation of the AIProviderInterface for the Google Gemini API.
    
    This class provides methods to interact with Google's Gemini models using 
    the Google Generative AI client library.
    """
    
    @staticmethod
    def _configure_api() -> None:
        """
        Configure the Google Generative AI API with the API key.
        
        Raises:
            Exception: If the API key is missing or invalid.
        """
        global _gemini_api_configured
        
        # Only configure if not already done
        if not _gemini_api_configured:
            api_key: Optional[str] = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise Exception("Google API key not found. Set the GEMINI_API_KEY environment variable.")
            
            try:
                genai.configure(api_key=api_key)
                _gemini_api_configured = True
            except Exception as e:
                _gemini_api_configured = False
                raise Exception(f"Failed to configure Google API: {e}")

    @staticmethod
    def generate_response(
        chat: Union[Chat, str], 
        model_key: str = "gemini-1.5-pro-latest", 
        temperature: float = 0, 
        silent_reason: str = "", 
        thinking_budget: Optional[int] = None
    ) -> Any:  # Return type changed to Any to avoid circular imports
        if g.VERBOSE_DEBUG:
            print(f"🔍 GOOGLE API ENTRY: model={model_key}, temp={temperature}, chat_type={type(chat).__name__}")
            if hasattr(chat, 'messages'):
                print(f"🔍 CHAT MESSAGES: {len(chat.messages)} messages")
                for i, (role, content) in enumerate(chat.messages):
                    print(f"🔍   MSG {i}: {role.value} -> {len(str(content))} chars")
            if hasattr(chat, 'base64_images'):
                print(f"🔍 BASE64 IMAGES: {len(chat.base64_images)} images")
                for i, img in enumerate(chat.base64_images):
                    print(f"🔍   IMG {i}: {len(img)} chars, starts with: {img[:50]}...")
            print(f"🔍 SILENT REASON: '{silent_reason}'")
        """
        Generates a response using the Google Gemini API.
        
        Args:
            chat (Union[Chat, str]): The chat object containing messages or a string prompt.
            model_key (str): The model identifier (defaults to gemini-1.5-pro-latest).
            temperature (float): The temperature setting for the model (0.0 to 1.0).
            silent_reason (str): Reason for silence if applicable.
            thinking_budget (Optional[int]): Token budget for model's internal reasoning process.
                                           Use -1 for dynamic, 0 to disable, or positive integer for fixed budget.
            
        Returns:
            Iterator[GenerateContentResponse]: A stream of response chunks from the Gemini API.
            
        Raises:
            RateLimitException: If the model is rate limited.
            TimeoutException: If the request times out.
            Exception: For other errors, to be handled by the router.
        """
        # Convert string to Chat object if needed
        if isinstance(chat, str):
            chat_obj = Chat()
            chat_obj.add_message(Role.USER, chat)
            chat = chat_obj

        # Get the prefix for debug logging
        prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""

        # Initialize rate limits for this model if not already done
        RateLimitConfig.initialize_model_limits(model_key, is_paid=False, provider="gemini")

        # Estimate token cost (rough approximation)
        token_cost = 0
        image_count = 0
        if isinstance(chat, Chat):
            # Rough token estimation: 4 chars per token
            text_content = str(chat)
            token_cost = len(text_content) // 4
            # Count images in chat
            for role, content in chat.messages:
                if isinstance(content, dict) and 'images' in content:
                    image_count += len(content['images'])
        
        # Check if request can be made with enhanced rate limiting
        can_make_request, reason = enhanced_rate_limit_tracker.can_make_request(
            model_key, token_cost=token_cost, image_count=image_count
        )
        
        if not can_make_request:
            if not silent_reason:
                g.debug_log(f"Google-Api: {colored('<', 'yellow')}{colored(model_key, 'yellow')}{colored('>', 'yellow')} is {colored(f'rate limited ({reason})', 'yellow')}", force_print=True, prefix=prefix)
            
            # Get suggested retry delay
            retry_delay = enhanced_rate_limit_tracker.get_suggested_retry_delay(model_key)
            if retry_delay == -1:
                raise RateLimitException(f"Model {model_key} exceeded maximum retry attempts")
            else:
                raise RateLimitException(f"Model {model_key} is rate limited: {reason}. Retry in {retry_delay:.1f}s")

        # Configure the API if not already done - let any error here bubble up to the router
        if g.VERBOSE_DEBUG:
            print(f"🔍 CONFIGURING API: _gemini_api_configured={_gemini_api_configured}")
        GoogleAPI._configure_api()
        
        # Get the appropriate model
        if g.VERBOSE_DEBUG:
            print(f"🔍 CREATING MODEL: {model_key}")
        model = genai.GenerativeModel(model_key)
        if g.VERBOSE_DEBUG:
            print(f"🔍 MODEL CREATED: {model}")
        
        # Set up the generation config with temperature and thinking budget
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            top_p=1.0,
            top_k=32,
            max_output_tokens=8192,
        )
        
        # Add thinking config if thinking_budget is specified and model supports it
        if thinking_budget is not None and ("2.5" in model_key or "2.0" in model_key):
            try:
                generation_config.thinking_config = genai.types.ThinkingConfig(
                    thinking_budget=thinking_budget
                )
            except Exception as thinking_error:
                # If thinking config fails, continue without it
                if not silent_reason:
                    g.debug_log(f"Google-Api: Warning - thinking config not supported: {thinking_error}", "yellow", prefix=prefix)
        elif thinking_budget is not None and not ("2.5" in model_key or "2.0" in model_key):
            # Warn if thinking budget is requested but model doesn't support it
            if not silent_reason:
                g.debug_log(f"Google-Api: Warning - thinking budget not supported by model {model_key}", "yellow", prefix=prefix)
        
        # Use the Chat's to_gemini method to create Gemini-compatible messages
        if isinstance(chat, Chat):
            if g.VERBOSE_DEBUG:
                print("🔍 CONVERTING CHAT TO GEMINI FORMAT")
            gemini_messages = chat.to_gemini()
            
            if g.VERBOSE_DEBUG:
                print(f"🔍 GEMINI MESSAGES: {len(gemini_messages)} messages")
                for i, msg in enumerate(gemini_messages):
                    print(f"🔍   GEMINI MSG {i}: role={msg.get('role', 'unknown')}, parts={len(msg.get('parts', []))}")
                    for j, part in enumerate(msg.get('parts', [])):
                        if isinstance(part, dict) and 'inline_data' in part:
                            mime_type = part['inline_data'].get('mime_type', 'unknown')
                            data_len = len(part['inline_data'].get('data', ''))
                            print(f"🔍     PART {j}: IMAGE ({mime_type}, {data_len} chars)")
                        elif isinstance(part, str):
                            print(f"🔍     PART {j}: TEXT ({len(part)} chars)")
                        else:
                            print(f"🔍     PART {j}: OTHER ({type(part).__name__})")
            
            # Debug vision requests
            has_images = any(
                isinstance(part, dict) and 'inline_data' in part 
                for msg in gemini_messages 
                for part in msg.get('parts', [])
            )
            if has_images and not silent_reason:
                g.debug_log(f"Google-Api: Sending vision request with {sum(len([p for p in msg.get('parts', []) if isinstance(p, dict) and 'inline_data' in p]) for msg in gemini_messages)} image(s)", "cyan", prefix=prefix)
        else:
            # Simple string input case (should not happen due to conversion above, but just in case)
            if g.VERBOSE_DEBUG:
                print(f"🔍 USING STRING INPUT: {len(str(chat))} chars")
            gemini_messages = [{"role": "user", "parts": [str(chat)]}]
            
        # Print status message with timeout info
        if silent_reason:
            temp_str = "" if temperature == 0 or temperature is None else f" at temperature {temperature}"
            g.debug_log(f"Google-Api: {colored('<', 'green')}{colored(model_key, 'green')}{colored('>', 'green')} is {colored('silently', 'green')} generating response{temp_str} (timeout: {g.GOOGLE_API_TIMEOUT_SECONDS}s)...", force_print=True, prefix=prefix)
        else:
            temp_str = "" if temperature == 0 or temperature is None else f" at temperature {temperature}"
            g.debug_log(f"Google-Api: {colored('<', 'green')}{colored(model_key, 'green')}{colored('>', 'green')} is generating response{temp_str} (timeout: {g.GOOGLE_API_TIMEOUT_SECONDS}s)...", "green", force_print=True, prefix=prefix)
        
        # Generate streaming response with timeout - let errors bubble up to the router
        # except for rate limit errors, which we'll handle here
        try:
            import threading
            
            # Debug the actual request being sent
            if g.VERBOSE_DEBUG:
                message_count = len(gemini_messages)
                total_parts = sum(len(msg.get('parts', [])) for msg in gemini_messages)
                image_parts = sum(len([p for p in msg.get('parts', []) if isinstance(p, dict) and 'inline_data' in p]) for msg in gemini_messages)
                print(f"🔍 REQUEST SUMMARY: {message_count} messages, {total_parts} parts, {image_parts} images")
                print(f"🔍 GENERATION CONFIG: temp={generation_config.temperature}, max_tokens={generation_config.max_output_tokens}")
            
            # Use threading for synchronous genai call with timeout
            def _generate_with_timeout():
                try:
                    if g.VERBOSE_DEBUG:
                        print("🔍 CALLING model.generate_content() with stream=True")
                        print(f"🔍 Model info: {model}")
                    
                    result = model.generate_content(
                        gemini_messages,
                        generation_config=generation_config,
                        stream=True
                    )
                    
                    if g.VERBOSE_DEBUG:
                        print(f"🔍 API CALL SUCCESSFUL: Got result {type(result).__name__}")
                    
                    return result
                except Exception as inner_e:
                    if g.VERBOSE_DEBUG:
                        print(f"🔍 INNER API ERROR: {str(inner_e)}")
                        print(f"🔍 INNER API ERROR TYPE: {type(inner_e).__name__}")
                        print(f"🔍 INNER API ERROR ARGS: {inner_e.args}")
                        import traceback
                        print("🔍 INNER API TRACEBACK:")
                        traceback.print_exc()
                    raise
            
            # Create a result container
            result_container = {'result': None, 'exception': None}
            
            def _run_generation():
                try:
                    result_container['result'] = _generate_with_timeout()
                except Exception as e:
                    result_container['exception'] = e
            
            # Run with extended timeout (120 seconds for image analysis tasks)
            thread = threading.Thread(target=_run_generation)
            thread.daemon = True
            thread.start()
            
            # Provide progress feedback during long operations
            timeout_seconds = float(g.GOOGLE_API_TIMEOUT_SECONDS)
            check_interval = 5.0  # Check every 5 seconds
            elapsed = 0.0
            
            while thread.is_alive() and elapsed < timeout_seconds:
                thread.join(timeout=check_interval)
                elapsed += check_interval
                if thread.is_alive() and not silent_reason:
                    # Show progress indicator every 15 seconds
                    if int(elapsed) % 15 == 0:
                        g.debug_log(f"Google-Api: {colored('<', 'yellow')}{colored(model_key, 'yellow')}{colored('>', 'yellow')} still processing... ({int(elapsed)}s elapsed)", "yellow", force_print=True, prefix=prefix)
            
            if thread.is_alive():
                # Thread is still running, timeout occurred
                raise Exception(f"Request timed out after {timeout_seconds} seconds for model {model_key}")
            
            if result_container['exception']:
                raise result_container['exception']
            
            response = result_container['result']
            
            # Record successful request
            enhanced_rate_limit_tracker.record_request(model_key, token_cost=token_cost, image_count=image_count)
            
            return response
        except Exception as e:
            # Use enhanced parser for better rate limit handling
            error_str = str(e)
            
            # Check if this is a rate limit error
            if ("429" in error_str or "quota" in error_str.lower() or 
                "rate" in error_str.lower() or "limit" in error_str.lower()):
                
                # Parse the error using the Gemini-specific parser
                parser = RateLimitParserFactory.get_parser("gemini")
                limit_type, cooldown_seconds, extra_info = parser.parse_error(error_str)
                
                # Update limits based on error info if needed
                RateLimitConfig.update_limits_from_error(model_key, error_str)
                
                # Apply the rate limit
                enhanced_rate_limit_tracker.apply_rate_limit(
                    model_key, limit_type, cooldown_seconds, 
                    retry_delay=extra_info.get('api_suggested_delay', 0)
                )
                
                # Show user-friendly message
                if not silent_reason:
                    limit_name = limit_type.value.replace('_', ' ').title()
                    g.debug_log(f"⚡ {limit_name} limit reached for {model_key} - retry in {cooldown_seconds}s", "yellow", force_print=True, prefix=prefix)
                
                # Mark the exception as already logged and raise as RateLimitException
                setattr(e, 'already_logged', True)
                raise RateLimitException(f"Model {model_key} {limit_type.value} limit exceeded. Try again in {cooldown_seconds:.1f} seconds")
            
            # Log other errors here so we don't get duplicate logs
            if not silent_reason:
                error_msg = f"❌ {model_key}: {str(e).split('.')[0]}"
                g.debug_log(error_msg, "red", is_error=True, prefix=prefix)
                
                # Mark the exception as already logged so the router won't log it again
                setattr(e, 'already_logged', True)
                
            # Re-raise the original exception
            raise

    @staticmethod
    def generate_embeddings(
        text: Union[str, List[str]], 
        model: str = "embedding-001", 
        chat: Optional[Chat] = None
    ) -> Optional[Union[List[float], List[List[float]]]]:
        """
        Generates embeddings for the given text using the Google Gemini API.
        
        Args:
            text (Union[str, List[str]]): The text or list of texts to generate embeddings for.
            model (str): The embedding model to use.
            chat (Optional[Chat]): The chat object for debug printing.
            
        Returns:
            Optional[Union[List[float], List[List[float]]]]: The generated embedding(s) or None if an error occurs.
        """
        try:
            # Configure the API if not already done
            GoogleAPI._configure_api()
            
            # Print status message
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                if isinstance(text, list):
                    g.debug_log(f"Google-Api: {colored('<', 'green')}{colored(model, 'green')}{colored('>', 'green')} is generating {len(text)} embeddings...", force_print=True, prefix=prefix)
                else:
                    g.debug_log(f"Google-Api: {colored('<', 'green')}{colored(model, 'green')}{colored('>', 'green')} is generating embedding...", force_print=True, prefix=prefix)
                    
            # Determine task type for embedding (can influence quality)
            task_type = "RETRIEVAL_DOCUMENT"
                    
            if isinstance(text, str):
                # Single text case
                result = genai.embed_content(
                    model=model,
                    content=text,
                    task_type=task_type
                )
                return result["embedding"]
            else:
                # List of texts case
                embeddings = []
                for t in text:
                    result = genai.embed_content(
                        model=model,
                        content=t,
                        task_type=task_type
                    )
                    embeddings.append(result["embedding"])
                return embeddings
                
        except Exception as e:
            error_msg = f"Google API embedding error: {e}"
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(error_msg, "red", is_error=True, prefix=prefix)
            logger.error(error_msg)
            # Keep returning None for embeddings as this appears to be the expected behavior
            # This is different from generate_response and generate_speech which re-raise exceptions
            return None

    @staticmethod
    def get_available_models(chat: Optional[Chat] = None) -> List[Dict[str, Any]]:
        """
        Discovers and returns available Gemini models.
        
        This method queries the Google Generative AI API to get a real-time list of 
        available models. Can be used for dynamic model discovery in the future.
        Currently, the LLM router uses a static list of known Google models for 
        performance reasons, but this method provides the foundation for dynamic discovery.
        
        Args:
            chat (Optional[Chat]): The chat object for debug printing.
            
        Returns:
            List[Dict[str, Any]]: List of available model information dictionaries.
                Each dictionary contains:
                - name: Model identifier (e.g., "models/gemini-1.5-pro")
                - display_name: Human-readable name
                - description: Model description
                - supported_generation_methods: List of supported methods
                - input_token_limit: Maximum input tokens (if available)
                - output_token_limit: Maximum output tokens (if available)
        """
        try:
            # Configure the API if not already done
            GoogleAPI._configure_api()
            
            # Print status message
            prefix = chat.get_debug_title_prefix() if chat and hasattr(chat, 'get_debug_title_prefix') else ""
            if chat:
                g.debug_log(f"Google-Api: {colored('<', 'green')}{colored('model-discovery', 'green')}{colored('>', 'green')} discovering available models...", force_print=True, prefix=prefix)
            
            models = []
            
            # Use genai.list_models() to retrieve available models
            for model in genai.list_models():
                model_info = {
                    'name': model.name,
                    'display_name': model.display_name,
                    'description': model.description or "",
                    'supported_generation_methods': list(model.supported_generation_methods) if model.supported_generation_methods else [],
                    'version': getattr(model, 'version', ''),
                    'input_token_limit': getattr(model, 'input_token_limit', None),
                    'output_token_limit': getattr(model, 'output_token_limit', None),
                    'temperature': getattr(model, 'temperature', None),
                    'top_p': getattr(model, 'top_p', None),
                    'top_k': getattr(model, 'top_k', None)
                }
                models.append(model_info)
            
            if chat:
                g.debug_log(f"Google-Api: {colored('<', 'green')}{colored('model-discovery', 'green')}{colored('>', 'green')} found {len(models)} available models", force_print=True, prefix=prefix)
            
            return models
            
        except Exception as e:
            error_msg = f"Google API model discovery error: {e}"
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(error_msg, "red", is_error=True, prefix=prefix)
            logger.error(error_msg)
            return []

    @staticmethod
    def generate_speech(
        text: str,
        output_file: str = "output.wav",
        model: str = "gemini-2.5-flash-preview-tts",
        temperature: float = 1.0,
        chat: Optional[Chat] = None,
        speaker_config: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generates speech from text using Google's TTS API with multi-speaker support.
        
        Args:
            text (str): The text to convert to speech.
            output_file (str): The name of the output file to save the audio to.
            model (str): The TTS model to use.
            temperature (float): The temperature setting for the model.
            chat (Optional[Chat]): The chat object for debug printing.
            speaker_config (Optional[List[Dict[str, str]]]): List of speaker configurations.
                Each dictionary should contain 'speaker' and 'voice' keys.
                If None, default voices will be used.
                
        Returns:
            str: The path to the saved audio file.
            
        Raises:
            Exception: If the API is not configured or if there is an error during generation.
        """
        try:
            # Imports are now at the top of the file
            
            # Configure the API if not already done
            GoogleAPI._configure_api()
            
            # Print status message
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(f"Google-Api: {colored('<', 'green')}{colored(model, 'green')}{colored('>', 'green')} is generating speech...", force_print=True, prefix=prefix)
            
            # Create the client
            client = genai.Client( # Using genai from top-level import
                api_key=os.environ.get("GEMINI_API_KEY"),
            )
            
            # Default speaker configuration if none provided
            if not speaker_config:
                speaker_config = [
                    {"speaker": "Chloe", "voice": "Kore"},
                    {"speaker": "Liam", "voice": "Iapetus"}
                ]
            
            # Create speaker voice configs
            speaker_voice_configs = []
            for config in speaker_config:
                speaker_voice_configs.append(
                    types.SpeakerVoiceConfig( # Using types from top-level import
                        speaker=config["speaker"],
                        voice_config=types.VoiceConfig( # Using types from top-level import
                            prebuilt_voice_config=types.PrebuiltVoiceConfig( # Using types from top-level import
                                voice_name=config["voice"]
                            )
                        ),
                    )
                )
            
            # Set up content for TTS
            contents = [
                types.Content( # Using types from top-level import
                    role="user",
                    parts=[
                        types.Part.from_text(text=text), # Using types from top-level import
                    ],
                ),
            ]
            
            # Configure TTS generation
            generate_content_config = types.GenerateContentConfig( # Using types from top-level import
                temperature=temperature,
                response_modalities=["audio"],
                speech_config=types.SpeechConfig( # Using types from top-level import
                    multi_speaker_voice_config=types.MultiSpeakerVoiceConfig( # Using types from top-level import
                        speaker_voice_configs=speaker_voice_configs
                    ),
                ),
            )
            
            # Generate speech content
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                if (
                    chunk.candidates is None
                    or chunk.candidates[0].content is None
                    or chunk.candidates[0].content.parts is None
                ):
                    continue
                
                if chunk.candidates[0].content.parts[0].inline_data:
                    inline_data = chunk.candidates[0].content.parts[0].inline_data
                    data_buffer = inline_data.data
                    file_extension = mimetypes.guess_extension(inline_data.mime_type)
                    
                    if file_extension is None:
                        file_extension = ".wav"
                        data_buffer = convert_to_wav(inline_data.data, inline_data.mime_type)
                    
                    # Ensure output_file has the correct extension
                    base_name, existing_ext = os.path.splitext(output_file)
                    if not existing_ext:
                        output_file = f"{base_name}{file_extension}"
                    
                    # Save the file
                    return save_binary_file(output_file, data_buffer)
                else:
                    # Handle text response
                    if chunk.text and chat:
                        prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                        g.debug_log(f"TTS Response: {chunk.text}", force_print=True, prefix=prefix)
            
            # If we get here without returning a file, raise an exception
            raise Exception("No audio data received from the API")
            
        except Exception as e:
            error_msg = f"Google API speech generation error: {e}"
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(error_msg, "red", is_error=True, prefix=prefix)
            logger.error(error_msg)
            # Mark the exception as already logged so the router won't log it again
            setattr(e, 'already_logged', True)
            # Re-raise the original exception
            raise