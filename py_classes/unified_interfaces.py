from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union, Any
import numpy as np
import speech_recognition as sr
import logging

from core.chat import Chat

# Configure logger with a NullHandler by default
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class BaseProviderInterface(ABC):
    """Base interface for all service providers with common methods"""
    
    @staticmethod
    def configure_logging(level: int = logging.ERROR):
        """
        Configure logging for all interfaces.
        
        Args:
            level (int): Logging level
        """
        # Remove any existing handlers and set up console handler
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)

        # Add a console handler that only shows specified level and above
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
        logger.setLevel(level)
        
    @staticmethod
    def get_debug_title_prefix(chat: Optional['Chat']) -> str:
        """
        Get a formatted prefix string for debug messages that includes the chat's debug title if available.
        
        Args:
            chat (Chat): The chat whose debug_title should be included
            
        Returns:
            str: The formatted prefix string
        """
        if chat is None:
            return ""
        return chat.get_debug_title_prefix()


class AudioProviderInterface(BaseProviderInterface):
    """
    Interface for audio provider services.
    Defines the contract for speech recognition and synthesis services.
    """
    
    @abstractmethod
    def transcribe_audio(self, audio_data: Union[sr.AudioData, np.ndarray], 
                        language: str = "", 
                        model: str = "",
                        sample_rate: int = 44100) -> Tuple[str, str]:
        """
        Transcribes audio data to text.
        
        Args:
            audio_data: Audio data to transcribe
            language: Language hint for transcription
            model: Model to use for transcription
            sample_rate: Sample rate of the audio data if provided as numpy array
            
        Returns:
            Tuple[str, str]: (transcribed text, detected language)
        """
        pass
    
    @abstractmethod
    def record_and_transcribe(self,
                             language: str = "",
                             model: str = "",
                             sample_rate: int = 44100,
                             threshold: float = 0.05,
                             silence_duration: float = 2.0,
                             min_duration: float = 1.0,
                             max_duration: float = 30.0,
                             use_wake_word: bool = True) -> Tuple[str, str, np.ndarray]:
        """
        Records audio from microphone and transcribes it.
        
        Args:
            language: Language hint for transcription
            model: Model to use for transcription
            sample_rate: Sample rate for recording
            threshold: Volume threshold for speech detection
            silence_duration: Duration of silence to stop recording (seconds)
            min_duration: Minimum recording duration (seconds)
            max_duration: Maximum recording duration (seconds)
            use_wake_word: Whether to wait for wake word before recording
            
        Returns:
            Tuple[str, str, np.ndarray]: (transcribed text, detected language, audio data)
        """
        pass
    
    @abstractmethod
    def speak(self,
             text: str,
             voice: str = "",
             speed: float = 1.0,
             output_path: Optional[str] = None,
             play: bool = True) -> Union[List[str], None]:
        """
        Converts text to speech.
        
        Args:
            text: Text to convert to speech
            voice: Voice identifier
            speed: Speaking speed
            output_path: Optional path to save audio file
            play: Whether to play the audio
            
        Returns:
            Union[List[str], None]: List of generated audio file paths if output_path is provided, None otherwise
        """
        pass
    
    @abstractmethod
    def play_notification_sound(self) -> None:
        """
        Plays a notification sound.
        """
        pass
    
    @abstractmethod
    def play_audio_data(self, audio_array: np.ndarray, sample_rate: int = 44100, blocking: bool = True) -> None:
        """
        Plays audio data.
        
        Args:
            audio_array: Audio data to play
            sample_rate: Sample rate of the audio data
            blocking: Whether to block until audio finishes playing
        """
        pass


class AIProviderInterface(BaseProviderInterface):
    """
    Interface for AI provider services.
    Defines the contract for text generation and other AI services.
    """
    
    @abstractmethod
    def generate_response(self, chat: Union['Chat', str], model_key: str,
                         temperature: Optional[float] = None, silent_reason: str = "", 
                         thinking_budget: Optional[int] = None) -> Any:
        """
        Generates a response stream based on the provided chat and model.

        Args:
            chat: The chat object containing messages or a string prompt.
            model_key: The model identifier.
            temperature: The temperature setting for the model.
            silent_reason: Reason for suppressing print statements.
            thinking_budget: Token budget for model's internal reasoning process (Gemini models only). 
                           Use -1 for dynamic, 0 to disable, or positive integer for fixed budget.

        Returns:
            Any: A stream object that yields response chunks.
        """
        pass
    
    
class UnifiedProviderInterface(AIProviderInterface, AudioProviderInterface):
    """
    Combined interface that provides both AI and audio capabilities.
    Implementations of this interface must implement all methods from both parent interfaces.
    """
    pass 