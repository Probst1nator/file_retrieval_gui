import os
import tempfile
import soundfile as sf
from typing import Tuple, Optional, List, Union
import numpy as np
import speech_recognition as sr

from py_classes.unified_interfaces import AudioProviderInterface
from core.globals import g
from shared.utils_audio import (
    transcribe_audio as utils_transcribe_audio,
    play_notification,
    play_audio,
    record_audio,
    text_to_speech
)

class WhisperInterface(AudioProviderInterface):
    """
    Implementation of the AudioProviderInterface using Whisper for transcription
    and Kokoro for text-to-speech.
    """
    
    def transcribe_audio(self, audio_data: Union[sr.AudioData, np.ndarray], 
                         language: str = "", 
                         model: str = "medium",
                         sample_rate: int = 44100) -> Tuple[str, str]:
        """
        Transcribes audio data using the local Whisper model.
        
        Args:
            audio_data: Either a speech_recognition AudioData object or a numpy array of audio samples
            language: Optional language hint for transcription
            model: Whisper model size to use (tiny, base, small, medium, large-v2)
            sample_rate: Sample rate of the audio data if provided as numpy array
            
        Returns:
            Tuple[str, str]: (transcribed text, detected language)
        """
        try:
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=g.CLIAGENT_PERSISTENT_STORAGE_PATH) as temp_audio_file:
                temp_audio_file_path = temp_audio_file.name
                
                # Handle different input types
                if isinstance(audio_data, sr.AudioData):
                    # Write the WAV data to the temp file
                    temp_audio_file.write(audio_data.get_wav_data())
                elif isinstance(audio_data, np.ndarray):
                    # Write numpy array to the temp file
                    sf.write(temp_audio_file_path, audio_data, sample_rate)
                else:
                    raise TypeError("audio_data must be either sr.AudioData or numpy.ndarray")
            
            # Read the audio data from the temporary file
            audio_array, file_sample_rate = sf.read(temp_audio_file_path)
            
            # Transcribe the audio using our utility function
            transcribed_text, detected_language = utils_transcribe_audio(
                audio_array=audio_array,
                sample_rate=file_sample_rate,
                whisper_model_key=model
            )
            
            return transcribed_text, detected_language
            
        except Exception as e:
            print(f"Whisper transcription error: {e}")
            return "", ""
        
        finally:
            # Clean up the temporary file
            if 'temp_audio_file_path' in locals() and os.path.exists(temp_audio_file_path):
                os.remove(temp_audio_file_path)
    
    def record_and_transcribe(
        self,
        language: str = "",
        model: str = "medium",
        sample_rate: int = 44100,
        threshold: float = 0.05,
        silence_duration: float = 2.0,
        min_duration: float = 1.0,
        max_duration: float = 30.0,
        use_wake_word: bool = True
    ) -> Tuple[str, str, np.ndarray]:
        """
        Record audio from the microphone and transcribe it.
        
        Args:
            language: Language hint for transcription
            model: Whisper model to use (tiny, base, small, medium, large-v2)
            sample_rate: Sample rate for recording
            threshold: Volume threshold for speech detection
            silence_duration: Duration of silence to stop recording (seconds)
            min_duration: Minimum recording duration (seconds)
            max_duration: Maximum recording duration (seconds)
            use_wake_word: Whether to wait for wake word before recording
            
        Returns:
            Tuple[str, str, np.ndarray]: (transcribed text, detected language, audio data)
        """
        try:
            # Record audio
            audio_data, sample_rate = record_audio(
                sample_rate=sample_rate,
                threshold=threshold,
                silence_duration=silence_duration,
                min_duration=min_duration,
                max_duration=max_duration,
                use_wake_word=use_wake_word
            )
            
            # If we got no audio, return empty results
            if len(audio_data) == 0:
                return "", "", audio_data
            
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=g.CLIAGENT_PERSISTENT_STORAGE_PATH) as temp_audio_file:
                temp_audio_file_path = temp_audio_file.name
                sf.write(temp_audio_file_path, audio_data, sample_rate)
            
            # Transcribe the audio
            transcribed_text, detected_language = utils_transcribe_audio(
                audio_path=temp_audio_file_path,
                whisper_model_key=model
            )
            
            return transcribed_text, detected_language, audio_data
            
        except Exception as e:
            print(f"Error in recording and transcription: {e}")
            return "", "", np.array([])
        
        finally:
            # Clean up the temporary file
            if 'temp_audio_file_path' in locals() and os.path.exists(temp_audio_file_path):
                os.remove(temp_audio_file_path)
    
    def speak(self,
             text: str,
             voice: str = "af_heart",
             speed: float = 1.0,
             output_path: Optional[str] = None,
             play: bool = True) -> Union[List[str], None]:
        """
        Converts text to speech using Kokoro TTS.
        
        Args:
            text: Text to convert to speech
            voice: Voice identifier (default: af_heart)
            speed: Speaking speed (default: 1.0)
            output_path: Optional path to save audio file
            play: Whether to play the audio (default: True)
            
        Returns:
            Union[List[str], None]: List of generated audio file paths if output_path is provided, None otherwise
        """
        return text_to_speech(
            text=text,
            voice=voice,
            speed=speed,
            output_path=output_path,
            play=play
        )
    
    def play_notification_sound(self) -> None:
        """Play a notification sound to indicate when to start speaking."""
        play_notification()
    
    def play_audio_data(self, audio_array: np.ndarray, sample_rate: int = 44100, blocking: bool = True) -> None:
        """
        Play audio data.
        
        Args:
            audio_array: The audio data to play
            sample_rate: Sample rate of the audio data
            blocking: Whether to block until audio finishes playing
        """
        play_audio(audio_array, sample_rate, blocking) 