# -*- coding: utf-8 -*-
"""
Audio processing utilities for voice interaction pipeline.

Provides wake word detection, audio recording, speech-to-text transcription,
and text-to-speech synthesis using local models with lazy loading for efficiency.
"""

import json
import logging
import queue
import struct
import subprocess
import time
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np
import psutil
import sounddevice as sd
import soundfile as sf
from termcolor import colored
from vosk import KaldiRecognizer, Model, SetLogLevel

# --- Global Resources ---
# We define our models globally so they are loaded only once. It's the lazy,
# efficient way to do things.
_whisper_model: Optional[Any] = None
_whisper_model_key: Optional[str] = None
_vosk_model: Optional[Model] = None
_tts_pipeline: Optional[Any] = None
_tts_model_id: Optional[str] = None
_device: Optional[str] = None

logger = logging.getLogger(__name__)


def get_torch_device() -> str:
    """
    Determine the optimal torch device (CUDA or CPU).
    
    Returns:
        str: Device string ('cuda' or 'cpu')
    """
    global _device
    if _device is None:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"<{colored('System', 'blue')}> Determined torch device: {_device}")
    return _device


def initialize_wake_word() -> None:
    """Initialize the Vosk wake word detector model."""
    global _vosk_model
    if not _vosk_model:
        print(f"<{colored('Vosk', 'green')}> Initializing wake word model...")
        SetLogLevel(-1)  # Vosk, please be quiet.
        _vosk_model = Model(lang="en-us")


def wait_for_wake_word() -> Optional[str]:
    """
    Listen for wake word detection using Vosk.
    
    Returns:
        Optional[str]: Detected wake word phrase, or None if detection failed
    """
    global _vosk_model
    initialize_wake_word()

    audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=10)
    target_sample_rate = 16000

    # Words our digital friend will respond to. Feel free to expand.
    wake_words = [
        "computer", "nova", "system", "hey computer", "hey nova",
        "ok computer", "ok nova", "okay computer", "okay nova"
    ]

    rec = KaldiRecognizer(_vosk_model, target_sample_rate, json.dumps(wake_words))

    def audio_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Audio stream callback for processing audio chunks."""
        if status:
            logger.warning(f"Audio stream status: {status}")
        try:
            audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            pass

    try:
        with sd.RawInputStream(
            samplerate=target_sample_rate,
            channels=1,
            dtype='int16',
            blocksize=4000,  # 250ms chunks
            callback=audio_callback
        ):
            print(f"<{colored('Vosk', 'green')}> Listening for a wake word... ({', '.join(wake_words)})")
            while True:
                audio_data = audio_queue.get()
                if rec.AcceptWaveform(audio_data):
                    result_json = json.loads(rec.Result())
                    text = result_json.get("text", "").strip()
                    if text:
                        print(f"<{colored('Vosk', 'green')}> Detected potential wake word: '{text}'")
                        return text
    except Exception as e:
        logger.error(f"Error in wake word detection: {e}\n{traceback.format_exc()}")
        return None


def play_notification() -> None:
    """
    Play a notification sound to signal it's time to speak.
    
    Generates a pleasant two-tone notification using G3 and D4 frequencies.
    """
    sample_rate = 44100
    duration = 0.4
    t = np.linspace(0., duration, int(sample_rate * duration), endpoint=False)
    freq1, freq2 = 196.00, 293.66

    tone1 = 0.3 * np.sin(2 * np.pi * freq1 * t)
    tone2 = 0.2 * np.sin(2 * np.pi * freq2 * t)
    signal = tone1 + tone2

    fade_len = int(0.05 * sample_rate)
    fade_in = np.linspace(0., 1., fade_len)
    fade_out = np.linspace(1., 0., fade_len)
    signal[:fade_len] *= fade_in
    signal[-fade_len:] *= fade_out

    play_audio(signal, sample_rate, blocking=False)


def play_audio(audio_array: np.ndarray, sample_rate: int, blocking: bool = True) -> None:
    """
    Play audio data through the default sound device.
    
    Args:
        audio_array: Audio data as numpy array
        sample_rate: Sample rate in Hz
        blocking: Whether to wait for playback to complete
    """
    try:
        audio_array = audio_array.astype(np.float32)
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array /= max_val
        
        sd.play(audio_array, sample_rate)
        if blocking:
            sd.wait()
    except Exception as e:
        logger.error(f"Could not play audio: {e}")
        sd.stop()


def record_audio(
    sample_rate: int = 16000,
    threshold: float = 0.02,
    silence_duration: float = 1.5,
    max_duration: float = 20.0,
) -> Optional[np.ndarray]:
    """
    Record audio from microphone with automatic silence detection.
    
    Args:
        sample_rate: Audio sample rate in Hz
        threshold: Volume threshold for speech detection
        silence_duration: Seconds of silence before stopping
        max_duration: Maximum recording duration in seconds
        
    Returns:
        Optional[np.ndarray]: Recorded audio data, or None if no speech detected
    """
    play_notification()
    print(f"<{colored('Recording', 'red')}> Listening... (stops after {silence_duration}s of silence)")

    q: queue.Queue[np.ndarray] = queue.Queue()
    recording = False
    silent_chunks = 0
    num_silent_chunks_to_stop = int(silence_duration * 10)

    def callback(indata: np.ndarray, frames: int, time_info: Any, status: Any):
        if status:
            logger.warning(status)
        q.put(indata.copy())

    recorded_frames = []
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', blocksize=int(sample_rate / 10), callback=callback):
        start_time = time.time()
        while True:
            frame = q.get()
            is_loud = np.abs(frame).mean() > threshold

            if time.time() - start_time > max_duration:
                print("\n<Recording> Reached maximum recording time.")
                break

            if recording:
                recorded_frames.append(frame)
                if is_loud:
                    silent_chunks = 0
                else:
                    silent_chunks += 1
                
                if silent_chunks > num_silent_chunks_to_stop:
                    print("\n<Recording> Detected end of speech.")
                    break
            elif is_loud:
                print("<Recording> Speech detected, starting...", end="", flush=True)
                recording = True
                recorded_frames.append(frame)

    if not recorded_frames:
        print("\n<Recording> No speech detected.")
        return None

    return np.concatenate(recorded_frames, axis=0)


def initialize_whisper_model(model_key: str = 'tiny') -> None:
    """
    Initialize Whisper model with memory-aware model selection.
    
    Args:
        model_key: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
    """
    global _whisper_model, _whisper_model_key
    try:
        import whisper
    except ImportError:
        logger.error("Whisper library not found. Please install it with 'pip install openai-whisper'")
        return

    if _whisper_model and _whisper_model_key == model_key:
        return

    available_mem = psutil.virtual_memory().available
    mem_reqs = {"large": 10e9, "medium": 5e9, "small": 2e9, "base": 1e9, "tiny": 5e8}
    if available_mem < mem_reqs.get(model_key, 0):
        logger.warning(f"Not enough memory for Whisper '{model_key}'. Falling back.")
        for key, req in sorted(mem_reqs.items(), key=lambda item: item[1], reverse=True):
            if available_mem > req:
                model_key = key
                break
        else:
            model_key = 'tiny'
    
    print(f"<{colored('Whisper', 'magenta')}> Loading '{model_key}' model...")
    _whisper_model = whisper.load_model(model_key, device=get_torch_device())
    _whisper_model_key = model_key
    print(f"<{colored('Whisper', 'magenta')}> Model loaded successfully.")


def transcribe_audio(audio_array: np.ndarray, sample_rate: int = 16000, whisper_model_key: str = 'tiny') -> Tuple[str, str]:
    """
    Transcribe audio using Whisper model.
    
    Args:
        audio_array: Audio data as numpy array
        sample_rate: Audio sample rate in Hz
        
    Returns:
        Tuple[str, str]: (transcribed_text, detected_language)
    """
    initialize_whisper_model()
    if not _whisper_model:
        return "Error: Whisper model not initialized.", "unknown"

    print(f"<{colored('Whisper', 'magenta')}> Transcribing audio...")
    audio_float32 = audio_array.astype(np.float32)

    result = _whisper_model.transcribe(audio_float32, fp16=(get_torch_device()=='cuda'))
    text = result.get("text", "").strip()
    language = result.get("language", "unknown")
    
    print(f"<{colored('Whisper', 'magenta')}> Transcription: '{text}'")
    return text, language


def initialize_tts_pipeline(model_id: str) -> bool:
    """
    Initialize Hugging Face TTS pipeline.
    
    Args:
        model_id: Hugging Face model identifier
        
    Returns:
        bool: True if initialization successful, False otherwise
    """
    global _tts_pipeline, _tts_model_id
    if _tts_pipeline and _tts_model_id == model_id:
        return True

    try:
        from transformers import pipeline
        print(f"<{colored('TTS', 'cyan')}> Initializing pipeline with model: {model_id}...")
        device = get_torch_device()
        _tts_pipeline = pipeline("text-to-speech", model=model_id, device=device)
        _tts_model_id = model_id
        print(f"<{colored('TTS', 'cyan')}> Pipeline ready!")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize TTS pipeline '{model_id}': {e}\n{traceback.format_exc()}")
        _tts_pipeline = None
        _tts_model_id = None
        return False


def text_to_speech(
    text: str,
    model_id: str = "microsoft/speecht5_tts",
    output_path: Optional[str] = None,
    play: bool = True,
    **generation_kwargs: Any
) -> Optional[str]:
    """
    Convert text to speech using Hugging Face models with fallback to system TTS.
    
    Args:
        text: Text to convert to speech
        model_id: Hugging Face model identifier
        output_path: Optional path to save audio file
        play: Whether to play audio immediately
        **generation_kwargs: Additional generation parameters
        
    Returns:
        Optional[str]: Path to saved audio file if output_path provided, None otherwise
    """
    # First try simpler TTS alternatives before Hugging Face
    print(f"<{colored('TTS', 'cyan')}> Converting text to speech: '{text[:50]}...'")
    
    # Try system TTS first (more reliable)
    if _try_system_tts(text):
        return None
        
    # Try Hugging Face model if system TTS fails
    if not initialize_tts_pipeline(model_id):
        print(f"<{colored('TTS Fallback', 'red')}> All TTS methods failed. Cannot speak.")
        return None

    print(f"<{colored('TTS', 'cyan')}> Synthesizing speech with Hugging Face model...")
    try:
        output = _tts_pipeline(text, **generation_kwargs)
        audio_array = output["audio"]
        sample_rate = output["sampling_rate"]

        if play:
            play_audio(audio_array, sample_rate)

        if output_path:
            if not output_path.lower().endswith('.wav'):
                output_path += ".wav"
            sf.write(output_path, audio_array, sample_rate)
            print(f"<{colored('TTS', 'cyan')}> Audio saved to: {output_path}")
            return output_path
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}\n{traceback.format_exc()}")
        # Final fallback to system TTS without capture_output
        _try_system_tts(text)
    
    return None


def _try_system_tts(text: str) -> bool:
    """
    Try system TTS using espeak or other available TTS engines.
    
    Args:
        text: Text to speak
        
    Returns:
        bool: True if successful, False otherwise
    """
    tts_commands = [
        ['espeak', text],
        ['espeak-ng', text],
        ['festival', '--tts'],
        ['say', text],  # macOS
        ['spd-say', text]  # speech-dispatcher
    ]
    
    for cmd in tts_commands:
        try:
            if cmd[0] == 'festival':
                # Festival needs text piped to stdin
                result = subprocess.run(cmd, input=text, text=True, check=True, timeout=10)
            else:
                # Don't capture output for audio to work properly
                result = subprocess.run(cmd, check=True, timeout=10)
            print(f"<{colored('TTS System', 'green')}> Speech synthesis successful using {cmd[0]}")
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
    
    print(f"<{colored('TTS System', 'red')}> No working system TTS found")
    return False


# --- Restored Helper Functions ---
# These functions were in your original file and are needed by other parts
# of your application (like the Google API interface).

def save_binary_file(file_name: str, data: bytes) -> str:
    """
    Save binary data to a file.
    
    Args:
        file_name: Name of the file to save
        data: Binary data to save
        
    Returns:
        str: Path to the saved file
        
    Raises:
        Exception: If file saving fails
    """
    try:
        with open(file_name, "wb") as f:
            f.write(data)
        logger.info(f"File saved to: {file_name}")
        return file_name
    except Exception as e:
        logger.error(f"Error saving file {file_name}: {e}")
        raise Exception(f"Failed to save file: {e}")


def parse_audio_mime_type(mime_type: str) -> Dict[str, int]:
    """
    Parse audio parameters from MIME type string.
    
    Args:
        mime_type: Audio MIME type string (e.g., "audio/L16;rate=24000")
        
    Returns:
        Dict[str, int]: Dictionary with 'bits_per_sample' and 'rate' keys
    """
    bits_per_sample = 16
    rate = 24000
    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                pass
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass

    return {"bits_per_sample": bits_per_sample, "rate": rate}


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """
    Convert audio data to WAV format with proper header.
    
    Args:
        audio_data: Raw audio data bytes
        mime_type: Audio MIME type for parameter extraction
        
    Returns:
        bytes: Complete WAV file data with header
    """
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size
    )
    return header + audio_data