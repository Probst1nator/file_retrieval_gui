import tempfile
import os
from typing import Optional, Callable, Union, TYPE_CHECKING
from openai import OpenAI
from termcolor import colored
import speech_recognition as sr
from core.globals import g
from tools.main_cli_agent.agent.text_painter.stream_painter import TextStreamPainter

from py_classes.unified_interfaces import AIProviderInterface

# Only used for type annotations, not actual imports
if TYPE_CHECKING:
    from core.chat import Chat, Role

class HumanAPI(AIProviderInterface):
    """
    Implementation of the ChatClientInterface for the OpenAI API.
    """

    @staticmethod
    def generate_response(
        chat: Union['Chat', str], 
        model_key: str = "human", 
        temperature: float = 0.0, 
        silent_reason: str = "", 
        callback: Optional[Callable] = None,
        thinking_budget: Optional[int] = None
    ) -> Optional[str]:
        """
        Generates a response by prompting the human in front of the terminal.

        Args:
            chat (Union[Chat, str]): The chat object containing messages or a string prompt.
            model_key (str): Unused
            temperature (float): Unused 
            silent_reason (str): Unused
            callback (Callable, optional): A function to call with each chunk of input data.
            
        Returns:
            Optional[str]: The generated response.
            
        Raises:
            Exception: If an error occurs during response generation, to be handled by the router.
        """
        # Convert string to Chat object if needed
        if isinstance(chat, str):
            chat_obj = Chat()
            chat_obj.add_message(Role.USER, chat)
            chat = chat_obj
        
        # Get the prefix for debug logging 
        prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
        
        # Informational logging (not error handling)
        g.debug_log("Human-Api: User is asked for a response...", "green", force_print=True, prefix=prefix)
        g.debug_log(colored(("# " * 20) + "CHAT BEGIN" + (" #" * 20), "yellow"), force_print=True, prefix=prefix)
        chat.print_chat()
        g.debug_log(colored(("# " * 20) + "CHAT STOP" + (" #" * 21), "yellow"), force_print=True, prefix=prefix)
        
        g.debug_log(colored("# # # Enter your multiline response. Type '--f' on a new line when finished.", "blue"), force_print=True, prefix=prefix)
        lines = []
        full_response = ""
        while True:
            line = input()
            if line == "--f":
                break
            lines.append(line)
            full_response = "\n".join(lines)
            
            # Call the callback function if provided
            if callback is not None:
                callback(full_response)

        token_stream_painter = TextStreamPainter()
        for character in full_response:
            g.debug_log(token_stream_painter.apply_color(character), end="", with_title=False, prefix=prefix)
        g.debug_log("", with_title=False, prefix=prefix)
        return full_response

    @staticmethod
    def transcribe_audio(audio_data: sr.AudioData, language: str = "", model: str = "whisper-1", chat: Optional['Chat'] = None) -> tuple[str,str]:
        """
        Transcribes an audio file using the OpenAI Whisper API.

        Args:
            audio_data (sr.AudioData): The audio data object.
            model (str): The model identifier for Whisper.
            language (str): The language of the audio.
            chat (Optional[Chat]): The chat object for debug printing.

        Returns:
            tuple[str,str]: (transcribed text from the audio file, language)
            
        Raises:
            Exception: If an error occurs during transcription.
        """
        temp_audio_file_path = None
        try:
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(f"Human-Api: Using OpenAI to transcribe audio using {colored('<', 'green')}{colored(model, 'green')}{colored('>', 'green')}...", force_print=True, prefix=prefix)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=g.CLIAGENT_PERSISTENT_STORAGE_PATH) as temp_audio_file:
                temp_audio_file.write(audio_data.get_wav_data())
                temp_audio_file_path = temp_audio_file.name

            with open(temp_audio_file_path, 'rb') as audio_file:
                response = client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    response_format="verbose_json",
                    language=language
                )
            
            no_speech_prob = response.segments[0]['no_speech_prob']
            if (no_speech_prob > 0.7):
                if chat:
                    prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                    g.debug_log("No speech detected", force_print=True, prefix=prefix)
                return "", "english"
            
            language = response.language
            
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(f"Transcription complete. Detected language: {language}", "green", force_print=True, prefix=prefix)
                
            return response.text, language

        except Exception as e:
            error_msg = f"OpenAI Whisper API error: {e}"
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(error_msg, "red", is_error=True, prefix=prefix)
            raise Exception(error_msg)

        finally:
            if temp_audio_file_path and os.path.exists(temp_audio_file_path):
                os.remove(temp_audio_file_path)