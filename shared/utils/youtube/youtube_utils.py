import os
import yt_dlp
from typing import Optional

class YouTube:
    """
    A static class to handle YouTube video operations, including downloading videos and converting to MP3.
    """

    @classmethod
    def download_video(cls, url: str, directory_path: str, filename: Optional[str] = None) -> str:
        """
        Download a YouTube video from the given URL to the specified directory.

        Args:
            url (str): The URL of the YouTube video to download.
            directory_path (str): The directory path where the video will be saved.
            filename (Optional[str]): The desired filename for the downloaded video.
                                      If not provided, yt-dlp will use the video's title.

        Returns:
            str: The path of the downloaded video file.

        Raises:
            ValueError: If the URL is invalid or the video is unavailable.
            OSError: If there's an issue with the directory path or file writing.
        """
        try:
            # Create the directory if it doesn't exist
            os.makedirs(directory_path, exist_ok=True)

            # Configure yt-dlp options
            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': os.path.join(directory_path, '%(title)s.%(ext)s') if not filename else os.path.join(directory_path, filename),
            }

            # Download the video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                output_path = ydl.prepare_filename(info)

            print(f"Video downloaded successfully: {output_path}")
            return output_path

        except Exception as e:
            raise ValueError(f"Error downloading video: {str(e)}")

    @classmethod
    def convert_video_to_mp3(cls, file_path: str) -> str:
        """
        Convert a video file to MP3 format.

        Args:
            file_path (str): The path to the video file.

        Returns:
            str: The path of the converted MP3 file.

        Raises:
            ValueError: If the file doesn't exist or there's an error during conversion.
        """
        from moviepy.editor import VideoFileClip
        try:
            if not os.path.exists(file_path):
                raise ValueError(f"File not found: {file_path}")

            # Get the directory and filename
            directory, full_filename = os.path.split(file_path)
            filename, _ = os.path.splitext(full_filename)

            # Create the output MP3 filename
            mp3_filename = cls._sanitize_filename(f"{filename}.mp3")
            mp3_path = os.path.join(directory, mp3_filename)

            # Convert video to MP3
            video = VideoFileClip(file_path)
            video.audio.write_audiofile(mp3_path)

            # Close the video file to release resources
            video.close()

            print(f"Video converted to MP3 successfully: {mp3_path}")
            return mp3_path

        except Exception as e:
            raise ValueError(f"Error converting video to MP3: {str(e)}")

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """
        Sanitize the filename to remove invalid characters.

        Args:
            filename (str): The original filename.

        Returns:
            str: The sanitized filename.
        """
        return "".join([c for c in filename if c.isalnum() or c in (' ', '-', '_', '.')])\
            .rstrip().replace(' ', '_')

# Example usage
if __name__ == "__main__":
    # Example of downloading a video
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    directory = "./downloads"
    
    try:
        downloaded_file = YouTube.download_video(url, directory)
        print(f"Video saved to: {downloaded_file}")

        # Example of converting the downloaded video to MP3
        mp3_file = YouTube.convert_video_to_mp3(downloaded_file)
        print(f"MP3 saved to: {mp3_file}")
    except ValueError as e:
        print(e)