# Audio Transcription with Timestamps

This script provides functionality to transcribe audio files with word-level timestamps using the `whisper-timestamped` package. It can also download and transcribe audio from YouTube videos.

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Usage

```bash
# Transcribe a local audio file
python transcribe_audio.py -a path/to/your/audio/file.mp3 [options]

# Download and transcribe a YouTube video
python transcribe_audio.py -y "https://www.youtube.com/watch?v=VIDEO_ID" [options]
```

Options:
- `-a, --audio_path`: Path to the audio file to transcribe
- `-y, --youtube_url`: YouTube video URL to download and transcribe
- `-m, --model`: Whisper model to use (tiny, base, small, medium, large). Default: base
- `-l, --language`: Language code (e.g., 'en', 'fr', 'es'). If not specified, will auto-detect
- `-o, --output`: Path to save the JSON output. Default: transcription_result.json
- `-d, --download_dir`: Directory to save downloaded YouTube audio. Default: downloads

Examples:
```bash
# Transcribe a local audio file with default settings
python transcribe_audio.py -a audio.mp3

# Download and transcribe a YouTube video using a larger model
python transcribe_audio.py -y "https://www.youtube.com/watch?v=VIDEO_ID" -m large

# Download to custom directory and specify language
python transcribe_audio.py -y "https://www.youtube.com/watch?v=VIDEO_ID" -d my_downloads -l en
```

### Python API Usage

```python
from transcribe_audio import transcribe_audio_with_timestamps, format_transcription, download_youtube_audio

# Download audio from YouTube
audio_path = download_youtube_audio("https://www.youtube.com/watch?v=VIDEO_ID")

# Transcribe the audio file
result = transcribe_audio_with_timestamps(
    audio_path=audio_path,
    model_name="base",  # Options: tiny, base, small, medium, large
    language="en"  # Optional: specify language code or None for auto-detection
)

# Get formatted transcription with timestamps
formatted_text = format_transcription(result)
print(formatted_text)
```

## Features

- Word-level timestamp accuracy
- Multiple language support
- Different model sizes available (tiny, base, small, medium, large)
- Automatic language detection
- Formatted output with timestamps in HH:MM:SS.mmm format
- YouTube video audio download support
- Low-resolution audio download for faster processing

## Output Format

The transcription will be output in the following format:
```
[00:00:00.000 --> 00:00:05.000] First segment of transcribed text
[00:00:05.000 --> 00:00:10.000] Second segment of transcribed text
...
```

The raw transcription result is also saved as a JSON file containing detailed information about each segment, including word-level timestamps.
