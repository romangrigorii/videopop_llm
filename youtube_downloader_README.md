# YouTube Video Downloader

A simple Python script to download YouTube videos in low resolution.

## Installation

1. Install the required dependencies:
```bash
pip install pytube
```

## Usage

### Command Line Usage

```bash
python youtube_downloader.py -u "https://www.youtube.com/watch?v=VIDEO_ID" [options]
```

Options:
- `-u, --url`: YouTube video URL to download
- `-o, --output`: Directory to save the downloaded video (default: downloads)
- `-r, --resolution`: Desired resolution (e.g., '360p', '480p'). If not specified, downloads lowest available.

Examples:
```bash
# Download video in lowest available resolution
python youtube_downloader.py -u "https://www.youtube.com/watch?v=VIDEO_ID"

# Download video in specific resolution
python youtube_downloader.py -u "https://www.youtube.com/watch?v=VIDEO_ID" -r 360p

# Download to custom directory
python youtube_downloader.py -u "https://www.youtube.com/watch?v=VIDEO_ID" -o my_videos
```

### Python API Usage

```python
from youtube_downloader import download_youtube_video

# Download video in lowest resolution
video_path = download_youtube_video("https://www.youtube.com/watch?v=VIDEO_ID")

# Download video in specific resolution
video_path = download_youtube_video(
    "https://www.youtube.com/watch?v=VIDEO_ID",
    resolution="360p"
)

# Download to custom directory
video_path = download_youtube_video(
    "https://www.youtube.com/watch?v=VIDEO_ID",
    output_path="my_videos"
)
```

## Features

- Downloads YouTube videos in low resolution
- Option to specify desired resolution
- Falls back to lowest available resolution if specified resolution is not available
- Shows available resolutions if requested resolution is not found
- Creates output directory if it doesn't exist
- Returns path to downloaded video file 