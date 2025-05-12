import os
from pytubefix import YouTube
from typing import Optional, List
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_urls_from_file(file_path: str) -> List[str]:
    """
    Read YouTube URLs from a file.
    Each URL should be on a separate line.
    Lines starting with # are treated as comments and ignored.
    
    Args:
        file_path (str): Path to the file containing YouTube URLs
    
    Returns:
        List[str]: List of YouTube URLs
    """
    urls = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    urls.append(line)
        return urls
    except Exception as e:
        raise Exception(f"Error reading URLs from file: {str(e)}")

def download_youtube_video(
    url: str, 
    output_path: str = "downloads",
    resolution: Optional[str] = None
) -> str:
    """
    Download a YouTube video in the specified or lowest available resolution.
    
    Args:
        url (str): YouTube video URL
        output_path (str): Directory to save the downloaded video file
        resolution (str, optional): Desired resolution (e.g., '360p', '480p'). If None, downloads lowest available.
    
    Returns:
        str: Path to the downloaded video file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Create YouTube object
        yt = YouTube(url)
        
        # Get available streams
        if resolution:
            # Try to get the specified resolution
            video_stream = yt.streams.filter(progressive=True, resolution=resolution).first()
            if not video_stream:
                logger.warning(f"Resolution {resolution} not available for {yt.title}. Available resolutions:")
                available_resolutions = [s.resolution for s in yt.streams.filter(progressive=True)]
                logger.warning(", ".join(available_resolutions))
                # Fall back to lowest resolution
                video_stream = yt.streams.filter(progressive=True).order_by('resolution').first()
        else:
            # Get the lowest resolution
            video_stream = yt.streams.filter(progressive=True).order_by('resolution').first()
        
        if not video_stream:
            raise ValueError("No video stream found for this video")
        
        # Download the video
        logger.info(f"Downloading video: {yt.title}")
        logger.info(f"Resolution: {video_stream.resolution}")
        video_file = video_stream.download(output_path=output_path)
        
        logger.info(f"Downloaded video saved to: {video_file}")
        return video_file
        
    except Exception as e:
        logger.error(f"Error downloading YouTube video {url}: {str(e)}")
        return None

def download_multiple_videos(
    urls: List[str],
    output_path: str = "downloads",
    resolution: Optional[str] = None
) -> List[str]:
    """
    Download multiple YouTube videos.
    
    Args:
        urls (List[str]): List of YouTube video URLs
        output_path (str): Directory to save the downloaded videos
        resolution (str, optional): Desired resolution for all videos
    
    Returns:
        List[str]: List of paths to successfully downloaded videos
    """
    successful_downloads = []
    
    for url in urls:
        try:
            video_path = download_youtube_video(url, output_path, resolution)
            if video_path:
                successful_downloads.append(video_path)
        except Exception as e:
            logger.error(f"Failed to download {url}: {str(e)}")
            continue
    
    return successful_downloads

if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download YouTube videos in low resolution")
    parser.add_argument("-u", "--url", help="YouTube video URL to download")
    parser.add_argument(
        "-f", 
        "--file",
        help="Path to a file containing YouTube URLs (one per line)"
    )
    parser.add_argument(
        "-o", 
        "--output",
        default="downloads",
        help="Directory to save the downloaded videos (default: downloads)"
    )
    parser.add_argument(
        "-r",
        "--resolution",
        help="Desired resolution (e.g., '360p', '480p'). If not specified, downloads lowest available."
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.url and not args.file:
        parser.error("Either --url or --file must be specified")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    if args.file:
        # Download multiple videos from file
        logger.info(f"Reading URLs from file: {args.file}")
        urls = read_urls_from_file(args.file)
        logger.info(f"Found {len(urls)} URLs to download")
        
        successful_downloads = download_multiple_videos(
            urls=urls,
            output_path=args.output,
            resolution=args.resolution
        )
        
        logger.info(f"Successfully downloaded {len(successful_downloads)} videos")
        for video_path in successful_downloads:
            logger.info(f"Downloaded: {video_path}")
    else:
        # Download single video
        video_path = download_youtube_video(
            url=args.url,
            output_path=args.output,
            resolution=args.resolution
        ) 