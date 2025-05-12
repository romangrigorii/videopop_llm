import os
import whisper
import torch
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Any
import json
from datetime import timedelta
import whisper_timestamped as whisper_ts
import traceback
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoContextExtractor:
    def __init__(self, model_name: str = "base"):
        """Initialize the video context extractor with a Whisper model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        logger.info(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name, device=self.device)
        
        # Create output directory
        self.output_dir = "contexts"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def sanitize_filename(self, filename: str) -> str:
        """Remove or replace problematic characters in filenames."""
        # Replace problematic characters with underscores
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove trailing spaces and dots
        sanitized = sanitized.strip('. ')
        return sanitized
    
    def get_video_files(self, directory: str = "downloads") -> List[str]:
        """Get all video files from the specified directory."""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        video_files = []
        
        try:
            for file in os.listdir(directory):
                if Path(file).suffix.lower() in video_extensions:
                    video_files.append(os.path.join(directory, file))
        except Exception as e:
            logger.error(f"Error reading directory {directory}: {str(e)}")
            logger.error(traceback.format_exc())
        
        return video_files
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from the text."""
        # Simple keyword extraction - can be enhanced with NLP libraries
        words = text.lower().split()
        # Remove common words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        return list(set(keywords))  # Remove duplicates
    
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """Process a single video and extract its context."""
        logger.info(f"Processing video: {video_path}")
        
        try:
            # Verify file exists and is readable
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            if not os.access(video_path, os.R_OK):
                raise PermissionError(f"Cannot read video file: {video_path}")
            
            # Get file size
            file_size = os.path.getsize(video_path)
            logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
            
            # Transcribe with timestamps
            logger.info("Starting transcription...")
            result = whisper_ts.transcribe(self.model, video_path)
            logger.info("Transcription completed")
            
            # Extract segments with timestamps
            segments = []
            total_duration = 0
            
            for segment in result['segments']:
                # Calculate segment duration
                segment_duration = segment['end'] - segment['start']
                total_duration = max(total_duration, segment['end'])
                
                segments.append({
                    'start': str(timedelta(seconds=segment['start'])),
                    'end': str(timedelta(seconds=segment['end'])),
                    'text': segment['text'],
                    'keywords': self.extract_keywords(segment['text'])
                })
            
            # Create context object
            context = {
                'video_name': os.path.basename(video_path),
                'duration': str(timedelta(seconds=total_duration)),
                'segments': segments,
                'all_keywords': list(set(
                    keyword for segment in segments for keyword in segment['keywords']
                ))
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {str(e)}")
            logger.error("Full error traceback:")
            logger.error(traceback.format_exc())
            return None
    
    def save_context(self, context: Dict[str, Any], video_name: str):
        """Save the context to a JSON file."""
        try:
            # Sanitize the filename
            safe_name = self.sanitize_filename(Path(video_name).stem)
            output_file = os.path.join(self.output_dir, f"{safe_name}_context.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(context, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved context to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving context for {video_name}: {str(e)}")
            logger.error(traceback.format_exc())
    
    def process_all_videos(self):
        """Process all videos in the downloads directory."""
        video_files = self.get_video_files()
        
        if not video_files:
            logger.warning("No video files found in the downloads directory")
            return
        
        logger.info(f"Found {len(video_files)} video files")
        
        for video_file in video_files:
            context = self.process_video(video_file)
            if context:
                self.save_context(context, video_file)

def main():
    try:
        # Initialize the extractor with the base model
        extractor = VideoContextExtractor(model_name="base")
        
        # Process all videos
        extractor.process_all_videos()
    except Exception as e:
        logger.error("Fatal error in main:")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 