import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoMashupCreator:
    def __init__(self, min_clip_duration: int = 5, max_clip_duration: int = 20):
        """Initialize the video mashup creator."""
        self.min_clip_duration = min_clip_duration
        self.max_clip_duration = max_clip_duration
        
        # Initialize the sentence transformer model
        logger.info("Loading sentence transformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and efficient model
        logger.info("Model loaded successfully")
        
        # Create output directory
        self.output_dir = "mashups"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get the embedding vector for a text using the sentence transformer model."""
        return self.model.encode(text, convert_to_numpy=True)
    
    def load_transcription(self, transcription_file: str) -> str:
        """Load the target transcription from a JSON file."""
        try:
            with open(transcription_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('text', '')
        except Exception as e:
            logger.error(f"Error loading transcription file: {str(e)}")
            raise
    
    def load_video_contexts(self, contexts_dir: str = "contexts") -> List[Dict[str, Any]]:
        """Load all video context files from the contexts directory."""
        contexts = []
        try:
            for file in os.listdir(contexts_dir):
                if file.endswith('_context.json'):
                    with open(os.path.join(contexts_dir, file), 'r', encoding='utf-8') as f:
                        context = json.load(f)
                        context['video_path'] = os.path.join('downloads', context['video_name'])
                        contexts.append(context)
            return contexts
        except Exception as e:
            logger.error(f"Error loading video contexts: {str(e)}")
            raise
    
    def time_to_seconds(self, time_str: str) -> float:
        """Convert time string (HH:MM:SS) to seconds."""
        h, m, s = map(float, time_str.split(':'))
        return h * 3600 + m * 60 + s
    
    def find_matching_clips(
        self,
        target_transcription: str,
        contexts: List[Dict[str, Any]],
        similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Find video clips that match the target transcription using semantic similarity."""
        logger.info(f"Analyzing {len(contexts)} videos for matching clips")
        logger.info(f"Target transcription: {target_transcription[:200]}...")
        
        # Get embedding for target transcription
        target_embedding = self.get_embedding(target_transcription)
        
        # Prepare all segments for comparison
        all_segments = []
        for context in contexts:
            if 'segments' not in context:
                logger.warning(f"No segments found in context for {context['video_path']}")
                continue
            logger.info(f"Processing video: {context['video_path']}")
            for segment in context['segments']:
                if not all(key in segment for key in ['text', 'start', 'end']):
                    logger.warning(f"Invalid segment format in {context['video_path']}")
                    continue
                all_segments.append({
                    'text': segment['text'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'video_path': context['video_path']
                })
        
        if not all_segments:
            logger.error("No valid segments found in any context")
            return []
        
        logger.info(f"Found {len(all_segments)} total segments to analyze")
        
        # Calculate semantic similarity scores
        segment_texts = [segment['text'] for segment in all_segments]
        segment_embeddings = self.model.encode(segment_texts, convert_to_numpy=True)
        similarity_scores = cosine_similarity([target_embedding], segment_embeddings)[0]
        
        # Log similarity score distribution
        max_score = max(similarity_scores)
        min_score = min(similarity_scores)
        avg_score = sum(similarity_scores) / len(similarity_scores)
        logger.info(f"Semantic similarity scores - Max: {max_score:.3f}, Min: {min_score:.3f}, Avg: {avg_score:.3f}")
        
        # Group segments by video and sort by similarity
        video_segments = {}
        for score, segment in zip(similarity_scores, all_segments):
            if score > similarity_threshold:
                video_path = segment['video_path']
                if video_path not in video_segments:
                    video_segments[video_path] = []
                video_segments[video_path].append((score, segment))
                logger.info(f"Found matching segment in {video_path} with score {score:.3f}")
                logger.info(f"Segment text: {segment['text'][:100]}...")
        
        if not video_segments:
            logger.warning(f"No segments found above threshold {similarity_threshold}")
            logger.info("Trying with lower threshold...")
            # Try with a lower threshold
            for score, segment in zip(similarity_scores, all_segments):
                if score > (similarity_threshold * 0.7):  # Try 70% of the original threshold
                    video_path = segment['video_path']
                    if video_path not in video_segments:
                        video_segments[video_path] = []
                    video_segments[video_path].append((score, segment))
                    logger.info(f"Found matching segment with lower threshold in {video_path} with score {score:.3f}")
        
        # Select top clips from each video
        matching_clips = []
        max_clips_per_video = 2
        
        for video_path, segments in video_segments.items():
            segments.sort(key=lambda x: x[0], reverse=True)
            logger.info(f"Processing {len(segments)} segments from {video_path}")
            
            for score, segment in segments[:max_clips_per_video]:
                try:
                    start_time = self.time_to_seconds(segment['start'])
                    end_time = self.time_to_seconds(segment['end'])
                    original_duration = end_time - start_time
                    
                    # Generate random duration between min and max
                    target_duration = np.random.uniform(self.min_clip_duration, self.max_clip_duration)
                    
                    # If original segment is longer than target duration, randomly select a portion
                    if original_duration > target_duration:
                        max_start = end_time - target_duration
                        start_time = start_time + (max_start - start_time) * np.random.random()
                        end_time = start_time + target_duration
                    # If original segment is shorter than min duration, extend it
                    elif original_duration < self.min_clip_duration:
                        extension = (target_duration - original_duration) / 2
                        start_time = max(0, start_time - extension)
                        end_time = end_time + extension
                    
                    logger.info(f"Clip duration: {end_time - start_time:.2f} seconds")
                    
                    matching_clips.append({
                        'video_path': video_path,
                        'start_time': start_time,
                        'end_time': end_time,
                        'similarity_score': score,
                        'text': segment['text']
                    })
                    logger.info(f"Added clip from {video_path} with score {score:.3f}")
                except Exception as e:
                    logger.error(f"Error processing segment: {str(e)}")
                    continue
        
        # Randomly shuffle the clips
        np.random.shuffle(matching_clips)
        
        if matching_clips:
            logger.info(f"Successfully found {len(matching_clips)} matching clips")
        else:
            logger.error("No matching clips found even with lower threshold")
            
        return matching_clips
    
    def create_mashup(
        self,
        matching_clips: List[Dict[str, Any]],
        audio_file: str,
        output_name: str
    ) -> str:
        """Create a video mashup from matching clips with the provided audio."""
        try:
            if not matching_clips:
                raise ValueError("No matching clips provided")
            
            # Load audio
            logger.info(f"Loading audio file: {audio_file}")
            audio = AudioFileClip(audio_file)
            audio_duration = audio.duration
            
            # Create video clips
            video_clips = []
            current_duration = 0
            
            for i, clip in enumerate(matching_clips):
                if current_duration >= audio_duration:
                    break
                
                try:
                    # Load video clip
                    video = VideoFileClip(clip['video_path'])
                    segment = video.subclip(clip['start_time'], clip['end_time'])
                    
                    # Resize if needed
                    if segment.size[0] > 1920 or segment.size[1] > 1080:
                        segment = segment.resize(width=1920)
                    
                    # Add crossfade effect
                    if i > 0:
                        segment = segment.crossfadein(0.5)
                    
                    video_clips.append(segment)
                    current_duration += segment.duration
                    
                except Exception as e:
                    logger.error(f"Error processing clip {i+1}: {str(e)}")
                    continue
            
            if not video_clips:
                raise ValueError("No valid video clips could be created")
            
            # Create final video
            logger.info("Creating final video...")
            final_video = concatenate_videoclips(video_clips, method="compose")
            
            # Trim to match audio duration
            if final_video.duration > audio_duration:
                final_video = final_video.subclip(0, audio_duration)
            
            # Set audio
            final_video = final_video.set_audio(audio)
            
            # Write output file
            output_path = os.path.join(self.output_dir, f"{output_name}.mp4")
            logger.info(f"Writing output file: {output_path}")
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                fps=24,
                preset='medium',
                threads=4,
                bitrate='5000k'
            )
            
            # Cleanup
            final_video.close()
            audio.close()
            for clip in video_clips:
                clip.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating video mashup: {str(e)}")
            logger.error(traceback.format_exc())
            raise

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create video mashup from matching clips")
    parser.add_argument(
        "-t",
        "--transcription",
        default="transcription_result.json",
        help="Path to the target transcription file"
    )
    parser.add_argument(
        "-a",
        "--audio",
        default="audio.mp3",
        help="Path to the audio file"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="mashup",
        help="Output video name (without extension)"
    )
    parser.add_argument(
        "-s",
        "--similarity",
        type=float,
        default=0.3,
        help="Similarity threshold for matching clips (0.0 to 1.0)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize creator
        creator = VideoMashupCreator()
        
        # Load transcription
        logger.info("Loading transcription...")
        transcription = creator.load_transcription(args.transcription)
        
        # Load video contexts
        logger.info("Loading video contexts...")
        contexts = creator.load_video_contexts()
        
        # Find matching clips
        logger.info("Finding matching clips...")
        matching_clips = creator.find_matching_clips(
            transcription,
            contexts,
            args.similarity
        )
        
        if not matching_clips:
            logger.error("No matching clips found!")
            return
        
        logger.info(f"Found {len(matching_clips)} matching clips")
        
        # Create mashup
        logger.info("Creating video mashup...")
        output_path = creator.create_mashup(
            matching_clips,
            args.audio,
            args.output
        )
        
        logger.info(f"Mashup created successfully: {output_path}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 