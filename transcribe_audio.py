import whisper_timestamped as whisper
import json
import argparse
from typing import Dict, Any, Optional

def transcribe_audio_with_timestamps(
    audio_path: str,
    model_name: str = "base",
    language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Transcribe an audio file with word-level timestamps using whisper-timestamped.
    
    Args:
        audio_path (str): Path to the audio file
        model_name (str): Whisper model to use (tiny, base, small, medium, large)
        language (str, optional): Language code (e.g., 'en', 'fr', 'es'). If None, will auto-detect.
    
    Returns:
        Dict[str, Any]: Transcription result with timestamps
    """
    # Load the model
    model = whisper.load_model(model_name)
    
    # Transcribe the audio
    result = whisper.transcribe(model, audio_path, language=language)
    
    return result

def format_transcription(result: Dict[str, Any]) -> str:
    """
    Format the transcription result into a readable string with timestamps.
    
    Args:
        result (Dict[str, Any]): Transcription result from transcribe_audio_with_timestamps
    
    Returns:
        str: Formatted transcription with timestamps
    """
    formatted_text = []
    
    for segment in result["segments"]:
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        text = segment["text"]
        formatted_text.append(f"[{start_time} --> {end_time}] {text}")
    
    return "\n".join(formatted_text)

def format_timestamp(seconds: float) -> str:
    """
    Format seconds into HH:MM:SS.mmm format.
    
    Args:
        seconds (float): Time in seconds
    
    Returns:
        str: Formatted timestamp
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Transcribe audio file with timestamps using Whisper")
    parser.add_argument("-a", "--audio_path", help="Path to the audio file to transcribe")
    parser.add_argument(
        "-m",
        "--model", 
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to use (default: base)"
    )
    parser.add_argument(
        "-l",
        "--language",
        help="Language code (e.g., 'en', 'fr', 'es'). If not specified, will auto-detect."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="transcription_result.json",
        help="Path to save the JSON output (default: transcription_result.json)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Transcribe the audio
    result = transcribe_audio_with_timestamps(
        audio_path=args.audio_path,
        model_name=args.model,
        language=args.language
    )
    
    # Print formatted transcription
    print("\nTranscription with timestamps:")
    print("-" * 50)
    print(format_transcription(result))
    
    # Save raw result to JSON file
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nRaw transcription saved to: {args.output}") 