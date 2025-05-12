import os
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from datetime import datetime
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
import face_recognition
import pickle
from pathlib import Path

# Add known faces database
KNOWN_FACES_DIR = "known_faces"
KNOWN_FACES_FILE = "known_faces.pkl"

def load_known_faces():
    """Load known faces database or create if it doesn't exist."""
    if os.path.exists(KNOWN_FACES_FILE):
        with open(KNOWN_FACES_FILE, 'rb') as f:
            return pickle.load(f)
    else:
        # Create known faces directory if it doesn't exist
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        return {}

def save_known_faces(known_faces):
    """Save known faces database."""
    with open(KNOWN_FACES_FILE, 'wb') as f:
        pickle.dump(known_faces, f)

def add_known_face(name: str, image_path: str):
    """Add a new known face to the database."""
    known_faces = load_known_faces()
    
    # Load image and get face encoding
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    
    if face_encodings:
        known_faces[name] = face_encodings[0]
        save_known_faces(known_faces)
        print(f"Added {name} to known faces database")
    else:
        print(f"No face found in {image_path}")

def detect_faces(frame: np.ndarray, known_faces: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """
    Detect and recognize faces in a frame.
    
    Args:
        frame: Video frame as numpy array
        known_faces: Dictionary of known face encodings
    
    Returns:
        List of detected faces with names and confidence scores
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    face_locations = face_recognition.face_locations(frame_rgb)
    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
    
    detected_faces = []
    for face_encoding in face_encodings:
        # Compare with known faces
        matches = face_recognition.compare_faces(
            list(known_faces.values()),
            face_encoding,
            tolerance=0.6
        )
        
        # Get face distances
        face_distances = face_recognition.face_distance(
            list(known_faces.values()),
            face_encoding
        )
        
        # Get best match
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = list(known_faces.keys())[best_match_index]
                confidence = 1 - face_distances[best_match_index]
                detected_faces.append({
                    "name": name,
                    "confidence": confidence
                })
    
    return detected_faces

def analyze_frame(frame: np.ndarray, model, transform, known_faces: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Analyze a frame using ResNet50 and face recognition.
    
    Args:
        frame: Video frame as numpy array
        model: Pre-trained ResNet50 model
        transform: Image transformation pipeline
        known_faces: Dictionary of known face encodings
    
    Returns:
        Dictionary containing scene analysis and detected faces
    """
    # Scene analysis
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = transform(frame_rgb)
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    top10_prob, top10_catid = torch.topk(probabilities, 10)
    
    with open('imagenet_classes.txt') as f:
        categories = [s.strip() for s in f.readlines()]
    
    scene_predictions = []
    for i in range(10):
        confidence = top10_prob[i].item()
        if confidence > 0.05:
            scene_predictions.append({
                "label": categories[top10_catid[i].item()],
                "confidence": confidence
            })
    
    # Face detection and recognition
    detected_faces = detect_faces(frame, known_faces)
    
    return {
        "scene": scene_predictions,
        "faces": detected_faces
    }

def extract_frames(video_path: str, frame_interval: int = 30) -> List[Dict[str, Any]]:
    """
    Extract frames from video at specified intervals.
    
    Args:
        video_path: Path to the video file
        frame_interval: Extract one frame every N frames
    
    Returns:
        List of frames with timestamps
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception("Error opening video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            frames.append({
                "timestamp": timestamp,
                "frame": frame
            })
        
        frame_count += 1
    
    cap.release()
    return frames

def extract_context(
    video_path: str,
    frame_interval: int = 30,
    segment_duration: int = 30
) -> List[Dict[str, Any]]:
    """
    Extract context from video using computer vision and face recognition.
    """
    try:
        # Load pre-trained model
        print("Loading vision model...")
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.eval()
        
        # Load known faces
        print("Loading known faces database...")
        known_faces = load_known_faces()
        
        # Define image transformation
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Extract frames
        print("Extracting frames...")
        frames = extract_frames(video_path, frame_interval)
        
        # Process frames into segments
        context_segments = []
        current_segment = {
            "start_time": 0,
            "end_time": 0,
            "content": [],
            "keywords": {},
            "faces": {}  # Track faces with their confidence scores
        }
        
        for frame_data in frames:
            timestamp = frame_data["timestamp"]
            frame = frame_data["frame"]
            
            # Analyze frame
            analysis = analyze_frame(frame, model, transform, known_faces)
            
            if timestamp - current_segment["start_time"] > segment_duration:
                if current_segment["content"]:
                    # Sort keywords by average confidence
                    sorted_keywords = sorted(
                        current_segment["keywords"].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    current_segment["keywords"] = dict(sorted_keywords)
                    
                    # Sort faces by average confidence
                    sorted_faces = sorted(
                        current_segment["faces"].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    current_segment["faces"] = dict(sorted_faces)
                    
                    context_segments.append(current_segment)
                
                current_segment = {
                    "start_time": timestamp,
                    "end_time": timestamp,
                    "content": [analysis],
                    "keywords": {},
                    "faces": {}
                }
            else:
                current_segment["content"].append(analysis)
                current_segment["end_time"] = timestamp
                
                # Update scene keywords
                for pred in analysis["scene"]:
                    label = pred["label"]
                    confidence = pred["confidence"]
                    if label in current_segment["keywords"]:
                        current_segment["keywords"][label] = (current_segment["keywords"][label] + confidence) / 2
                    else:
                        current_segment["keywords"][label] = confidence
                
                # Update detected faces
                for face in analysis["faces"]:
                    name = face["name"]
                    confidence = face["confidence"]
                    if name in current_segment["faces"]:
                        current_segment["faces"][name] = (current_segment["faces"][name] + confidence) / 2
                    else:
                        current_segment["faces"][name] = confidence
        
        # Add the last segment if it's not empty
        if current_segment["content"]:
            sorted_keywords = sorted(
                current_segment["keywords"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            current_segment["keywords"] = dict(sorted_keywords)
            
            sorted_faces = sorted(
                current_segment["faces"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            current_segment["faces"] = dict(sorted_faces)
            
            context_segments.append(current_segment)
        
        return context_segments
        
    except Exception as e:
        raise Exception(f"Error extracting context: {str(e)}")

def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS format."""
    return str(datetime.utcfromtimestamp(seconds).strftime('%H:%M:%S'))

def save_context(context_segments: List[Dict[str, Any]], output_file: str):
    """Save context segments to a file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for segment in context_segments:
            f.write(f"[{format_timestamp(segment['start_time'])} - {format_timestamp(segment['end_time'])}]\n")
            
            # Write detected faces
            if segment["faces"]:
                f.write("Detected People:\n")
                for name, confidence in segment["faces"].items():
                    f.write(f"  - {name} ({confidence:.2%})\n")
                f.write("\n")
            
            # Write scene content
            f.write("Visual Content:\n")
            for i, content in enumerate(segment["content"]):
                f.write(f"  Frame {i+1}:\n")
                for pred in content["scene"]:
                    f.write(f"    - {pred['label']} ({pred['confidence']:.2%})\n")
            
            f.write("\nTop Keywords (with confidence):\n")
            for keyword, confidence in list(segment["keywords"].items())[:10]:
                f.write(f"  - {keyword} ({confidence:.2%})\n")
            f.write("-" * 80 + "\n")

def get_video_files(directory: str = "downloads") -> List[str]:
    """Get all video files from the specified directory."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    video_files = []
    
    for file in os.listdir(directory):
        if os.path.splitext(file)[1].lower() in video_extensions:
            video_files.append(os.path.join(directory, file))
    
    return video_files

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract visual context from videos in downloads directory")
    parser.add_argument("-v", "--video", help="Specific video file in downloads directory to process")
    parser.add_argument("-f", "--frame_interval", type=int, default=30,
                      help="Extract one frame every N frames")
    parser.add_argument("-s", "--segment_duration", type=int, default=30,
                      help="Duration in seconds for each context segment")
    parser.add_argument("-o", "--output_dir", default="contexts",
                      help="Directory to save context outputs")
    parser.add_argument("--add_face", nargs=2, metavar=("NAME", "IMAGE_PATH"),
                      help="Add a new face to the known faces database")
    
    args = parser.parse_args()
    
    try:
        # Handle adding new face
        if args.add_face:
            name, image_path = args.add_face
            add_known_face(name, image_path)
            exit(0)
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Get video files to process
        if args.video:
            video_files = [os.path.join("downloads", args.video)]
            if not os.path.exists(video_files[0]):
                raise FileNotFoundError(f"Video file not found: {video_files[0]}")
        else:
            video_files = get_video_files()
            if not video_files:
                raise FileNotFoundError("No video files found in downloads directory")
        
        # Process each video
        for video_path in video_files:
            print(f"\nProcessing: {os.path.basename(video_path)}")
            
            # Extract context
            context_segments = extract_context(
                video_path=video_path,
                frame_interval=args.frame_interval,
                segment_duration=args.segment_duration
            )
            
            # Save context
            output_file = os.path.join(
                args.output_dir,
                f"{os.path.splitext(os.path.basename(video_path))[0]}_context.txt"
            )
            save_context(context_segments, output_file)
            print(f"Context saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}") 