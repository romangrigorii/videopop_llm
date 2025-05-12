# Video Context Extractor

A tool that extracts visual context from videos using computer vision, identifying objects, scenes, and famous people in each frame.

## Installation

1. Install the required dependencies:
```bash
pip install opencv-python torch torchvision numpy face_recognition
```

2. Download ImageNet class names:
```bash
wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

## Usage

### Adding Known Faces

Before processing videos, you need to add known faces to the database:

```bash
python video_context_extractor.py --add_face "Person Name" path/to/face/image.jpg
```

You can add multiple faces by running the command multiple times with different names and images.

### Command Line Usage

```bash
# Process all videos in downloads directory
python video_context_extractor.py [options]

# Process a specific video in downloads directory
python video_context_extractor.py -v video_name.mp4 [options]
```

Options:
- `-v, --video`: Specific video file in downloads directory to process
- `-f, --frame_interval`: Extract one frame every N frames (default: 30)
- `-s, --segment_duration`: Duration in seconds for each context segment (default: 30)
- `-o, --output_dir`: Directory to save context outputs (default: contexts)
- `--add_face`: Add a new face to the known faces database (requires name and image path)

Examples:
```bash
# Add a known face
python video_context_extractor.py --add_face "John Doe" john_photo.jpg

# Process all videos with default settings
python video_context_extractor.py

# Process specific video with custom frame interval
python video_context_extractor.py -v my_video.mp4 -f 15

# Process all videos with longer segments
python video_context_extractor.py -s 60
```

### Output Format

The context is saved in text files in the output directory (default: contexts) with the following format:
```
[00:00:00 - 00:00:30]
Detected People:
  - John Doe (95.2%)
  - Jane Smith (87.5%)

Visual Content:
  Frame 1:
    - person (95.2%)
    - chair (87.5%)
    - table (82.3%)
    - laptop (78.9%)
    - cup (75.4%)
    - desk (72.1%)
    - computer (68.7%)
    - monitor (65.4%)
    - keyboard (62.3%)
    - office (58.9%)
  Frame 2:
    - person (93.8%)
    - desk (85.6%)
    - computer (81.2%)
    ...

Top Keywords (with confidence):
  - person (94.5%)
  - desk (84.2%)
  - computer (80.1%)
  - chair (87.5%)
  - table (82.3%)
  - laptop (78.9%)
  - cup (75.4%)
  - monitor (65.4%)
  - keyboard (62.3%)
  - office (58.9%)
--------------------------------------------------------------------------------
```

## Features

- Processes videos from the downloads directory
- Uses ResNet50 for object and scene detection
- Face detection and recognition for known people
- Extracts frames at configurable intervals
- Divides content into manageable segments
- Identifies objects, scenes, and people in each frame
- Creates separate context file for each video
- Customizable frame extraction interval
- Customizable segment duration
- High-confidence object detection (threshold > 0.05)
- Face recognition with confidence scores 