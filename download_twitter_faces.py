import os
import requests
import json
import time
from pathlib import Path
import face_recognition
import cv2
import numpy as np
from typing import List, Dict, Any
import pickle
import concurrent.futures
from tqdm import tqdm
import tweepy
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directories
KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

class TwitterFamousPeople:
    def __init__(self, bearer_token: str):
        """Initialize Twitter API client."""
        self.client = tweepy.Client(bearer_token=bearer_token)
        self.api = tweepy.API(tweepy.OAuth2BearerHandler(bearer_token))
        
    def get_top_accounts(self, limit: int = 5000) -> List[Dict[str, Any]]:
        """Get top Twitter accounts by follower count."""
        logger.info("Fetching top Twitter accounts...")
        
        # List of verified accounts to start with
        seed_accounts = [
            'elonmusk', 'BarackObama', 'justinbieber', 'katyperry', 'rihanna',
            'Cristiano', 'taylorswift13', 'ladygaga', 'TheEllenShow', 'YouTube',
            'instagram', 'shakira', 'jtimberlake', 'KimKardashian', 'britneyspears',
            'ArianaGrande', 'selenagomez', 'ddlovato', 'jlo', 'MileyCyrus',
            'narendramodi', 'BillGates', 'Oprah', 'KingJames', 'neymarjr'
        ]
        
        famous_accounts = []
        processed_ids = set()
        
        # Process seed accounts first
        for username in seed_accounts:
            try:
                user = self.client.get_user(username=username)
                if user.data:
                    famous_accounts.append({
                        'id': user.data.id,
                        'username': user.data.username,
                        'name': user.data.name,
                        'followers_count': user.data.public_metrics['followers_count']
                    })
                    processed_ids.add(user.data.id)
            except Exception as e:
                logger.error(f"Error processing {username}: {str(e)}")
        
        # Get followers of seed accounts
        for account in tqdm(famous_accounts[:10], desc="Getting followers of seed accounts"):
            try:
                followers = self.client.get_users_followers(
                    account['id'],
                    max_results=100,
                    user_fields=['public_metrics', 'profile_image_url']
                )
                
                for follower in followers.data or []:
                    if len(famous_accounts) >= limit:
                        break
                        
                    if follower.id not in processed_ids:
                        famous_accounts.append({
                            'id': follower.id,
                            'username': follower.username,
                            'name': follower.name,
                            'followers_count': follower.public_metrics['followers_count']
                        })
                        processed_ids.add(follower.id)
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error getting followers for {account['username']}: {str(e)}")
        
        # Sort by follower count
        famous_accounts.sort(key=lambda x: x['followers_count'], reverse=True)
        return famous_accounts[:limit]
    
    def get_profile_image(self, username: str) -> str:
        """Get high-resolution profile image URL for a Twitter user."""
        try:
            user = self.client.get_user(
                username=username,
                user_fields=['profile_image_url']
            )
            
            if user.data and user.data.profile_image_url:
                # Get the original size image
                return user.data.profile_image_url.replace('_normal', '')
        except Exception as e:
            logger.error(f"Error getting profile image for {username}: {str(e)}")
        return None

def download_image(url: str, save_path: str) -> bool:
    """Download an image from a URL and save it to the specified path."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return False

def process_image(image_path: str) -> bool:
    """Process an image to ensure it's suitable for face recognition."""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return False
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(image_rgb)
        if not face_locations:
            return False
        
        # Get the largest face
        face_sizes = [(bottom - top) * (right - left) for top, right, bottom, left in face_locations]
        largest_face_idx = np.argmax(face_sizes)
        top, right, bottom, left = face_locations[largest_face_idx]
        
        # Add padding
        height, width = image_rgb.shape[:2]
        padding = int(max(right - left, bottom - top) * 0.2)
        top = max(0, top - padding)
        bottom = min(height, bottom + padding)
        left = max(0, left - padding)
        right = min(width, right + padding)
        
        # Crop and resize
        face_image = image_rgb[top:bottom, left:right]
        face_image = cv2.resize(face_image, (224, 224))
        
        # Save processed image
        cv2.imwrite(image_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
        return True
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return False

def process_person(name: str, username: str, image_url: str) -> bool:
    """Process a single person's image."""
    try:
        # Create filename
        filename = f"{username.lower()}.jpg"
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        
        # Skip if already exists
        if os.path.exists(image_path):
            return True
        
        # Download image
        if not download_image(image_url, image_path):
            return False
        
        # Process image
        if not process_image(image_path):
            if os.path.exists(image_path):
                os.remove(image_path)
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error processing {name}: {str(e)}")
        return False

def download_famous_faces(bearer_token: str):
    """Download and process images of famous people from Twitter."""
    # Initialize Twitter client
    twitter = TwitterFamousPeople(bearer_token)
    
    # Get top accounts
    famous_accounts = twitter.get_top_accounts()
    logger.info(f"Found {len(famous_accounts)} famous accounts")
    
    # Process accounts in parallel
    successful_downloads = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for account in famous_accounts:
            image_url = twitter.get_profile_image(account['username'])
            if image_url:
                futures.append(executor.submit(
                    process_person,
                    account['name'],
                    account['username'],
                    image_url
                ))
        
        # Show progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            if future.result():
                successful_downloads.append(account['username'])
    
    return successful_downloads

def add_faces_to_database(successful_downloads: List[str]):
    """Add successfully downloaded faces to the face recognition database."""
    known_faces = {}
    
    for username in tqdm(successful_downloads, desc="Adding faces to database"):
        filename = f"{username.lower()}.jpg"
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        
        try:
            # Load image and get face encoding
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                known_faces[username] = face_encodings[0]
        except Exception as e:
            logger.error(f"Error processing {username}'s face: {str(e)}")
    
    # Save face database
    if known_faces:
        with open('known_faces.pkl', 'wb') as f:
            pickle.dump(known_faces, f)
        logger.info(f"Successfully added {len(known_faces)} faces to the database")
    else:
        logger.warning("No faces were added to the database")

if __name__ == "__main__":
    # Get Twitter API bearer token from environment variable
    bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
    if not bearer_token:
        logger.error("Please set TWITTER_BEARER_TOKEN environment variable")
        exit(1)
    
    logger.info("Downloading famous faces from Twitter...")
    successful_downloads = download_famous_faces(bearer_token)
    
    if successful_downloads:
        logger.info("Adding faces to database...")
        add_faces_to_database(successful_downloads)
    else:
        logger.warning("No faces were successfully downloaded") 