import os
import requests
from bs4 import BeautifulSoup
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
import random

# Create directories
KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Wikipedia pages to scrape for famous people
WIKI_PAGES = [
    "https://en.wikipedia.org/wiki/List_of_actors",
    "https://en.wikipedia.org/wiki/List_of_actresses",
    "https://en.wikipedia.org/wiki/List_of_musicians",
    "https://en.wikipedia.org/wiki/List_of_sportspeople",
    "https://en.wikipedia.org/wiki/List_of_politicians",
    "https://en.wikipedia.org/wiki/List_of_scientists",
    "https://en.wikipedia.org/wiki/List_of_businesspeople"
]

def get_famous_people_from_wiki() -> Dict[str, str]:
    """Scrape Wikipedia pages to get a list of famous people and their image URLs."""
    famous_people = {}
    
    for wiki_page in WIKI_PAGES:
        try:
            print(f"\nScraping {wiki_page}...")
            response = requests.get(wiki_page)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all links to people's Wikipedia pages
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if href.startswith('/wiki/') and not any(x in href for x in ['Category:', 'Wikipedia:', 'Help:', 'File:', 'Template:']):
                    name = link.text.strip()
                    if name and len(name) > 2:  # Basic name validation
                        # Get the person's Wikipedia page
                        person_url = f"https://en.wikipedia.org{href}"
                        try:
                            person_response = requests.get(person_url)
                            person_response.raise_for_status()
                            person_soup = BeautifulSoup(person_response.text, 'html.parser')
                            
                            # Find the person's image
                            image_link = person_soup.find('a', {'class': 'image'})
                            if image_link:
                                image_url = f"https://en.wikipedia.org{image_link['href']}"
                                famous_people[name] = image_url
                                print(f"Found {name}")
                            
                            # Be nice to Wikipedia
                            time.sleep(0.5)
                            
                        except Exception as e:
                            print(f"Error processing {name}: {str(e)}")
                            continue
            
            # Be nice to Wikipedia
            time.sleep(1)
            
        except Exception as e:
            print(f"Error scraping {wiki_page}: {str(e)}")
            continue
    
    return famous_people

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
        print(f"Error downloading {url}: {str(e)}")
        return False

def get_image_url(wiki_url: str) -> str:
    """Extract the direct image URL from a Wikimedia Commons page."""
    try:
        response = requests.get(wiki_url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find the original file link
        file_link = soup.find('a', {'class': 'internal'})
        if file_link:
            return f"https://commons.wikimedia.org{file_link['href']}"
    except Exception as e:
        print(f"Error getting image URL from {wiki_url}: {str(e)}")
    return None

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
        print(f"Error processing {image_path}: {str(e)}")
        return False

def process_person(name: str, image_url: str) -> bool:
    """Process a single person's image."""
    try:
        # Create filename
        filename = f"{name.lower().replace(' ', '_')}.jpg"
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        
        # Skip if already exists
        if os.path.exists(image_path):
            return True
        
        # Get direct image URL
        direct_url = get_image_url(image_url)
        if not direct_url:
            return False
        
        # Download image
        if not download_image(direct_url, image_path):
            return False
        
        # Process image
        if not process_image(image_path):
            if os.path.exists(image_path):
                os.remove(image_path)
            return False
        
        return True
    except Exception as e:
        print(f"Error processing {name}: {str(e)}")
        return False

def download_famous_faces():
    """Download and process images of famous people."""
    # Get list of famous people
    print("Getting list of famous people from Wikipedia...")
    famous_people = get_famous_people_from_wiki()
    
    # Limit to 5000 people
    if len(famous_people) > 5000:
        famous_people = dict(random.sample(list(famous_people.items()), 5000))
    
    print(f"\nFound {len(famous_people)} famous people")
    
    # Process people in parallel
    successful_downloads = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for name, url in famous_people.items():
            futures.append(executor.submit(process_person, name, url))
        
        # Show progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            if future.result():
                successful_downloads.append(name)
    
    return successful_downloads

def add_faces_to_database(successful_downloads: List[str]):
    """Add successfully downloaded faces to the face recognition database."""
    known_faces = {}
    
    for name in tqdm(successful_downloads, desc="Adding faces to database"):
        filename = f"{name.lower().replace(' ', '_')}.jpg"
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        
        try:
            # Load image and get face encoding
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                known_faces[name] = face_encodings[0]
        except Exception as e:
            print(f"Error processing {name}'s face: {str(e)}")
    
    # Save face database
    if known_faces:
        with open('known_faces.pkl', 'wb') as f:
            pickle.dump(known_faces, f)
        print(f"\nSuccessfully added {len(known_faces)} faces to the database")
    else:
        print("\nNo faces were added to the database")

if __name__ == "__main__":
    print("Downloading famous faces...")
    successful_downloads = download_famous_faces()
    
    if successful_downloads:
        print("\nAdding faces to database...")
        add_faces_to_database(successful_downloads)
    else:
        print("\nNo faces were successfully downloaded") 