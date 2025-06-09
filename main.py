from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import uvicorn
import os
import shutil
from pathlib import Path
from uuid import uuid4
from ultralytics import YOLO
import tempfile
from utils.helperFunctions import extract_unique_signs_ordered, get_frame
from utils.gpmfFunctions import get_all_coords
from utils.trafficSignMapper import TrafficSignGPSMapper
import torch
import boto3
from pydantic import BaseModel
from urllib.parse import urlparse
import requests
from dotenv import load_dotenv

#fastapi dev main.py - CMD

app = FastAPI(title="Traffic Sign Detection API")

load_dotenv()  # Load environment variables from .env file

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize YOLO model
MODEL_PATH = "best.pt"  # Update this to your model path
best_model = None

# Create directory for results
BASE_DIR = Path(__file__).parent  # Gets the directory where main.py is located
RESULTS_DIR = BASE_DIR / "videos"
RESULTS_DIR.mkdir(exist_ok=True)  # Create directory if it doesn't exist

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

s3_client = boto3.client(
    's3',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# Pydantic model for request body
class VideoProcessRequest(BaseModel):
    s3_url: str
    conf: float = 0.75
    id_calculation: int = 0

@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Welcome to the Traffic Sign Detection API!"}

@app.on_event("startup")
async def load_model():
    """Load the YOLO model when the API starts"""
    global best_model
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    best_model = YOLO(MODEL_PATH)
    best_model.to(device)  # Move model to GPU if available
    
    print(f"Model loaded from {MODEL_PATH} on {device}")

def cleanup_files(file_path, result_dir=None):
    """Remove temporary files after request is processed"""
    if os.path.exists(file_path):
        os.remove(file_path)
    if result_dir and os.path.exists(result_dir):
        shutil.rmtree(result_dir)

@app.post("/detect")
async def detect_signs(
    request: VideoProcessRequest,
    background_tasks: BackgroundTasks = None,
):
    """
    Endpoint to detect and crop traffic signs from an uploaded image
    """
    try:
        parsed_url = urlparse(request.s3_url)
        if not parsed_url.scheme in ['http', 'https'] or not parsed_url.netloc:
            raise HTTPException(status_code=400, detail="Invalid S3 URL format.")
        
        # Basic check for S3-like URL structure
        if 's3' not in parsed_url.netloc.lower() and 'amazonaws.com' not in parsed_url.netloc.lower():
            print(f"Warning: URL doesn't appear to be an S3 URL: {request.s3_url}")
        
    except Exception as e:
        print(f"URL parsing error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid S3 URL format: {str(e)}")

    folder_name = f"video-{request.id_calculation}"
    video_dir = RESULTS_DIR / folder_name
    video_dir.mkdir(exist_ok=True)

    output_dir = video_dir / "output"
    output_dir.mkdir(exist_ok=True)
   
    # Extract filename from S3 URL
    s3_filename = Path(parsed_url.path).name
    if not s3_filename:
        s3_filename = f"video_{request.id_calculation}.mp4"
    
    # Download video from S3
    video_path = video_dir / f"downloaded_{s3_filename}"
    
    download_success, download_error = download_video_from_s3(request.s3_url, video_path)
    if not download_success:
        raise HTTPException(status_code=500, detail=f"Failed to download video from S3: {download_error}")

    try:
        unique_signs = extract_unique_signs_ordered(
                best_model, 
                video_path, 
                conf=0.82,
                imgsz=640,
                max_distance=100,      # Adjust based on video resolution/movement
                max_disappeared=15,    # How many frames a sign can be missing
                min_appearances=3      # Minimum detections to be considered valid
            )

        result_signs = []

        for i, sign in enumerate(unique_signs, 1):
            duration = sign['last_seen'] - sign['first_seen'] + 1

            result_signs.append({
                'id': sign['id'],
                'class': sign['class'],
                'position': sign['position'],
                'first_seen': sign['first_seen'],
                'last_seen': sign['last_seen'],
                'duration': duration,
            })

        coord_result = get_all_coords(video_path)
        
        mapper = TrafficSignGPSMapper(fps=30.0, analysis_frame_interval=2)
        mapper.load_sign_data(result_signs)
        mapper.load_gps_data(coord_result)

        mapped_signs = mapper.map_signs_to_gps()

        # Prepare frames data for API
        frames_data = []
        
        for ms in mapped_signs:
            actual_frame = ms['mid_frame'] * 2

            frame = get_frame(video_path, actual_frame)  # Note: using video_path instead of output_dir

            if frame is not None:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Convert frame to base64 for API transmission
                _, buffer = cv2.imencode('.jpg', frame_bgr)
                frame_base64 = buffer.tobytes().hex()  # Convert to hex string for JSON transmission
                
                frames_data.append({
                    'sign_id': ms['sign_id'],
                    'frame_number': actual_frame,
                    'mid_frame': ms['mid_frame'],
                    'gps_data': ms.get('gps_data', {}),
                    'frame_data': frame_base64,
                    'class': ms.get('class', ''),
                    'position': ms.get('position', {})
                })
            else:
                print(f"Failed to get frame for sign {ms['sign_id']}")

        # Send frames to external API
        if frames_data:
            api_response = {"message": "Frames ready to be sent to API", "frames_count": len(frames_data)}
            #api_response = send_frames_to_api(frames_data, request.id_calculation)
        else:
            api_response = {"message": "No frames to send"}

        # Schedule cleanup in background
        if background_tasks:
            background_tasks.add_task(cleanup_files, video_path, video_dir)

        return {
            "message": "Video processed successfully",
            "video_id": request.id_calculation,
            "signs_detected": len(result_signs),
            "frames_sent": len(frames_data),
            "api_response": api_response
        }

    except Exception as e:
        # Clean up on error
        cleanup_files(video_path, video_dir)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

def download_video_from_s3(s3_url: str, destination_path: Path) -> tuple[bool, str]:
    """Download video from S3 URL to local path"""
    try:
        print(f"Downloading video from: {s3_url}")
        print(f"Destination path: {destination_path}")
        
        # Parse S3 URL to extract bucket and key
        parsed_url = urlparse(s3_url)
        
        # Handle both path-style and virtual-hosted-style URLs
        if parsed_url.netloc.startswith('s3.'):
            # Path-style: https://s3.region.amazonaws.com/bucket/key
            path_parts = parsed_url.path.strip('/').split('/', 1)
            bucket_name = path_parts[0]
            object_key = path_parts[1] if len(path_parts) > 1 else ''
        else:
            # Virtual-hosted-style: https://bucket.s3.region.amazonaws.com/key
            bucket_name = parsed_url.netloc.split('.')[0]
            object_key = parsed_url.path.strip('/')
        
        print(f"Bucket: {bucket_name}")
        print(f"Object key: {object_key}")
        
        # Try AWS SDK first (if credentials are available)
        try:
            # Use the pre-configured s3_client
            print("Using AWS SDK to download...")
            
            # Check if we can access the object
            s3_client.head_object(Bucket=bucket_name, Key=object_key)
            
            s3_client.download_file(bucket_name, object_key, str(destination_path))
            
            file_size = destination_path.stat().st_size
            print(f"Downloaded via AWS SDK. File size: {file_size / (1024*1024):.2f} MB")
            return True, "Success"
        
        except Exception as aws_error:
            print(f"AWS SDK failed: {str(aws_error)}")
            print("Falling back to HTTP request...")
        
        # Fallback to HTTP request with various approaches
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Try original URL first
        response = requests.get(s3_url, stream=True, headers=headers, timeout=60)
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 403:
            # Try different URL formats
            alternative_urls = [
                f"https://s3.eu-north-1.amazonaws.com/{bucket_name}/{object_key}",
                f"https://{bucket_name}.s3.amazonaws.com/{object_key}",
                f"https://s3.amazonaws.com/{bucket_name}/{object_key}"
            ]
            
            for alt_url in alternative_urls:
                print(f"Trying alternative URL: {alt_url}")
                response = requests.get(alt_url, stream=True, headers=headers, timeout=60)
                print(f"Alternative URL response: {response.status_code}")
                if response.status_code == 200:
                    break
        
        response.raise_for_status()
        
        # Download the file
        content_length = response.headers.get('content-length')
        if content_length:
            print(f"Expected file size: {int(content_length) / (1024*1024):.2f} MB")
        
        total_size = 0
        with open(destination_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
        
        print(f"Video downloaded successfully to: {destination_path}")
        print(f"Downloaded size: {total_size / (1024*1024):.2f} MB")
        
        # Verify file exists and has content
        if not destination_path.exists():
            return False, "Downloaded file does not exist"
        
        file_size = destination_path.stat().st_size
        if file_size == 0:
            return False, "Downloaded file is empty"
        
        print(f"Final file size: {file_size / (1024*1024):.2f} MB")
        return True, "Success"
        
    except requests.exceptions.RequestException as e:
        error_msg = f"HTTP request error: {str(e)}"
        print(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Error downloading video: {str(e)}"
        print(error_msg)
        return False, error_msg

def send_frames_to_api(frames_data: list, video_id: int):
    """Send frame data to the external API"""
    try:
        payload = {
            "video_id": video_id,
            "frames": frames_data
        }
        
        response = requests.post(
            FRAMES_API_ENDPOINT,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        response.raise_for_status()
        print(f"Successfully sent {len(frames_data)} frames to API")
        return response.json()
    except Exception as e:
        print(f"Error sending frames to API: {str(e)}")
        raise

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)