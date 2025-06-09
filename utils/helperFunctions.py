from utils.simpleTracker import SimpleTracker
from collections import defaultdict
import cv2
import numpy as np

def get_sign_position(bbox, image_width, image_height):
    """Determine if sign is left, right, or top based on highway context"""
    x1, y1, x2, y2 = bbox
    
    # Calculate center point of the bounding box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Highway-specific position logic
    # Top: Signs that are overhead/above the road (upper portion of image)
    if center_y < image_height * 0.4:  # Upper 40% of image
        return "iznad"
    
    # Left vs Right: Based on which side of the highway
    # Use a more central division since highway signs are typically on sides
    elif center_x < image_width * 0.45:  # Left side (with some margin)
        return "levo"
    elif center_x > image_width * 0.55:  # Right side (with some margin)
        return "desno"
    else:
        # Signs in the very center could be overhead/gantry signs
        return "iznad"

def extract_unique_signs_ordered(model, video_path, conf=0.5, vid_stride=2, imgsz=640, 
                                max_distance=100, max_disappeared=15, min_appearances=3):
    """
    Extract unique traffic signs in order of appearance with robust tracking
    
    Args:
        max_distance: Maximum distance to consider detections as same sign
        max_disappeared: Frames an object can be missing before being removed
        min_appearances: Minimum appearances to consider a sign as valid
    
    Returns:
        list: Unique signs in order of appearance
        dict: Class counts
        list: Unique class names
    """
    
    # Initialize tracker
    tracker = SimpleTracker(max_disappeared=max_disappeared, max_distance=max_distance)
    
    # Get predictions
    results = model.predict(
        source=video_path,
        conf=conf,
        vid_stride=vid_stride,
        imgsz=imgsz,
        stream=True,
        save=False
    )
    
    frame_count = 0
    height = None
    width = None
    
    for result in results:
        frame_count += 1
        
        detections = []

        if width is None and height is None:
            height = result.orig_shape[0]
            width = result.orig_shape[1]
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_indices = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_idx in zip(boxes, confidences, class_indices):
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                class_name = result.names[cls_idx]
                position = get_sign_position([x1, y1, x2, y2], width, height)

                detection = {
                    'class': class_name,
                    'position': position,
                    'center': (center_x, center_y),
                    'box': box,
                    'confidence': conf
                }
                detections.append(detection)
        
        # Update tracker
        tracker.update(detections, frame_count)
    
    # Filter signs by minimum appearances
    valid_signs = [sign for sign in tracker.unique_signs 
                  if sign['appearances'] >= min_appearances]
    
    # Sort by first appearance (already in order, but just to be sure)
    valid_signs.sort(key=lambda x: x['first_seen'])
    
    # Get class counts and unique classes
    class_counts = defaultdict(int)
    for sign in valid_signs:
        class_counts[sign['class']] += 1
    
    unique_classes = list(set(sign['class'] for sign in valid_signs))
    
    return valid_signs

def get_frame(video_path: str, frame_number: int):
    """
    Simple function to get a frame from video.
    
    Args:
        video_path: Path to video file
        frame_number: Frame number to extract
        
    Returns:
        Frame as numpy array (RGB) or None if failed
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None