import cv2
import numpy as np

class SimpleTracker:
    def __init__(self, max_disappeared=10, max_distance=100):
        self.next_id = 0
        self.objects = {}  # id: {info}
        self.disappeared = {}  # id: frame_count
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.unique_signs = []  # Final list in order of appearance
        
    def register(self, detection, frame_num):
        """Register a new object"""
        sign_info = {
            'id': self.next_id,
            'class': detection['class'],
            'position': detection['position'],
            'center': detection['center'],
            'box': detection['box'],
            'confidence': detection['confidence'],
            'first_seen': frame_num,
            'last_seen': frame_num,
            'appearances': 1
        }
        
        self.objects[self.next_id] = sign_info
        self.disappeared[self.next_id] = 0
        
        # Add to unique signs list in order of appearance
        self.unique_signs.append(sign_info)
        
        self.next_id += 1
        
    def deregister(self, object_id):
        """Remove an object that hasn't been seen for too long"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def update(self, detections, frame_num):
        """Update tracker with new detections"""
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return
        
        if len(self.objects) == 0:
            # No existing objects, register all detections
            for detection in detections:
                self.register(detection, frame_num)
        else:
            # Match detections to existing objects
            object_centers = np.array([obj['center'] for obj in self.objects.values()])
            object_ids = list(self.objects.keys())
            detection_centers = np.array([det['center'] for det in detections])
            
            # Calculate distance matrix
            distances = np.linalg.norm(object_centers[:, np.newaxis] - detection_centers, axis=2)
            
            # Find best matches
            used_detection_indices = set()
            used_object_indices = set()
            
            # Sort by distance to get best matches first
            rows, cols = np.where(distances <= self.max_distance)
            distances_valid = distances[rows, cols]
            sorted_indices = np.argsort(distances_valid)
            
            for idx in sorted_indices:
                row, col = rows[idx], cols[idx]
                if row not in used_object_indices and col not in used_detection_indices:
                    # Match found
                    object_id = object_ids[row]
                    detection = detections[col]
                    
                    # Update object
                    self.objects[object_id]['center'] = detection['center']
                    self.objects[object_id]['position'] = detection['position']
                    self.objects[object_id]['box'] = detection['box']
                    self.objects[object_id]['confidence'] = max(
                        self.objects[object_id]['confidence'], 
                        detection['confidence']
                    )
                    self.objects[object_id]['last_seen'] = frame_num
                    self.objects[object_id]['appearances'] += 1
                    
                    # Update in unique_signs list
                    for sign in self.unique_signs:
                        if sign['id'] == object_id:
                            sign.update(self.objects[object_id])
                            break
                    
                    self.disappeared[object_id] = 0
                    
                    used_object_indices.add(row)
                    used_detection_indices.add(col)
            
            # Handle unmatched detections (new objects)
            for i, detection in enumerate(detections):
                if i not in used_detection_indices:
                    self.register(detection, frame_num)
            
            # Handle unmatched objects (disappeared)
            for i, object_id in enumerate(object_ids):
                if i not in used_object_indices:
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)