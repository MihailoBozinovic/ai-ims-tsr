import math
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Tuple

class TrafficSignGPSMapper:
    def __init__(self, fps: float = 30.0, analysis_frame_interval: int = 2):
        """
        Initialize the mapper with video and analysis parameters.
        
        Args:
            fps: Video frames per second
            analysis_frame_interval: Frame interval used for traffic sign analysis (every Nth frame)
        """
        self.fps = fps
        self.analysis_frame_interval = analysis_frame_interval
        self.signs_data = []
        self.gps_data = []
        self.mapped_signs = []
        
    def load_sign_data(self, signs_list: List[Dict]):
        """Load traffic sign detection data from list of dictionaries."""
        self.signs_data = signs_list
        print(f"Loaded {len(signs_list)} traffic sign detections")
        
    def load_gps_data(self, gps_list: List[Dict]):
        """Load GPS data from list of dictionaries."""
        # Convert to DataFrame for easier processing
        self.gps_df = pd.DataFrame(gps_list)
        
        # Convert timestamp strings to datetime objects
        self.gps_df['timestamp'] = pd.to_datetime(self.gps_df['timestamp'])
        
        # Calculate cumulative distance
        self.gps_df['cumulative_distance'] = self.gps_df['distance'].cumsum()
        
        # Sort by timestamp to ensure proper order
        self.gps_df = self.gps_df.sort_values('timestamp').reset_index(drop=True)
        
        self.gps_data = gps_list
        
    def frame_to_timestamp(self, frame_number: int, start_timestamp: datetime) -> datetime:
        """
        Convert frame number to timestamp.
        Since analysis was done every 2nd frame, we need to account for that.
        """
        # Actual frame number in original video
        actual_frame = frame_number * self.analysis_frame_interval
        
        # Time offset from start
        time_offset = actual_frame / self.fps
        
        # Calculate timestamp
        timestamp = start_timestamp + timedelta(seconds=time_offset)
        return timestamp
    
    def find_closest_gps_point(self, target_timestamp: datetime) -> Tuple[int, Dict]:
        """Find the GPS point closest to the target timestamp."""
        # Calculate time differences
        time_diffs = abs(self.gps_df['timestamp'] - target_timestamp)
        closest_idx = time_diffs.idxmin()
        
        return closest_idx, self.gps_df.iloc[closest_idx].to_dict()
    
    def interpolate_gps_position(self, target_timestamp: datetime) -> Dict:
        """
        Interpolate GPS position at target timestamp between two GPS points.
        """
        # Convert target timestamp to comparable format
        target_ts = pd.Timestamp(target_timestamp)
        
        # Find surrounding GPS points
        time_diffs = self.gps_df['timestamp'] - target_ts
        
        # Find points before and after target time
        before_mask = time_diffs <= pd.Timedelta(0)
        after_mask = time_diffs >= pd.Timedelta(0)
        
        if not before_mask.any():
            # Target is before all GPS points, use first point
            result = self.gps_df.iloc[0].to_dict()
            result['interpolated'] = False
            return result
        
        if not after_mask.any():
            # Target is after all GPS points, use last point
            result = self.gps_df.iloc[-1].to_dict()
            result['interpolated'] = False
            return result
        
        # Get the closest points before and after
        before_df = self.gps_df[before_mask]
        after_df = self.gps_df[after_mask]
        
        # Get the last point before and first point after
        before_idx = before_df.index[-1] if len(before_df) > 0 else 0
        after_idx = after_df.index[0] if len(after_df) > 0 else len(self.gps_df) - 1
        
        point_before = self.gps_df.iloc[before_idx]
        point_after = self.gps_df.iloc[after_idx]
        
        # If it's the same point or very close, return exact match
        if before_idx == after_idx or abs((point_after['timestamp'] - point_before['timestamp']).total_seconds()) < 0.001:
            result = point_before.to_dict()
            result['interpolated'] = False
            return result
        
        # Calculate interpolation factor
        time_diff_total = (point_after['timestamp'] - point_before['timestamp']).total_seconds()
        time_diff_target = (target_ts - point_before['timestamp']).total_seconds()
        
        if time_diff_total == 0:
            factor = 0
        else:
            factor = max(0, min(1, time_diff_target / time_diff_total))  # Clamp between 0 and 1
        
        # Interpolate coordinates with higher precision
        lat = float(point_before['lat']) + factor * (float(point_after['lat']) - float(point_before['lat']))
        lon = float(point_before['lon']) + factor * (float(point_after['lon']) - float(point_before['lon']))
        
        # Interpolate other values
        distance = float(point_before['distance']) + factor * (float(point_after['distance']) - float(point_before['distance']))
        cumulative_distance = float(point_before['cumulative_distance']) + factor * (float(point_after['cumulative_distance']) - float(point_before['cumulative_distance']))
        
        # Interpolate bearing (handle circular nature of angles)
        bearing_before = float(point_before['bearing'])
        bearing_after = float(point_after['bearing'])
        
        # Handle bearing wraparound (e.g., 359° to 1°)
        bearing_diff = bearing_after - bearing_before
        if bearing_diff > 180:
            bearing_diff -= 360
        elif bearing_diff < -180:
            bearing_diff += 360
        
        bearing = bearing_before + factor * bearing_diff
        if bearing < 0:
            bearing += 360
        elif bearing >= 360:
            bearing -= 360
        
        return {
            'timestamp': target_timestamp,
            'lat': lat,
            'lon': lon,
            'distance': distance,
            'bearing': bearing,
            'cumulative_distance': cumulative_distance,
            'interpolated': True,
            'interpolation_factor': factor,
            'before_idx': before_idx,
            'after_idx': after_idx
        }

    def adjust_coordinates_by_position(self, lat: float, lon: float, bearing: float, position: str) -> Tuple[float, float]:
        """
        Adjust GPS coordinates based on sign position relative to the road.
        
        Args:
            lat: Original latitude
            lon: Original longitude  
            bearing: Direction of travel in degrees
            position: Sign position ('levo'=left, 'desno'=right, 'iznad'=above)
            
        Returns:
            Tuple of adjusted (latitude, longitude)
        """
        # Position-specific distances
        if position.lower() == 'levo':  # Left side
            lateral_distance = 3.0  # meters to the left
            forward_distance = 1.5  # meters forward
        elif position.lower() == 'desno':  # Right side  
            lateral_distance = -4.0  # meters to the right (negative for right)
            forward_distance = 1.5  # meters forward
        elif position.lower() == 'iznad':  # Above (overhead signs)
            lateral_distance = 0.0  # No lateral offset
            forward_distance = 5  # meters forward
        else:
            # Unknown position, no adjustment
            return lat, lon
        
        # Convert bearing to radians (bearing is from North, clockwise)
        bearing_rad = math.radians(bearing)
        
        # Calculate perpendicular direction for lateral movement
        # Left is 90 degrees counterclockwise from bearing, right is 90 degrees clockwise
        lateral_bearing_rad = bearing_rad - math.pi/2  # 90 degrees counterclockwise
        
        # Forward direction (same as bearing)
        forward_bearing_rad = bearing_rad
        
        # Earth radius in meters
        R = 6378137.0
        
        # Calculate coordinate changes
        # Lateral movement (perpendicular to direction of travel)
        if lateral_distance != 0:
            lat_lateral = lat + (lateral_distance * math.cos(lateral_bearing_rad)) / R * (180 / math.pi)
            lon_lateral = lon + (lateral_distance * math.sin(lateral_bearing_rad)) / (R * math.cos(math.radians(lat))) * (180 / math.pi)
        else:
            lat_lateral = lat
            lon_lateral = lon
        
        # Forward movement (along direction of travel)
        if forward_distance != 0:
            lat_adjusted = lat_lateral + (forward_distance * math.cos(forward_bearing_rad)) / R * (180 / math.pi)
            lon_adjusted = lon_lateral + (forward_distance * math.sin(forward_bearing_rad)) / (R * math.cos(math.radians(lat_lateral))) * (180 / math.pi)
        else:
            lat_adjusted = lat_lateral
            lon_adjusted = lon_lateral
        
        return lat_adjusted, lon_adjusted
    
    def map_signs_to_gps(self) -> List[Dict]:
        """Map traffic signs to GPS coordinates."""
        if not self.signs_data or self.gps_df.empty:
            print("No sign or GPS data loaded")
            return []
        
        # Use the first GPS timestamp as video start time
        video_start_time = self.gps_df['timestamp'].min()
        print(f"Using video start time: {video_start_time}")
        
        mapped_signs = []
        
        for sign in self.signs_data:
            # Calculate timestamps for sign detection
            first_seen_time = self.frame_to_timestamp(sign['first_seen'], video_start_time)
            last_seen_time = self.frame_to_timestamp(sign['last_seen'], video_start_time)
            mid_time = first_seen_time + (last_seen_time - first_seen_time) / 2
            
            # Get GPS position at middle of detection
            gps_position = self.interpolate_gps_position(mid_time)

            adjusted_lat, adjusted_lon = self.adjust_coordinates_by_position(
                gps_position['lat'], 
                gps_position['lon'], 
                gps_position['bearing'], 
                sign['position']
            )
            
            # Create mapped sign entry
            mapped_sign = {
                'sign_id': sign['id'],
                'sign_class': sign['class'],
                'position': sign['position'],
                'lat_original': gps_position['lat'],
                'lon_original': gps_position['lon'],
                'lat': adjusted_lat,
                'lon': adjusted_lon,
                'bearing': gps_position['bearing'],
                'distance_from_start': gps_position['cumulative_distance'],
                'mid_frame': (sign['first_seen'] + sign['last_seen']) // 2,
            }
            
            mapped_signs.append(mapped_sign)
        
        self.mapped_signs = mapped_signs
        return mapped_signs