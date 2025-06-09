import gpmf
import math
from geopy.distance import geodesic
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Tuple

def get_all_coords(video_path: str) -> List[Tuple[str, float, float]]:
    """
    Extract all GPS coordinates from a GPMF video file.
    
    Args:
        video_path (str): Path to the video file.
        
    Returns:
        List[Tuple[float, float]]: List of (latitude, longitude) tuples.
    """
    # Read the binary stream from the file
    stream = gpmf.io.extract_gpmf_stream(video_path)

    # Extract GPS low level data from the stream
    gps_blocks = gpmf.gps.extract_gps_blocks(stream)

    # Parse low level data into more usable format
    gps_data = list(map(gpmf.gps.parse_gps_block, gps_blocks))

    all_coords = []

    for block in gps_data:
        latitudes = block.latitude
        longitudes = block.longitude
        timestamp = block.timestamp

        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
        dt_utc2 = dt + timedelta(hours=2)
        
        # Convert back to string if you want to keep the same format
        timestamp_utc2 = dt_utc2.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        for lat, lon in zip(latitudes, longitudes):
            all_coords.append((timestamp_utc2, lat, lon))

    coord_result = []

    for i in range(len(all_coords) - 1):
        p1 = all_coords[i][1:3]  # Extract latitude and longitude
        p2 = all_coords[i + 1][1:3]  # Extract latitude and longitude
        distance = geodesic(p1, p2).meters
        bearing = calculate_bearing(p1, p2)

        coord_result.append({
            "timestamp": all_coords[i][0],
            "lat": all_coords[i][1],
            "lon": all_coords[i][2],
            "distance": distance,
            "bearing": bearing
        })

    return coord_result

def calculate_bearing(p1, p2):
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlon = lon2 - lon1

    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)

    initial_bearing = math.atan2(x, y)
    compass_bearing = (math.degrees(initial_bearing) + 360) % 360
    return compass_bearing