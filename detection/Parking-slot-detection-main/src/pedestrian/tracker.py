from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.pedestrian.detector import PedestrianDetection


@dataclass
class PedestrianTrack:
    """Tracked pedestrian with history."""
    track_id: int
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    confidence: float
    age: int = 1
    missed_frames: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    speeds: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def __post_init__(self):
        self.history.append(self.center)
    
    @property
    def avg_speed(self) -> float:
        """Average speed over history (pixels per frame)."""
        if len(self.speeds) == 0:
            return 0.0
        return sum(self.speeds) / len(self.speeds)
    
    @property
    def is_moving(self) -> bool:
        """Check if pedestrian is moving."""
        return self.avg_speed > 3.0
    
    @property
    def is_stopped(self) -> bool:
        """Check if pedestrian is stopped."""
        return self.avg_speed < 1.0 and self.age > 5


class PedestrianTracker:
    """
    Track pedestrians across frames using IoU-based matching.
    
    Features:
    - Kalman filter for smooth trajectories (optional)
    - Speed calculation
    - Stop detection
    - Trajectory history
    """
    
    def __init__(self, config: dict):
        self.max_missed_frames = int(config.get("max_missed_frames", 15))
        self.iou_threshold = float(config.get("iou_threshold", 0.3))
        self.use_kalman = bool(config.get("use_kalman", False))
        self.next_track_id = 1
        self.tracks: dict[int, PedestrianTrack] = {}
        
        # For Kalman filter (optional)
        self.kalman_filters: dict[int, Any] = {}
    
    def update(self, detections: list[PedestrianDetection]) -> list[PedestrianTrack]:
        """Update tracks with new detections."""
        updated_tracks: dict[int, PedestrianTrack] = {}
        unmatched_detection_indices = set(range(len(detections)))
        
        # Match existing tracks to detections
        matches = self._match_tracks_to_detections(detections)
        
        for track_id, detection_idx in matches:
            detection = detections[detection_idx]
            track = self.tracks[track_id]
            
            # Update track with new detection
            center = self._bbox_center(detection.bbox)
            speed = self._calculate_speed(track.center, center)
            
            track.bbox = detection.bbox
            track.center = center
            track.confidence = detection.confidence
            track.age += 1
            track.missed_frames = 0
            track.speeds.append(speed)
            track.history.append(center)
            
            # Apply Kalman filter if enabled
            if self.use_kalman:
                center = self._kalman_update(track_id, center)
            
            updated_tracks[track_id] = track
            unmatched_detection_indices.discard(detection_idx)
        
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detection_indices:
            detection = detections[detection_idx]
            center = self._bbox_center(detection.bbox)
            
            track = PedestrianTrack(
                track_id=self.next_track_id,
                bbox=detection.bbox,
                center=center,
                confidence=detection.confidence,
            )
            updated_tracks[self.next_track_id] = track
            
            if self.use_kalman:
                self._kalman_init(self.next_track_id, center)
            
            self.next_track_id += 1
        
        # Handle missing tracks (not matched)
        for track_id, track in self.tracks.items():
            if track_id not in updated_tracks:
                if track.missed_frames < self.max_missed_frames:
                    track.missed_frames += 1
                    updated_tracks[track_id] = track
                # else: track expires
        
        self.tracks = updated_tracks
        return list(self.tracks.values())
    
    def _match_tracks_to_detections(self, detections: list[PedestrianDetection]) -> list[tuple[int, int]]:
        """Match tracks to detections using IoU."""
        if not self.tracks or not detections:
            return []
        
        candidates = []
        for track_id, track in self.tracks.items():
            for detection_idx, detection in enumerate(detections):
                iou = self._bbox_iou(track.bbox, detection.bbox)
                if iou >= self.iou_threshold:
                    candidates.append((iou, track_id, detection_idx))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        matched_tracks = set()
        matched_detections = set()
        matches = []
        
        for iou, track_id, detection_idx in candidates:
            if track_id in matched_tracks or detection_idx in matched_detections:
                continue
            matched_tracks.add(track_id)
            matched_detections.add(detection_idx)
            matches.append((track_id, detection_idx))
        
        return matches
    
    @staticmethod
    def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        """Calculate bounding box center."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    
    @staticmethod
    def _bbox_iou(bbox1: tuple[float, float, float, float], 
                  bbox2: tuple[float, float, float, float]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def _calculate_speed(prev_center: tuple[float, float], 
                         curr_center: tuple[float, float]) -> float:
        """Calculate speed in pixels per frame."""
        dx = prev_center[0] - curr_center[0]
        dy = prev_center[1] - curr_center[1]
        return math.hypot(dx, dy)
    
    def _kalman_init(self, track_id: int, center: tuple[float, float]) -> None:
        """Initialize Kalman filter for track."""
        try:
            import cv2
            kalman = cv2.KalmanFilter(4, 2)
            kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
            kalman.transitionMatrix = np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], np.float32)
            kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
            state = np.array([[center[0]], [center[1]], [0], [0]], np.float32)
            kalman.statePre = state
            self.kalman_filters[track_id] = kalman
        except ImportError:
            self.use_kalman = False
    
    def _kalman_update(self, track_id: int, center: tuple[float, float]) -> tuple[float, float]:
        """Update Kalman filter and return predicted position."""
        if track_id not in self.kalman_filters:
            return center
        
        kalman = self.kalman_filters[track_id]
        measurement = np.array([[center[0]], [center[1]]], np.float32)
        kalman.correct(measurement)
        prediction = kalman.predict()
        return (float(prediction[0]), float(prediction[1]))