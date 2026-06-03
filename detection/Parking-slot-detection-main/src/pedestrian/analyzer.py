from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.detection.schemas import ParkingSlot
from src.geometry.point_in_polygon import point_in_polygon
from src.pedestrian.tracker import PedestrianTrack


@dataclass
class BehaviorEvent:
    """Record of a pedestrian behavior event."""
    timestamp: int
    track_id: int
    event_type: str  # "entered_slot", "exited_slot", "stopped", "started", "interaction"
    details: dict[str, Any] = field(default_factory=dict)


class PedestrianBehaviorAnalyzer:
    """
    Analyze pedestrian behavior patterns.
    
    Capabilities:
    - Detect when pedestrians enter/exit parking slots
    - Calculate walking speeds and trajectories
    - Detect stopped pedestrians (potential vehicle entry/exit)
    - Detect group walking
    - Calculate proximity to vehicles
    - Predict future paths
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.slots: list[ParkingSlot] = []
        self.slot_occupancy: dict[int, list[int]] = defaultdict(list)  # slot_id -> [track_ids]
        self.events: list[BehaviorEvent] = []
        self.frame_rate = float(config.get("frame_rate", 30.0))
        self.stop_speed_threshold = float(config.get("stop_speed_threshold", 2.0))  # pixels/frame
        self.proximity_threshold = float(config.get("proximity_threshold", 50.0))  # pixels
        self.gaze_threshold = float(config.get("gaze_threshold", 45.0))  # degrees
    
    def set_parking_slots(self, slots: list[ParkingSlot]) -> None:
        """Set parking slots for analysis."""
        self.slots = slots
        self.slot_occupancy.clear()
    
    def analyze(self, tracks: list[PedestrianTrack], frame_idx: int, 
                vehicle_centers: list[tuple[float, float]] | None = None) -> dict[str, Any]:
        results = {
            "slot_crossings": [],
            "stopped_pedestrians": [],
            "groups": [],
            "proximity_alerts": [],
            "behavior_events": [],
        }
        
        # Analyze each track
        for track in tracks:
            # Check parking slot crossings
            slot_id = self._check_slot_crossing(track, frame_idx)
            if slot_id is not None:
                results["slot_crossings"].append({
                    "track_id": track.track_id,
                    "slot_id": slot_id,
                    "center": track.center,
                })
            
            # Check if stopped
            if track.is_stopped:
                results["stopped_pedestrians"].append({
                    "track_id": track.track_id,
                    "center": track.center,
                    "stop_duration": track.missed_frames / self.frame_rate,
                })
                self._record_event(frame_idx, track.track_id, "stopped", {
                    "duration": track.missed_frames / self.frame_rate
                })
            
            # Check proximity to vehicles
            if vehicle_centers:
                alerts = self._check_vehicle_proximity(track, vehicle_centers)
                results["proximity_alerts"].extend(alerts)
        
        # Detect groups
        results["groups"] = self._detect_groups(tracks)
        
        # Generate behavior events
        results["behavior_events"] = self._get_recent_events(frame_idx)
        
        return results
    
    def _check_slot_crossing(self, track: PedestrianTrack, frame_idx: int) -> int | None:
        """Check if pedestrian is inside any parking slot."""
        # Use foot position (bottom center of bbox)
        foot_x = (track.bbox[0] + track.bbox[2]) / 2
        foot_y = track.bbox[3]
        foot_point = (foot_x, foot_y)
        
        for slot in self.slots:
            if point_in_polygon(foot_point, slot.points):
                # Record entry if this is new
                if track.track_id not in self.slot_occupancy[slot.slot_id]:
                    self._record_event(frame_idx, track.track_id, "entered_slot", {
                        "slot_id": slot.slot_id
                    })
                    self.slot_occupancy[slot.slot_id].append(track.track_id)
                return slot.slot_id
        
        # Check if leaving a slot
        for slot_id, track_ids in self.slot_occupancy.items():
            if track.track_id in track_ids:
                track_ids.remove(track.track_id)
                self._record_event(frame_idx, track.track_id, "exited_slot", {
                    "slot_id": slot_id
                })
        
        return None
    
    def _check_vehicle_proximity(self, track: PedestrianTrack, 
                                  vehicle_centers: list[tuple[float, float]]) -> list[dict]:
        """Check if pedestrian is too close to vehicles."""
        alerts = []
        for v_center in vehicle_centers:
            distance = np.linalg.norm(np.array(track.center) - np.array(v_center))
            if distance < self.proximity_threshold:
                risk_level = "high" if distance < 25 else "medium"
                alerts.append({
                    "track_id": track.track_id,
                    "vehicle_center": v_center,
                    "distance": float(distance),
                    "risk_level": risk_level,
                })
                
                if risk_level == "high":
                    self._record_event(track.track_id, None, "high_risk_proximity", {
                        "distance": float(distance)
                    })
        return alerts
    
    def _detect_groups(self, tracks: list[PedestrianTrack]) -> list[dict]:
        """Detect groups of pedestrians walking together."""
        if len(tracks) < 2:
            return []
        
        groups = []
        used = set()
        
        for i, track_i in enumerate(tracks):
            if i in used:
                continue
            
            group = [track_i]
            for j, track_j in enumerate(tracks[i+1:], i+1):
                if j in used:
                    continue
                
                distance = np.linalg.norm(np.array(track_i.center) - np.array(track_j.center))
                if distance < self.proximity_threshold * 1.5:
                    # Check if moving in similar direction
                    angle_diff = self._angle_between_directions(track_i, track_j)
                    if angle_diff < self.gaze_threshold:
                        group.append(track_j)
                        used.add(j)
            
            if len(group) >= 2:
                groups.append({
                    "size": len(group),
                    "track_ids": [t.track_id for t in group],
                    "center": self._group_center([t.center for t in group]),
                    "avg_speed": np.mean([t.avg_speed for t in group]),
                })
                used.add(i)
        
        return groups
    
    def predict_path(self, track: PedestrianTrack, future_frames: int = 30) -> list[tuple[float, float]]:
        """
        Predict future path using linear extrapolation.
        
        Args:
            track: Pedestrian track
            future_frames: Number of frames to predict
            
        Returns:
            List of predicted (x, y) positions
        """
        if len(track.history) < 3:
            return []
        
        # Calculate velocity from recent history
        recent = list(track.history)[-5:]
        velocities = []
        for i in range(1, len(recent)):
            velocities.append((
                recent[i][0] - recent[i-1][0],
                recent[i][1] - recent[i-1][1]
            ))
        
        avg_velocity = np.mean(velocities, axis=0) if velocities else (0, 0)
        
        # Extrapolate
        predictions = []
        current = np.array(track.center)
        for _ in range(future_frames):
            current = current + avg_velocity
            predictions.append(tuple(current))
        
        return predictions
    
    def calculate_trajectory_efficiency(self, track: PedestrianTrack) -> float:
        """
        Calculate how efficient the pedestrian's path is.
        
        Efficiency = direct_distance / actual_path_length
        Higher = more direct path (less wandering)
        """
        if len(track.history) < 2:
            return 1.0
        
        start = track.history[0]
        end = track.history[-1]
        direct_distance = np.linalg.norm(np.array(end) - np.array(start))
        
        actual_length = 0
        positions = list(track.history)
        for i in range(1, len(positions)):
            actual_length += np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1]))
        
        return direct_distance / max(actual_length, 1e-6)
    
    def _record_event(self, frame_idx: int, track_id: int, event_type: str, details: dict) -> None:
        """Record behavior event."""
        self.events.append(BehaviorEvent(
            timestamp=frame_idx,
            track_id=track_id,
            event_type=event_type,
            details=details,
        ))
        
        # Keep only recent events
        if len(self.events) > 1000:
            self.events = self.events[-500:]
    
    def _get_recent_events(self, frame_idx: int, lookback: int = 30) -> list[dict]:
        """Get events from recent frames."""
        recent = []
        for event in self.events[-lookback:]:
            if frame_idx - event.timestamp <= lookback:
                recent.append({
                    "timestamp": event.timestamp,
                    "track_id": event.track_id,
                    "event_type": event.event_type,
                    "details": event.details,
                })
        return recent
    
    @staticmethod
    def _angle_between_directions(track_a: PedestrianTrack, track_b: PedestrianTrack) -> float:
        """Calculate angle difference between two tracks' directions."""
        if len(track_a.history) < 2 or len(track_b.history) < 2:
            return 0.0
        
        dir_a = np.array(track_a.center) - np.array(track_a.history[-2])
        dir_b = np.array(track_b.center) - np.array(track_b.history[-2])
        
        if np.linalg.norm(dir_a) == 0 or np.linalg.norm(dir_b) == 0:
            return 0.0
        
        cos_angle = np.dot(dir_a, dir_b) / (np.linalg.norm(dir_a) * np.linalg.norm(dir_b))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    @staticmethod
    def _group_center(centers: list[tuple[float, float]]) -> tuple[float, float]:
        """Calculate center of group."""
        return (np.mean([c[0] for c in centers]), np.mean([c[1] for c in centers]))
    
    def get_statistics(self) -> dict[str, Any]:
        """Get aggregated behavior statistics."""
        if not self.events:
            return {}
        
        entered_slots = [e for e in self.events if e.event_type == "entured_slot"]
        stopped = [e for e in self.events if e.event_type == "stopped"]
        high_risk = [e for e in self.events if e.event_type == "high_risk_proximity"]
        
        return {
            "total_events": len(self.events),
            "slot_entries": len(entered_slots),
            "stops_detected": len(stopped),
            "high_risk_alerts": len(high_risk),
            "unique_pedestrians": len(set(e.track_id for e in self.events)),
        }