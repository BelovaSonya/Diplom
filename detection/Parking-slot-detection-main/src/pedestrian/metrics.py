from __future__ import annotations
from collections import Counter
from typing import Any
import numpy as np


class PedestrianMetrics:
    def __init__(self):
        self.detection_stats = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }
        self.tracking_stats = {
            "id_switches": 0,
            "fragments": 0,
            "total_tracks": 0,
        }
        self.behavior_stats = Counter()
    
    def update_detection(self, gt_pedestrians: list, pred_pedestrians: list, iou_threshold: float = 0.5):
        """
        Update detection metrics with frame results.
        
        Args:
            gt_pedestrians: List of ground truth bboxes
            pred_pedestrians: List of predicted bboxes
            iou_threshold: IoU threshold for match
        """
        matches = self._match_detections(gt_pedestrians, pred_pedestrians, iou_threshold)
        
        self.detection_stats["true_positives"] += len(matches)
        self.detection_stats["false_positives"] += len(pred_pedestrians) - len(matches)
        self.detection_stats["false_negatives"] += len(gt_pedestrians) - len(matches)
    
    def update_tracking(self, gt_tracks: dict, pred_tracks: dict):
        """
        Update tracking metrics.
        
        Args:
            gt_tracks: Ground truth track ID -> bbox sequence
            pred_tracks: Predicted track ID -> bbox sequence
        """
        # Count ID switches (simplified)
        matches = self._match_tracks(gt_tracks, pred_tracks)
        self.tracking_stats["id_switches"] += len(matches) - len(set(matches.values()))
        self.tracking_stats["total_tracks"] += len(pred_tracks)
    
    def update_behavior(self, behavior_results: dict):
        """Update behavior statistics."""
        self.behavior_stats.update({
            "slot_crossings": len(behavior_results.get("slot_crossings", [])),
            "stops": len(behavior_results.get("stopped_pedestrians", [])),
            "groups": len(behavior_results.get("groups", [])),
            "proximity_alerts": len(behavior_results.get("proximity_alerts", [])),
        })
    
    def get_detection_metrics(self) -> dict[str, float]:
        """Calculate precision, recall, F1 for detection."""
        tp = self.detection_stats["true_positives"]
        fp = self.detection_stats["false_positives"]
        fn = self.detection_stats["false_negatives"]
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }
    
    def get_tracking_metrics(self) -> dict[str, float]:
        """Calculate MOTA-like metrics."""
        tp = self.detection_stats["true_positives"]
        fp = self.detection_stats["false_positives"]
        fn = self.detection_stats["false_negatives"]
        id_switches = self.tracking_stats["id_switches"]
        
        total_gt = tp + fn
        mota = 1.0 - (fp + fn + id_switches) / max(total_gt, 1)
        
        return {
            "mota": max(0, mota),
            "id_switches": id_switches,
            "total_tracks": self.tracking_stats["total_tracks"],
        }
    
    def get_behavior_metrics(self) -> dict[str, Any]:
        """Get aggregated behavior statistics."""
        total = sum(self.behavior_stats.values()) or 1
        
        return {
            "slot_crossings": self.behavior_stats.get("slot_crossings", 0),
            "stops_detected": self.behavior_stats.get("stops", 0),
            "groups_detected": self.behavior_stats.get("groups", 0),
            "proximity_alerts": self.behavior_stats.get("proximity_alerts", 0),
            "crossing_per_frame": self.behavior_stats.get("slot_crossings", 0) / max(total, 1),
        }
    
    def get_all_metrics(self) -> dict[str, dict]:
        """Get all metrics as a nested dictionary."""
        return {
            "detection": self.get_detection_metrics(),
            "tracking": self.get_tracking_metrics(),
            "behavior": self.get_behavior_metrics(),
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.detection_stats = {"true_positives": 0, "false_positives": 0, "false_negatives": 0}
        self.tracking_stats = {"id_switches": 0, "fragments": 0, "total_tracks": 0}
        self.behavior_stats.clear()
    
    @staticmethod
    def _match_detections(gt: list, pred: list, iou_threshold: float) -> list:
        """Match ground truth to predictions using IoU."""
        from src.pedestrian.tracker import PedestrianTracker
        
        matches = []
        used_gt = set()
        used_pred = set()
        
        candidates = []
        for i, gt_box in enumerate(gt):
            for j, pred_box in enumerate(pred):
                iou = PedestrianTracker._bbox_iou(gt_box, pred_box)
                if iou >= iou_threshold:
                    candidates.append((iou, i, j))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        for iou, i, j in candidates:
            if i in used_gt or j in used_pred:
                continue
            used_gt.add(i)
            used_pred.add(j)
            matches.append((i, j))
        
        return matches
    
    @staticmethod
    def _match_tracks(gt_tracks: dict, pred_tracks: dict) -> dict:
        """Simple track matching based on overlap."""
        matches = {}
        used_pred = set()
        
        for gt_id, gt_seq in gt_tracks.items():
            best_match = None
            best_overlap = 0
            
            for pred_id, pred_seq in pred_tracks.items():
                if pred_id in used_pred:
                    continue
                
                overlap = PedestrianMetrics._sequence_overlap(gt_seq, pred_seq)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = pred_id
            
            if best_match is not None and best_overlap > 0.3:
                matches[gt_id] = best_match
                used_pred.add(best_match)
        
        return matches
    
    @staticmethod
    def _sequence_overlap(seq1: list, seq2: list) -> float:
        """Calculate overlap ratio between two sequences."""
        if not seq1 or not seq2:
            return 0.0
        
        # Simplified: just count frame overlap
        frames1 = {s[0] for s in seq1} if isinstance(seq1[0], tuple) else set(range(len(seq1)))
        frames2 = {s[0] for s in seq2} if isinstance(seq2[0], tuple) else set(range(len(seq2)))
        
        intersection = len(frames1 & frames2)
        union = len(frames1 | frames2)
        
        return intersection / max(union, 1)