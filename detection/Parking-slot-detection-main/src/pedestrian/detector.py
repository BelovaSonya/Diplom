from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import cv2
import numpy as np


@dataclass
class PedestrianDetection:
    """Single pedestrian detection result."""
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_name: str = "person"
    keypoints: list[tuple[float, float, float]] | None = None  # x, y, confidence


class PedestrianDetector:
       def __init__(self, config: dict):
        self.config = config
        self.backend = str(config.get("backend", "yolo")).lower()
        self.model_path = str(config.get("model_path", "yolov8n.pt"))
        self.device = str(config.get("device", "cpu"))
        self.conf_threshold = float(config.get("conf_threshold", 0.35))
        self.iou_threshold = float(config.get("iou_threshold", 0.45))
        self.imgsz = int(config.get("imgsz", 640))
        self.enable_keypoints = bool(config.get("enable_keypoints", False))
        self.model = None
        
        if self.backend not in {"yolo", "mock"}:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def detect(self, frame: np.ndarray) -> list[PedestrianDetection]:
        """Detect pedestrians in frame."""
        if self.backend == "yolo":
            return self._detect_yolo(frame)
        return self._detect_mock(frame)
    
    def _detect_yolo(self, frame: np.ndarray) -> list[PedestrianDetection]:
        """Detect using YOLO model."""
        self._load_model()
        assert self.model is not None
        
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        
        detections = []
        if not results:
            return detections
        
        result = results[0]
        
        # Filter for person class (class_id == 0 in COCO)
        if result.boxes is not None:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                if int(cls.item()) == 0:  # person class
                    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                    detections.append(PedestrianDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(conf.item()),
                        class_name="person"
                    ))
        
        # Extract keypoints if enabled
        if self.enable_keypoints and result.keypoints is not None:
            for i, detection in enumerate(detections):
                if i < len(result.keypoints):
                    kpts = result.keypoints[i]
                    keypoints = []
                    for j in range(0, len(kpts), 3):
                        x, y, conf = float(kpts[j]), float(kpts[j+1]), float(kpts[j+2])
                        if conf > 0.5:
                            keypoints.append((x, y, conf))
                    detection.keypoints = keypoints
        
        return detections
    
    def _detect_mock(self, frame: np.ndarray) -> list[PedestrianDetection]:
        """Generate synthetic detections for testing."""
        height, width = frame.shape[:2]
        detections = []
        
        # Add a few synthetic pedestrians
        positions = [
            (width * 0.3, height * 0.7),
            (width * 0.5, height * 0.65),
            (width * 0.7, height * 0.72),
        ]
        
        for idx, (cx, cy) in enumerate(positions):
            w, h = width * 0.05, height * 0.12
            x1, y1 = cx - w/2, cy - h
            x2, y2 = cx + w/2, cy
            detections.append(PedestrianDetection(
                bbox=(x1, y1, x2, y2),
                confidence=0.85 - idx * 0.1,
                class_name="person"
            ))
        
        return detections
    
    def _load_model(self) -> None:
        """Load YOLO model lazily."""
        if self.model is not None:
            return
        
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "PedestrianDetector requires ultralytics. "
                "Install with: pip install ultralytics"
            ) from exc
        
        if not Path(self.model_path).exists():
            # Try default model
            self.model_path = "yolov8n.pt"
        
        self.model = YOLO(self.model_path)