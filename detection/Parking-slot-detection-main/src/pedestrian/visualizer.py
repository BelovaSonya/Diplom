from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import cv2
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pedestrian import (PedestrianDetector, PedestrianTracker,
                            PedestrianBehaviorAnalyzer, PedestrianMetrics,
                            PedestrianVisualizer)
from src.detection.vehicle_detector import VehicleDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pedestrian behavior analysis pipeline")
    parser.add_argument("--input", required=True, help="Input video file or image directory")
    parser.add_argument("--output-dir", default="outputs/pedestrian_analysis")
    parser.add_argument("--limit", type=int, help="Number of frames to process")
    parser.add_argument("--detector-conf", type=float, default=0.35)
    parser.add_argument("--show-preview", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--frame-rate", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    pedestrian_detector = PedestrianDetector({
        "backend": "yolo",
        "conf_threshold": args.detector_conf,
        "device": "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu",
    })
    
    pedestrian_tracker = PedestrianTracker({
        "max_missed_frames": 15,
        "iou_threshold": 0.3,
        "use_kalman": True,
    })
    
    behavior_analyzer = PedestrianBehaviorAnalyzer({
        "frame_rate": args.frame_rate,
        "stop_speed_threshold": 2.0,
        "proximity_threshold": 50.0,
    })
    
    vehicle_detector = VehicleDetector({
        "backend": "yolo",
        "enabled": True,
        "classes": ["car", "truck", "bus"],
        "conf_threshold": 0.4,
    })
    
    visualizer = PedestrianVisualizer({
        "show_trails": True,
        "show_speed": True,
        "trail_length": 20,
    })
    
    metrics = PedestrianMetrics()
    
    # Process video or image directory
    frames_processed = 0
    all_results = []
    
    # Open video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or args.frame_rate
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    video_writer = None
    if args.save_video:
        video_path = output_dir / "output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    frame_idx = 0
    progress = tqdm(desc="Processing frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if args.limit and frame_idx >= args.limit:
            break
        
        # Run pipeline
        pedestrians = pedestrian_detector.detect(frame)
        pedestrian_tracks = pedestrian_tracker.update(pedestrians)
        
        # Get vehicle centers for proximity analysis
        vehicles = vehicle_detector.detect(frame)
        vehicle_centers = []
        for v in vehicles:
            x1, y1, x2, y2 = v.bbox
            vehicle_centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
        
        # Analyze behavior
        analysis = behavior_analyzer.analyze(pedestrian_tracks, frame_idx, vehicle_centers)
        
        # Update metrics (simplified)
        # In real scenario, you'd need ground truth
        
        # Visualize
        output_frame = visualizer.draw(frame, pedestrian_tracks, analysis)
        
        # Add frame info
        cv2.putText(output_frame, f"Frame: {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output_frame, f"Pedestrians: {len(pedestrian_tracks)}", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Save results
        if args.show_preview:
            cv2.imshow("Pedestrian Analysis", output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if video_writer:
            video_writer.write(output_frame)
        
        # Record results
        all_results.append({
            "frame": frame_idx,
            "pedestrians": len(pedestrian_tracks),
            "vehicles": len(vehicles),
            "slot_crossings": len(analysis.get("slot_crossings", [])),
            "stops": len(analysis.get("stopped_pedestrians", [])),
            "groups": len(analysis.get("groups", [])),
            "alerts": len(analysis.get("proximity_alerts", [])),
        })
        
        frames_processed += 1
        frame_idx += 1
        progress.update(1)
    
    progress.close()
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    # Generate summary
    summary = {
        "input": args.input,
        "frames_processed": frames_processed,
        "total_pedestrians_detected": sum(r["pedestrians"] for r in all_results),
        "total_slot_crossings": sum(r["slot_crossings"] for r in all_results),
        "total_stops": sum(r["stops"] for r in all_results),
        "total_groups": sum(r["groups"] for r in all_results),
        "total_proximity_alerts": sum(r["alerts"] for r in all_results),
        "behavior_statistics": behavior_analyzer.get_statistics(),
    }
    
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    
    print(json.dumps(summary, indent=2))
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()