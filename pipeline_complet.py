import os
import pandas as pd
import numpy as np
from natsort import natsorted
import cv2
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from track_test import CustomParticleTracker

# SAHI imports
from sahi.predict import predict
from sahi import AutoDetectionModel

# Ultralytics imports
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CombinedDetectionTrackingPipeline:
    """
    Pipeline combiné pour détection YOLO et tracking de particules
    """

    def __init__(self,
                 model_path: str = "best_yolo8.pt",
                 tracker_model_path: str ='best_particle_embedding_model.pth',
                 confidence_threshold: float = 0.4,
                 device: str = "cpu",
                 use_sahi: bool = True,
                 slice_height: int = 750,
                 slice_width: int = 750,
                 overlap_ratio: float = 0.2):
        """
        Initialize le pipeline

        Args:
            model_path: Chemin vers le modèle YOLO
            tracker_model_path: Chemin vers le modèle de tracking
            confidence_threshold: Seuil de confiance
            device: Device ("cpu" ou "cuda")
            use_sahi: Utiliser SAHI ou YOLO direct
            slice_height/width: Tailles des slices pour SAHI
            overlap_ratio: Ratio d'overlap pour SAHI
        """
        self.model_path = model_path
        self.tracker_model_path = tracker_model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.use_sahi = use_sahi
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_ratio = overlap_ratio

        # Initialize YOLO model if not using SAHI
        if not use_sahi:
            logger.info(f"Loading YOLO model from: {model_path}")
            self.yolo_model = YOLO(model_path)
            logger.info(f"YOLO model loaded successfully")

    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Preprocess image for YOLO (convert grayscale to RGB if needed)
        """
        try:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return None

            # Convert grayscale to RGB if needed
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3:
                if image.shape[2] == 1:  # Single channel
                    image = np.squeeze(image, axis=2)
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 3:  # BGR
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            return image

        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            return None

    def run_sahi_detection(self, source_dir: str, output_dir: str) -> pd.DataFrame:
        """
        Run SAHI detection
        """
        logger.info("Running SAHI detection...")

        result = predict(
            model_type="ultralytics",
            model_path=self.model_path,
            model_device=self.device,
            model_confidence_threshold=self.confidence_threshold,
            source=source_dir,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
            export_pickle=True,
            visual_hide_labels=True,
            visual_hide_conf=True,
            return_dict=True,
            project=output_dir
        )

        # Process pickle files
        pickle_dir = Path(result["export_dir"]) / "pickles"
        pickle_files = list(pickle_dir.glob("**/*.pickle"))

        rows = []
        for pkl_file in pickle_files:
            try:
                import pickle
                with open(pkl_file, "rb") as f:
                    preds = pickle.load(f)

                img_name = pkl_file.stem
                for op in preds:
                    rows.append({
                        "image": img_name,
                        "label": op.category.name,
                        "score": op.score.value,
                        "x1": op.bbox.minx,
                        "y1": op.bbox.miny,
                        "x2": op.bbox.maxx,
                        "y2": op.bbox.maxy,
                        "width": op.bbox.maxx - op.bbox.minx,
                        "height": op.bbox.maxy - op.bbox.miny,
                    })
            except Exception as e:
                logger.error(f"Error processing {pkl_file}: {str(e)}")

        df = pd.DataFrame(rows)
        return df

    def run_yolo_detection(self, source_dir: str) -> pd.DataFrame:
        """
        Run direct YOLO detection
        """
        logger.info("Running direct YOLO detection...")

        source_path = Path(source_dir)
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']

        # Get all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(source_path.glob(f"*{ext}"))
            image_files.extend(source_path.glob(f"*{ext.upper()}"))

        image_files = natsorted(image_files)

        all_detections = []
        for img_file in image_files:
            image = self.preprocess_image(str(img_file))
            if image is None:
                continue

            # Run inference
            results = self.yolo_model(
                image,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False
            )

            # Process results
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()

                    for box, conf, cls in zip(boxes, confidences, classes):
                        x1, y1, x2, y2 = box
                        class_name = self.yolo_model.names[int(cls)] if hasattr(self.yolo_model, 'names') else str(
                            int(cls))

                        all_detections.append({
                            "image": img_file.stem,
                            "label": class_name,
                            "score": float(conf),
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                            "width": float(x2 - x1),
                            "height": float(y2 - y1),
                        })

        return pd.DataFrame(all_detections)

    def load_frames(self, frames_dir: str):
        """Load and normalize frames"""
        img_names = natsorted([
            os.path.splitext(f)[0]
            for f in os.listdir(frames_dir)
            if f.lower().endswith((".tiff", ".tif", ".png", ".jpeg", ".jpg"))
        ])

        frames = []
        for name in img_names:
            actual_file = None
            for ext in ['.tiff', '.tif', '.png', '.jpeg', '.jpg']:
                if os.path.exists(os.path.join(frames_dir, name + ext)):
                    actual_file = name + ext
                    break

            if actual_file is None:
                logger.warning(f"Could not find file for {name}")
                continue

            img = cv2.imread(os.path.join(frames_dir, actual_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
                frames.append(img)

        return img_names, frames

    def save_detection_frames(self, frames_dir: str, detections_df: pd.DataFrame, output_dir: str):
        """Save all frames with detections drawn"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Load frames
        img_names, frames = self.load_frames(frames_dir)

        # Create detections dict
        detections_dict = {}
        for img_name, group in detections_df.groupby('image'):
            detections_dict[img_name] = group

        for frame, img_name in zip(frames, img_names):
            # Convert to BGR for annotation
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Draw detections if any
            if img_name in detections_dict:
                detections = detections_dict[img_name]
                for _, det in detections.iterrows():
                    x1, y1 = int(det['x1']), int(det['y1'])
                    x2, y2 = int(det['x2']), int(det['y2'])

                    # Draw bounding box
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    # Draw score
                    cv2.putText(frame_bgr, f"{det['score']:.2f}",
                                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Save frame
            output_file = output_path / f"{img_name}.jpg"
            cv2.imwrite(str(output_file), frame_bgr)

        logger.info(f"Detection frames saved to: {output_path}")

    def run_tracking(self, frames_dir: str, detections_df: pd.DataFrame,
                     output_dir: str) -> Dict[str, Any]:
        """
        Run tracking on frames using detections
        """
        logger.info("Starting tracking...")

        # Create output directories
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        frames_output = output_path / "tracking_frames"
        frames_output.mkdir(exist_ok=True)

        # Load frames
        img_names, frames = self.load_frames(frames_dir)

        # Prepare detections dict
        detections_dict = {}
        for img_name, group in detections_df.groupby('image'):
            detections = group[['x1', 'y1', 'width', 'height', 'score']].values
            detections_dict[img_name] = detections.astype(np.float32)

        # Initialize tracker and video writer
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = output_path / "tracking_video.mp4"
        out = cv2.VideoWriter(str(video_path), fourcc, 2.0, (width, height))

        tracker = CustomParticleTracker(self.tracker_model_path)
        track_paths = defaultdict(list)
        trajectory_data = []

        for frame_idx, (frame, img_name) in enumerate(zip(frames, img_names)):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Get detections for this frame
            detections = detections_dict.get(img_name, np.zeros((0, 5), dtype=np.float32))

            if len(detections) > 0:
                # Convert to tracker format
                valid_detections = []
                for det in detections:
                    if len(det) == 5:
                        bbox = [float(det[0]), float(det[1]), float(det[2]), float(det[3])]
                        confidence = float(det[4])
                        valid_detections.append([bbox, confidence])

                if valid_detections:
                    try:
                        # Update tracker
                        tracks = tracker.update_tracks(valid_detections, frame=frame_bgr)

                        # Draw tracking results
                        for track in tracks:
                            if not track.is_confirmed():
                                continue

                            bbox = track.to_tlbr()
                            track_id = track.track_id
                            x1, y1, x2, y2 = map(int, bbox)

                            # Draw circle at detection point
                            cv2.circle(frame_bgr, (x1, y1), radius=4,
                                       color=(0, 255, 0), thickness=1)

                            # Store trajectory data
                            trajectory_data.append({
                                'frame': frame_idx,
                                'id': track_id,
                                'x1': x1,
                                'y1': y1
                            })

                            # Store for trajectory drawing
                            track_paths[track_id].append((x1, y1))

                    except Exception as e:
                        logger.error(f"Frame {frame_idx}: Error in tracking - {str(e)}")

            # Draw trajectory paths
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            for tid, pts in track_paths.items():
                if len(pts) < 2:
                    continue
                color_idx = hash(str(tid)) % len(colors)
                color = colors[color_idx]

                for i in range(1, len(pts)):
                    cv2.line(frame_bgr, pts[i - 1], pts[i], color, 1)

            # Save frame
            frame_file = frames_output / f"frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(frame_file), frame_bgr)

            # Write to video
            out.write(frame_bgr)

        out.release()

        # Save trajectory CSV
        trajectories_df = pd.DataFrame(trajectory_data)
        csv_path = output_path / "trajectories.csv"
        trajectories_df.to_csv(csv_path, index=False)

        logger.info(f"Tracking completed!")
        logger.info(f"Video saved to: {video_path}")
        logger.info(f"Trajectories CSV saved to: {csv_path}")
        logger.info(f"Tracking frames saved to: {frames_output}")

        return {
            "video_path": str(video_path),
            "csv_path": str(csv_path),
            "frames_path": str(frames_output),
            "trajectories_df": trajectories_df
        }

    def run_complete_pipeline(self, source_dir: str, output_base_dir: str = "pipeline_results"):
        """
        Run complete detection + tracking pipeline
        """
        logger.info("=" * 60)
        logger.info("STARTING COMPLETE DETECTION + TRACKING PIPELINE")
        logger.info("=" * 60)

        output_path = Path(output_base_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Step 1: Detection
        logger.info("\n📍 STEP 1: DETECTION")
        if self.use_sahi:
            detections_df = self.run_sahi_detection(source_dir, str(output_path / "sahi_results"))
        else:
            detections_df = self.run_yolo_detection(source_dir)

        # Save detection CSV
        detection_csv = output_path / "detections.csv"
        detections_df.to_csv(detection_csv, index=False)
        logger.info(f"✅ Detection CSV saved: {detection_csv}")

        # Step 2: Save detection frames
        logger.info("\n📍 STEP 2: SAVING DETECTION FRAMES")
        detection_frames_dir = output_path / "detection_frames"
        self.save_detection_frames(source_dir, detections_df, detection_frames_dir)

        # Step 3: Tracking
        logger.info("\n📍 STEP 3: TRACKING")
        tracking_results = self.run_tracking(source_dir, detections_df,
                                             str(output_path / "tracking_results"))

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"📁 Base output directory: {output_path.resolve()}")
        logger.info(f"📄 Detection CSV: {detection_csv}")
        logger.info(f"📁 Detection frames: {detection_frames_dir}")
        logger.info(f"🎥 Tracking video: {tracking_results['video_path']}")
        logger.info(f"📄 Trajectories CSV: {tracking_results['csv_path']}")
        logger.info(f"📁 Tracking frames: {tracking_results['frames_path']}")

        if not detections_df.empty:
            logger.info(f"🔍 Total detections: {len(detections_df)}")
            logger.info(f"🖼️ Images with detections: {detections_df['image'].nunique()}")

        if not tracking_results['trajectories_df'].empty:
            trajectories_df = tracking_results['trajectories_df']
            logger.info(f"🎯 Total trajectory points: {len(trajectories_df)}")
            logger.info(f"🛤️ Unique tracks: {trajectories_df['id'].nunique()}")

        return {
            "detections_df": detections_df,
            "detection_csv": str(detection_csv),
            "detection_frames_dir": str(detection_frames_dir),
            "tracking_results": tracking_results,
            "output_dir": str(output_path)
        }


def main():
    """
    Example usage
    """
    # Configuration
    pipeline = CombinedDetectionTrackingPipeline(
        model_path="best_yolo8.pt",
        tracker_model_path="best_particle_embedding_model.pth",
        confidence_threshold=0.4,
        device="cpu",
        use_sahi=True,  # True pour SAHI, False pour YOLO direct
        slice_height=550,
        slice_width=550,
        overlap_ratio=0.2
    )

    # Run pipeline
    results = pipeline.run_complete_pipeline(
        source_dir="LysoTracker Expt D",
        output_base_dir="complete_pipeline_results_lyso_C"
    )

    return results


if __name__ == "__main__":
    results = main()