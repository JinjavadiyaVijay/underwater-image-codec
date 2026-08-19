"""YOLO detector wrapper for fish/lobster detection.

Uses ultralytics YOLOv8n/v11n to detect subjects in underwater frames.
Returns bounding boxes for cropping before codec encoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Detection:
    """A single detection result."""
    bbox: tuple[int, int, int, int]  # (x, y, w, h) in pixel coordinates
    confidence: float
    class_id: int
    class_name: str


class YOLODetector:
    """YOLO wrapper for underwater fish/lobster detection.

    Wraps ultralytics YOLOv8n or v11n. Falls back to full-frame
    if ultralytics is not installed.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_variant: str = "yolov8n",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        target_classes: list[str] | None = None,
        device: str = "auto",
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes or ["fish", "lobster"]
        self._model = None
        self._model_path = model_path
        self._model_variant = model_variant
        self._device = device

    def _load_model(self):
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO
            if self._model_path:
                self._model = YOLO(str(self._model_path))
            else:
                self._model = YOLO(f"{self._model_variant}.pt")
            print(f"Loaded YOLO model: {self._model_variant}")
        except ImportError:
            print("WARNING: ultralytics not installed. Using full-frame fallback.")
            self._model = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect fish/lobster in a frame.

        Args:
            frame: RGB image (H, W, 3) uint8.

        Returns:
            List of Detection objects, sorted by confidence (descending).
        """
        self._load_model()

        if self._model is None:
            # Fallback: return full frame as a single detection
            h, w = frame.shape[:2]
            return [Detection(
                bbox=(0, 0, w, h),
                confidence=1.0,
                class_id=0,
                class_name="full_frame",
            )]

        results = self._model.predict(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names.get(cls_id, "unknown")
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                detections.append(Detection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detect_and_crop(
        self,
        frame: np.ndarray,
        max_detections: int = 1,
        margin: float = 0.1,
    ) -> list[tuple[np.ndarray, Detection]]:
        """Detect and crop fish/lobster subjects.

        Args:
            frame: RGB image (H, W, 3) uint8.
            max_detections: Maximum number of crops to return.
            margin: Fractional margin to add around bounding box.

        Returns:
            List of (crop, detection) tuples.
        """
        detections = self.detect(frame)[:max_detections]
        h, w = frame.shape[:2]

        results = []
        for det in detections:
            x, y, bw, bh = det.bbox

            # Add margin
            mx = int(bw * margin)
            my = int(bh * margin)
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(w, x + bw + mx)
            y2 = min(h, y + bh + my)

            crop = frame[y1:y2, x1:x2].copy()
            results.append((crop, det))

        return results
