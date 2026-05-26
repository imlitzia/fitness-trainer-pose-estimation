"""
Face tracking with auto-zoom crop and facial fatigue overlays.
Uses Face Mesh, Face Detection, and Pose face landmarks as fallbacks.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from face_estimation.facial_fatigue import FacialFatigueDetector, FACE_OVAL

# MediaPipe Pose landmarks for head region (when face mesh fails)
POSE_FACE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8]


class FaceTracker:
    """Detect face, crop/zoom to face region, run facial fatigue analysis."""

    OUTPUT_SIZE = (480, 480)
    PADDING_RATIO = 0.45
    SMOOTH_ALPHA = 0.3

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.4,
        )
        self.fatigue = FacialFatigueDetector()
        self._bbox_cache: Optional[Tuple[int, int, int, int]] = None
        self._frames_without_face = 0

    def close(self):
        if self.face_mesh:
            self.face_mesh.close()
        if self.face_detection:
            self.face_detection.close()

    def reset(self):
        self.fatigue.reset()
        self._bbox_cache = None
        self._frames_without_face = 0

    def process_frame(
        self,
        frame: np.ndarray,
        pose_landmarks=None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process BGR frame; return zoomed face view and status dict."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        landmarks = self._detect_face_mesh(rgb)
        bbox = None

        if landmarks:
            bbox = self._landmarks_bbox(landmarks, w, h)
        else:
            bbox = self._detect_face_bbox(rgb, w, h, pose_landmarks)

        if bbox is None:
            self._frames_without_face += 1
            self.fatigue.mark_no_face()
            status = self.fatigue.get_status()
            status["tracking"] = False
            return self._no_face_frame(hint_idx=self._frames_without_face), status

        self._frames_without_face = 0
        bbox = self._apply_bbox_smoothing(bbox, w, h)
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            self.fatigue.mark_no_face()
            status = self.fatigue.get_status()
            status["tracking"] = False
            return self._no_face_frame(), status

        crop_h, crop_w = crop.shape[:2]
        zoomed = cv2.resize(crop, self.OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)

        if landmarks is None:
            landmarks = self._detect_face_mesh_on_crop(zoomed)

        status = {"tracking": True, "facial_fatigue_score": 100, "facial_fatigue_level": "fresh"}
        if landmarks:
            status = self.fatigue.update(landmarks, crop_w, crop_h)
            status["tracking"] = True
            self._draw_face_overlay(zoomed, landmarks, crop_w, crop_h, bbox, w, h)
        else:
            cv2.rectangle(zoomed, (8, 8), (zoomed.shape[1] - 8, zoomed.shape[0] - 8), (0, 220, 255), 2)
            status["tracking"] = True
            status["facial_messages"] = ["Face region tracked (refine pose toward camera)"]

        self._draw_fatigue_hud(zoomed, status)
        return zoomed, status

    def _detect_face_mesh(self, rgb: np.ndarray):
        results = self.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0].landmark
        return None

    def _detect_face_mesh_on_crop(self, crop_bgr: np.ndarray):
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return self._detect_face_mesh(rgb)

    def _detect_face_bbox(self, rgb: np.ndarray, w: int, h: int, pose_landmarks) -> Optional[Tuple[int, int, int, int]]:
        det = self.face_detection.process(rgb)
        if det.detections:
            bb = det.detections[0].location_data.relative_bounding_box
            x1 = int(bb.xmin * w)
            y1 = int(bb.ymin * h)
            x2 = int((bb.xmin + bb.width) * w)
            y2 = int((bb.ymin + bb.height) * h)
            return self._pad_bbox(x1, y1, x2, y2, w, h)

        if pose_landmarks:
            return self._bbox_from_pose(pose_landmarks, w, h)
        return None

    def _bbox_from_pose(self, pose_landmarks, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
        try:
            xs, ys = [], []
            for idx in POSE_FACE_INDICES:
                lm = pose_landmarks[idx]
                if lm.visibility is not None and lm.visibility < 0.5:
                    continue
                xs.append(lm.x * w)
                ys.append(lm.y * h)
            if len(xs) < 3:
                return None
            x1, x2 = int(min(xs)), int(max(xs))
            y1, y2 = int(min(ys)), int(max(ys))
            bw, bh = x2 - x1, y2 - y1
            pad_x = int(bw * 0.8)
            pad_y = int(bh * 1.0)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - int(pad_y * 1.2))
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            if x2 - x1 < 60 or y2 - y1 < 60:
                return None
            return (x1, y1, x2, y2)
        except (IndexError, AttributeError):
            return None

    def _pad_bbox(self, x1, y1, x2, y2, w, h) -> Tuple[int, int, int, int]:
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * self.PADDING_RATIO)
        pad_y = int(bh * self.PADDING_RATIO)
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(w, x2 + pad_x),
            min(h, y2 + pad_y),
        )

    def _landmarks_bbox(self, landmarks, fw: int, fh: int) -> Tuple[int, int, int, int]:
        xs = [landmarks[i].x * fw for i in FACE_OVAL]
        ys = [landmarks[i].y * fh for i in FACE_OVAL]
        return self._pad_bbox(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)), fw, fh)

    def _apply_bbox_smoothing(self, bbox: Tuple[int, int, int, int], fw: int, fh: int) -> Tuple[int, int, int, int]:
        if self._bbox_cache is None:
            self._bbox_cache = bbox
            return bbox
        a = self.SMOOTH_ALPHA
        sx1, sy1, sx2, sy2 = self._bbox_cache
        x1, y1, x2, y2 = bbox
        nx1 = int((1 - a) * sx1 + a * x1)
        ny1 = int((1 - a) * sy1 + a * y1)
        nx2 = int((1 - a) * sx2 + a * x2)
        ny2 = int((1 - a) * sy2 + a * y2)
        nx1, ny1 = max(0, nx1), max(0, ny1)
        nx2, ny2 = min(fw, nx2), min(fh, ny2)
        if nx2 - nx1 < 40 or ny2 - ny1 < 40:
            return self._bbox_cache
        self._bbox_cache = (nx1, ny1, nx2, ny2)
        return self._bbox_cache

    def _draw_face_overlay(
        self,
        zoomed: np.ndarray,
        landmarks,
        crop_w: int,
        crop_h: int,
        bbox: Tuple[int, int, int, int],
        full_w: int,
        full_h: int,
    ):
        x1, y1, x2, y2 = bbox
        zh, zw = zoomed.shape[:2]
        pts = []
        for idx in FACE_OVAL:
            lm = landmarks[idx]
            px = int((lm.x * full_w - x1) / max(crop_w, 1) * zw)
            py = int((lm.y * full_h - y1) / max(crop_h, 1) * zh)
            pts.append([px, py])
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(zoomed, [pts], True, (0, 220, 255), 1, cv2.LINE_AA)

    def _draw_fatigue_hud(self, frame: np.ndarray, status: Dict[str, Any]):
        score = status.get("facial_fatigue_score", 100)
        level = status.get("facial_fatigue_level", "fresh")
        color = (0, 200, 0) if score >= 75 else (0, 200, 255) if score >= 50 else (0, 100, 255) if score >= 35 else (0, 0, 220)

        cv2.rectangle(frame, (8, 8), (280, 72), (30, 30, 30), -1)
        cv2.rectangle(frame, (8, 8), (280, 72), color, 2)
        cv2.putText(
            frame, f"FACIAL FATIGUE: {score}%",
            (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
        cv2.putText(
            frame, level.replace("_", " ").title(),
            (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )
        msgs = status.get("facial_messages") or []
        if msgs:
            cv2.putText(
                frame, msgs[0][:42],
                (16, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 255), 1,
            )

    def _no_face_frame(self, hint_idx: int = 0) -> np.ndarray:
        out = np.zeros((self.OUTPUT_SIZE[1], self.OUTPUT_SIZE[0], 3), dtype=np.uint8)
        out[:] = (26, 26, 46)
        cv2.putText(
            out, "Looking for face...",
            (70, self.OUTPUT_SIZE[1] // 2 - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2,
        )
        hints = [
            "Face the camera directly",
            "Move closer or improve lighting",
            "Using body pose to find head...",
        ]
        hint = hints[min(hint_idx, len(hints) - 1)]
        cv2.putText(
            out, hint,
            (40, self.OUTPUT_SIZE[1] // 2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 160), 1,
        )
        return out
