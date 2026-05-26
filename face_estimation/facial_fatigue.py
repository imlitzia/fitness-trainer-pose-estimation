"""
Facial fatigue detection from MediaPipe Face Mesh landmarks.

Signals:
- Eye Aspect Ratio (EAR) — drooping / heavy eyelids
- Mouth Aspect Ratio (MAR) — yawning
- Blink rate — elevated blinking when fatigued
- Head stability — increased nodding / sway
"""

import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


# MediaPipe Face Mesh landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 0, 17, 78, 308]
NOSE_TIP = 1
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(landmarks, indices: List[int], w: int, h: int) -> float:
    """EAR from six eye landmarks (p1-p6 order)."""
    pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in indices]
    vertical = _dist(pts[1], pts[5]) + _dist(pts[2], pts[4])
    horizontal = _dist(pts[0], pts[3])
    if horizontal < 1e-6:
        return 0.3
    return vertical / (2.0 * horizontal)


def mouth_aspect_ratio(landmarks, w: int, h: int) -> float:
    """MAR — mouth open / yawn proxy."""
    pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in MOUTH]
    vertical = _dist(pts[2], pts[3]) + _dist(pts[4], pts[5])
    horizontal = _dist(pts[0], pts[1])
    if horizontal < 1e-6:
        return 0.0
    return vertical / (2.0 * horizontal)


class FacialFatigueDetector:
    """Real-time facial fatigue score from face mesh landmarks."""

    EAR_BLINK_THRESHOLD = 0.21
    BASELINE_SEC = 4.0
    WINDOW_SEC = 8.0

    def __init__(self):
        self._session_start: Optional[float] = None
        self._ear_history: Deque[Tuple[float, float]] = deque(maxlen=240)
        self._mar_history: Deque[Tuple[float, float]] = deque(maxlen=120)
        self._nose_history: Deque[Tuple[float, float, float]] = deque(maxlen=90)
        self._blink_times: Deque[float] = deque(maxlen=60)
        self._eye_closed = False

        self._baseline_ear: Optional[float] = None
        self._baseline_mar: Optional[float] = None
        self._baseline_head_std: Optional[float] = None

        self._fatigue_score = 100
        self._fatigue_level = "fresh"
        self._signals: Dict[str, Any] = {}
        self._messages: List[str] = []
        self._face_detected = False

    def reset(self):
        self._session_start = None
        self._ear_history.clear()
        self._mar_history.clear()
        self._nose_history.clear()
        self._blink_times.clear()
        self._eye_closed = False
        self._baseline_ear = None
        self._baseline_mar = None
        self._baseline_head_std = None
        self._fatigue_score = 100
        self._fatigue_level = "fresh"
        self._signals = {}
        self._messages = []
        self._face_detected = False

    def update(self, landmarks, frame_w: int, frame_h: int) -> Dict[str, Any]:
        """Process one frame of face mesh landmarks."""
        ts = time.time()
        if self._session_start is None:
            self._session_start = ts

        self._face_detected = True

        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, frame_w, frame_h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, frame_w, frame_h)
        ear = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(landmarks, frame_w, frame_h)

        nose = landmarks[NOSE_TIP]
        nx, ny = nose.x * frame_w, nose.y * frame_h

        self._ear_history.append((ts, ear))
        self._mar_history.append((ts, mar))
        self._nose_history.append((ts, nx, ny))

        if ear < self.EAR_BLINK_THRESHOLD and not self._eye_closed:
            self._blink_times.append(ts)
            self._eye_closed = True
        elif ear >= self.EAR_BLINK_THRESHOLD:
            self._eye_closed = False

        elapsed = ts - self._session_start
        if elapsed >= self.BASELINE_SEC and self._baseline_ear is None:
            self._compute_baseline()

        if self._baseline_ear is not None:
            self._update_fatigue_score(ts)

        return self.get_status()

    def mark_no_face(self):
        self._face_detected = False

    def _compute_baseline(self):
        if len(self._ear_history) < 10:
            return
        ears = [e for _, e in self._ear_history]
        self._baseline_ear = float(np.mean(ears))
        if self._mar_history:
            self._baseline_mar = float(np.mean([m for _, m in self._mar_history]))
        if len(self._nose_history) >= 10:
            xs = [p[1] for p in self._nose_history]
            ys = [p[2] for p in self._nose_history]
            self._baseline_head_std = float(np.std(xs) + np.std(ys))

    def _recent_blink_rate(self, ts: float) -> float:
        """Blinks per minute over last WINDOW_SEC."""
        cutoff = ts - self.WINDOW_SEC
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()
        count = len(self._blink_times)
        window_min = self.WINDOW_SEC / 60.0
        return count / window_min if window_min > 0 else 0.0

    def _recent_head_jitter(self) -> float:
        if len(self._nose_history) < 8:
            return 0.0
        recent = list(self._nose_history)[-30:]
        xs = [p[1] for p in recent]
        ys = [p[2] for p in recent]
        return float(np.std(xs) + np.std(ys))

    def _update_fatigue_score(self, ts: float):
        recent_ears = [e for t, e in self._ear_history if t >= ts - 2.0]
        recent_mar = [m for t, m in self._mar_history if t >= ts - 2.0]
        if not recent_ears:
            return

        current_ear = float(np.mean(recent_ears))
        current_mar = float(np.mean(recent_mar)) if recent_mar else 0.0
        blink_rate = self._recent_blink_rate(ts)
        head_jitter = self._recent_head_jitter()

        ear_ratio = current_ear / self._baseline_ear if self._baseline_ear else 1.0
        mar_ratio = current_mar / self._baseline_mar if self._baseline_mar and self._baseline_mar > 0 else 1.0
        head_ratio = (
            head_jitter / self._baseline_head_std
            if self._baseline_head_std and self._baseline_head_std > 1
            else 1.0
        )

        ear_score = _ratio_score(ear_ratio, good=1.0, warn=0.92, bad=0.82, higher_better=True)
        mar_score = _ratio_score(mar_ratio, good=1.0, warn=1.35, bad=1.7, higher_better=False)
        blink_score = 100 if blink_rate < 18 else (70 if blink_rate < 28 else max(20, 100 - int(blink_rate * 2)))
        head_score = _ratio_score(head_ratio, good=1.0, warn=1.4, bad=2.0, higher_better=False)

        self._fatigue_score = int(
            0.35 * ear_score + 0.25 * mar_score + 0.20 * blink_score + 0.20 * head_score
        )
        self._fatigue_level = _level(self._fatigue_score)

        self._signals = {
            "ear": {"current": round(current_ear, 3), "baseline": round(self._baseline_ear, 3), "ratio": round(ear_ratio, 2)},
            "mouth": {"current": round(current_mar, 3), "ratio": round(mar_ratio, 2)},
            "blink_rate_per_min": round(blink_rate, 1),
            "head_jitter": round(head_jitter, 1),
            "head_ratio": round(head_ratio, 2),
        }
        self._messages = _facial_messages(ear_ratio, mar_ratio, blink_rate, head_ratio, self._fatigue_level)

    def get_status(self) -> Dict[str, Any]:
        return {
            "facial_fatigue_score": self._fatigue_score,
            "facial_fatigue_level": self._fatigue_level,
            "facial_signals": self._signals,
            "facial_messages": self._messages,
            "face_detected": self._face_detected,
        }


def _ratio_score(ratio, good, warn, bad, higher_better) -> int:
    if higher_better:
        if ratio >= good:
            return 100
        if ratio >= warn:
            return int(70 + 30 * (ratio - warn) / (good - warn))
        if ratio >= bad:
            return int(35 + 35 * (ratio - bad) / (warn - bad))
        return max(0, int(35 * ratio / bad))
    if ratio <= good:
        return 100
    if ratio <= warn:
        return int(70 + 30 * (warn - ratio) / (warn - good))
    if ratio <= bad:
        return int(35 + 35 * (bad - ratio) / (bad - warn))
    return max(0, 20)


def _level(score: int) -> str:
    if score >= 75:
        return "fresh"
    if score >= 55:
        return "moderate"
    if score >= 35:
        return "high"
    return "critical"


def _facial_messages(ear_r, mar_r, blink, head_r, level) -> List[str]:
    msgs = []
    if ear_r < 0.90:
        msgs.append("Eyelids drooping - visual fatigue detected")
    if mar_r > 1.4:
        msgs.append("Frequent yawning / open mouth - rest recommended")
    if blink > 25:
        msgs.append("Elevated blink rate - eyes may be strained")
    if head_r > 1.5:
        msgs.append("Head instability increasing - focus or take a break")
    if level in ("high", "critical") and len(msgs) < 2:
        msgs.append("Facial fatigue building - consider ending the set")
    return msgs[:3]
