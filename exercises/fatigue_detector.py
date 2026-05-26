"""
Real-time fatigue detection from movement quality signals.

Tracks per-rep and rolling metrics:
- Rep velocity (ROM / rep duration)
- Movement shakiness (angle jitter during the rep)
- ROM reduction vs early-rep baseline
- Longer pauses between reps
"""

import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple


class FatigueDetector:
    """Detects accumulating fatigue during a repetition-based exercise set."""

    BASELINE_REPS = 2
    COMPARISON_REPS = 3
    MIN_REPS_FOR_ANALYSIS = 3

    def __init__(
        self,
        primary_angle_key: str = "primary",
        baseline_reps: int = 2,
        comparison_reps: int = 3,
    ):
        self.primary_angle_key = primary_angle_key
        self.baseline_reps = baseline_reps
        self.comparison_reps = comparison_reps

        self._frame_buffer: Deque[Tuple[float, float]] = deque(maxlen=30)
        self._rep_active = False
        self._rep_start_time: Optional[float] = None
        self._rep_angle_min = 180.0
        self._rep_angle_max = 0.0
        self._rep_jitter_samples: List[float] = []

        self._last_rep_end_time: Optional[float] = None
        self._completed_reps: List[Dict[str, float]] = []

        self._live_shakiness = 0.0
        self._fatigue_score = 100
        self._fatigue_level = "fresh"
        self._signals: Dict[str, Any] = {}
        self._messages: List[str] = []

    def reset(self):
        """Clear all session metrics."""
        self._frame_buffer.clear()
        self._rep_active = False
        self._rep_start_time = None
        self._rep_angle_min = 180.0
        self._rep_angle_max = 0.0
        self._rep_jitter_samples.clear()
        self._last_rep_end_time = None
        self._completed_reps.clear()
        self._live_shakiness = 0.0
        self._fatigue_score = 100
        self._fatigue_level = "fresh"
        self._signals = {}
        self._messages = []

    def start_rep(self, timestamp: Optional[float] = None):
        """Mark the beginning of a new repetition."""
        ts = timestamp or time.time()
        self._rep_active = True
        self._rep_start_time = ts
        self._rep_angle_min = 180.0
        self._rep_angle_max = 0.0
        self._rep_jitter_samples.clear()

        if self._last_rep_end_time is not None:
            pause = ts - self._last_rep_end_time
            if self._completed_reps:
                self._completed_reps[-1]["inter_rep_pause"] = pause

    def update_frame(
        self,
        primary_angle: float,
        timestamp: Optional[float] = None,
        in_movement: bool = True,
    ):
        """
        Feed per-frame primary joint angle for live shakiness and ROM tracking.

        Args:
            primary_angle: Main exercise angle (degrees)
            timestamp: Frame time (seconds)
            in_movement: False when resting between reps at top position
        """
        ts = timestamp or time.time()
        self._frame_buffer.append((ts, primary_angle))

        if not in_movement:
            return

        if self._rep_active:
            self._rep_angle_min = min(self._rep_angle_min, primary_angle)
            self._rep_angle_max = max(self._rep_angle_max, primary_angle)

        if len(self._frame_buffer) >= 3:
            t0, a0 = self._frame_buffer[-3]
            t1, a1 = self._frame_buffer[-2]
            t2, a2 = self._frame_buffer[-1]
            vel1 = (a1 - a0) / max(t1 - t0, 1e-6)
            vel2 = (a2 - a1) / max(t2 - t1, 1e-6)
            jitter = abs(vel2 - vel1)
            if self._rep_active:
                self._rep_jitter_samples.append(jitter)
            self._live_shakiness = self._ema(self._live_shakiness, jitter, 0.25)

    @staticmethod
    def _ema(prev: float, value: float, alpha: float) -> float:
        if prev <= 0:
            return value
        return alpha * value + (1 - alpha) * prev

    def complete_rep(self, rep_duration: float, timestamp: Optional[float] = None):
        """
        Finalize metrics when a rep is counted.

        Args:
            rep_duration: Full rep cycle time in seconds
            timestamp: Rep completion time
        """
        ts = timestamp or time.time()
        rom = max(self._rep_angle_max - self._rep_angle_min, 1.0)
        velocity = rom / max(rep_duration, 0.3)
        shakiness = (
            sum(self._rep_jitter_samples) / len(self._rep_jitter_samples)
            if self._rep_jitter_samples
            else self._live_shakiness
        )

        inter_rep_pause = 0.0
        if self._completed_reps and "inter_rep_pause" not in self._completed_reps[-1]:
            if self._rep_start_time and self._last_rep_end_time:
                inter_rep_pause = self._rep_start_time - self._last_rep_end_time

        rep_data = {
            "rom": rom,
            "velocity": velocity,
            "shakiness": shakiness,
            "duration": rep_duration,
            "inter_rep_pause": inter_rep_pause,
        }
        self._completed_reps.append(rep_data)

        self._rep_active = False
        self._last_rep_end_time = ts
        self._rep_start_time = None

        self._update_fatigue_assessment()

    def _update_fatigue_assessment(self):
        """Recompute composite fatigue score from completed reps."""
        n = len(self._completed_reps)
        if n < self.MIN_REPS_FOR_ANALYSIS:
            self._fatigue_score = 100
            self._fatigue_level = "fresh" if n == 0 else "warming_up"
            self._signals = {"reps_recorded": n, "reps_needed": self.MIN_REPS_FOR_ANALYSIS}
            self._messages = (
                [f"Collecting baseline ({n}/{self.MIN_REPS_FOR_ANALYSIS} reps)"]
                if n > 0
                else []
            )
            return

        baseline = self._completed_reps[: self.baseline_reps]
        recent = self._completed_reps[-self.comparison_reps :]

        base_vel = _mean([r["velocity"] for r in baseline])
        base_rom = _mean([r["rom"] for r in baseline])
        base_shake = _mean([r["shakiness"] for r in baseline])
        base_pause = _mean([r["inter_rep_pause"] for r in baseline if r["inter_rep_pause"] > 0])

        rec_vel = _mean([r["velocity"] for r in recent])
        rec_rom = _mean([r["rom"] for r in recent])
        rec_shake = _mean([r["shakiness"] for r in recent])
        rec_pause = _mean([r["inter_rep_pause"] for r in recent if r["inter_rep_pause"] > 0])

        vel_ratio = rec_vel / base_vel if base_vel > 0 else 1.0
        rom_ratio = rec_rom / base_rom if base_rom > 0 else 1.0
        shake_ratio = rec_shake / base_shake if base_shake > 0 else 1.0
        pause_ratio = rec_pause / base_pause if base_pause > 0 else 1.0

        vel_score = _ratio_to_score(vel_ratio, good=1.0, warn=0.88, bad=0.72, higher_is_better=True)
        rom_score = _ratio_to_score(rom_ratio, good=1.0, warn=0.92, bad=0.82, higher_is_better=True)
        shake_score = _ratio_to_score(
            shake_ratio, good=1.0, warn=1.25, bad=1.6, higher_is_better=False
        )
        pause_score = _ratio_to_score(
            pause_ratio, good=1.0, warn=1.35, bad=1.8, higher_is_better=False
        )

        self._fatigue_score = int(
            0.30 * vel_score + 0.30 * rom_score + 0.20 * shake_score + 0.20 * pause_score
        )
        self._fatigue_level = _score_to_level(self._fatigue_score)

        self._signals = {
            "reps_recorded": n,
            "velocity": {
                "current_deg_per_sec": round(rec_vel, 1),
                "baseline_deg_per_sec": round(base_vel, 1),
                "ratio": round(vel_ratio, 2),
                "score": vel_score,
            },
            "rom": {
                "current_deg": round(rec_rom, 1),
                "baseline_deg": round(base_rom, 1),
                "ratio": round(rom_ratio, 2),
                "score": rom_score,
            },
            "shakiness": {
                "current": round(rec_shake, 2),
                "baseline": round(base_shake, 2),
                "ratio": round(shake_ratio, 2),
                "score": shake_score,
                "live": round(self._live_shakiness, 2),
            },
            "pause": {
                "current_sec": round(rec_pause, 2),
                "baseline_sec": round(base_pause, 2),
                "ratio": round(pause_ratio, 2),
                "score": pause_score,
            },
        }
        self._messages = _build_messages(
            vel_ratio, rom_ratio, shake_ratio, pause_ratio, self._fatigue_level
        )

    def get_status(self) -> Dict[str, Any]:
        """Current fatigue assessment for API and UI."""
        return {
            "fatigue_score": self._fatigue_score,
            "fatigue_level": self._fatigue_level,
            "live_shakiness": round(self._live_shakiness, 2),
            "reps_analyzed": len(self._completed_reps),
            "signals": self._signals,
            "messages": self._messages,
        }

    def get_overlay_color_bgr(self) -> Tuple[int, int, int]:
        """BGR color for on-frame fatigue indicator."""
        score = self._fatigue_score
        if score >= 75:
            return (0, 200, 0)
        if score >= 50:
            return (0, 200, 255)
        if score >= 30:
            return (0, 140, 255)
        return (0, 0, 220)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ratio_to_score(
    ratio: float,
    good: float,
    warn: float,
    bad: float,
    higher_is_better: bool,
) -> int:
    """Map a metric ratio to a 0-100 sub-score."""
    if higher_is_better:
        if ratio >= good:
            return 100
        if ratio >= warn:
            return int(70 + 30 * (ratio - warn) / (good - warn))
        if ratio >= bad:
            return int(35 + 35 * (ratio - bad) / (warn - bad))
        return max(0, int(35 * ratio / bad))
    # Lower is better (shakiness, pause)
    if ratio <= good:
        return 100
    if ratio <= warn:
        return int(70 + 30 * (warn - ratio) / (warn - good))
    if ratio <= bad:
        return int(35 + 35 * (bad - ratio) / (bad - warn))
    return max(0, int(35 * (2.0 - ratio) / (2.0 - bad)))


def _score_to_level(score: int) -> str:
    if score >= 75:
        return "fresh"
    if score >= 55:
        return "moderate"
    if score >= 35:
        return "high"
    return "critical"


def _build_messages(
    vel_ratio: float,
    rom_ratio: float,
    shake_ratio: float,
    pause_ratio: float,
    level: str,
) -> List[str]:
    msgs = []
    if vel_ratio < 0.88:
        msgs.append("Rep velocity slowing - consider shorter rest or fewer reps")
    if rom_ratio < 0.90:
        msgs.append("Range of motion decreasing - fatigue may limit depth")
    if shake_ratio > 1.25:
        msgs.append("Movement becoming shakier - stabilize before next rep")
    if pause_ratio > 1.35:
        msgs.append("Longer pauses between reps - take a set break soon")
    if level == "critical" and not msgs:
        msgs.append("High fatigue detected - end set or rest")
    elif level == "high" and len(msgs) < 2:
        msgs.append("Fatigue building - monitor form closely")
    return msgs[:3]
