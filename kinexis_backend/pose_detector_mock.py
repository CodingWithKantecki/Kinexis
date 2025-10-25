"""
Mock Pose Detection Module for Development
Simulates MediaPipe pose detection for testing without actual MediaPipe
This allows the backend to run on Python 3.13 where MediaPipe isn't available yet
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
import json
from datetime import datetime
import random

class MockPoseLandmark:
    """Mock MediaPipe PoseLandmark enum"""
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

class PoseDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """Initialize Mock Pose detector"""
        print("Warning: Using MOCK pose detector for development (MediaPipe not available for Python 3.13)")

        self.mp_pose = type('obj', (object,), {
            'PoseLandmark': MockPoseLandmark,
            'POSE_CONNECTIONS': []
        })()

        # Store previous angles for rep counting
        self.previous_angles = {}
        self.rep_counters = {}
        self.rep_stage = {}  # 'up' or 'down' for each exercise

        # Simulation variables
        self.frame_count = 0
        self.simulation_angle = 30

    def calculate_angle(self, point1: List[float], point2: List[float], point3: List[float]) -> float:
        """
        Calculate angle between three points
        point2 is the vertex
        """
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def _simulate_movement(self) -> float:
        """Simulate gradual movement for testing"""
        self.frame_count += 1
        # Create a sine wave pattern for natural movement simulation
        angle = 30 + 60 * abs(np.sin(self.frame_count * 0.05))
        return angle

    def detect_shoulder_abduction(self, landmarks=None) -> Dict:
        """Mock shoulder abduction detection"""
        # Simulate angle changes
        simulated_angle = self._simulate_movement()

        # Count reps based on angle thresholds
        rep_counted = self._count_reps(
            'shoulder_abduction',
            simulated_angle,
            threshold_up=70,
            threshold_down=40
        )

        return {
            'exercise': 'shoulder_abduction',
            'right_angle': simulated_angle + random.uniform(-5, 5),
            'left_angle': 30 + random.uniform(-5, 5),
            'active_side': 'right',
            'active_angle': simulated_angle,
            'rep_count': self.rep_counters.get('shoulder_abduction', 0),
            'rep_stage': self.rep_stage.get('shoulder_abduction', 'down'),
            'rep_counted': rep_counted,
            'target_angle': 150,
            'progress_percentage': min(100, (simulated_angle / 150) * 100),
            'detection_confidence': 0.95,
            'mock_data': True  # Flag to indicate this is simulated
        }

    def detect_knee_flexion(self, landmarks=None) -> Dict:
        """Mock knee flexion detection"""
        # Simulate angle changes
        simulated_flexion = self._simulate_movement()

        # Count reps based on flexion thresholds
        rep_counted = self._count_reps(
            'knee_flexion',
            simulated_flexion,
            threshold_up=70,
            threshold_down=30
        )

        return {
            'exercise': 'knee_flexion',
            'right_flexion': simulated_flexion + random.uniform(-3, 3),
            'left_flexion': 20 + random.uniform(-3, 3),
            'active_side': 'right',
            'active_flexion': simulated_flexion,
            'rep_count': self.rep_counters.get('knee_flexion', 0),
            'rep_stage': self.rep_stage.get('knee_flexion', 'down'),
            'rep_counted': rep_counted,
            'target_angle': 120,
            'progress_percentage': min(100, (simulated_flexion / 120) * 100),
            'detection_confidence': 0.93,
            'mock_data': True
        }

    def detect_shoulder_flexion(self, landmarks=None) -> Dict:
        """Mock shoulder flexion detection"""
        # Simulate angle changes
        simulated_angle = self._simulate_movement()

        # Count reps based on angle thresholds
        rep_counted = self._count_reps(
            'shoulder_flexion',
            simulated_angle,
            threshold_up=70,
            threshold_down=40
        )

        return {
            'exercise': 'shoulder_flexion',
            'right_angle': simulated_angle + random.uniform(-4, 4),
            'left_angle': 25 + random.uniform(-4, 4),
            'active_side': 'right',
            'active_angle': simulated_angle,
            'rep_count': self.rep_counters.get('shoulder_flexion', 0),
            'rep_stage': self.rep_stage.get('shoulder_flexion', 'down'),
            'rep_counted': rep_counted,
            'target_angle': 150,
            'progress_percentage': min(100, (simulated_angle / 150) * 100),
            'detection_confidence': 0.94,
            'mock_data': True
        }

    def _count_reps(self, exercise: str, angle: float, threshold_up: float, threshold_down: float) -> bool:
        """Count repetitions based on angle thresholds"""
        if exercise not in self.rep_counters:
            self.rep_counters[exercise] = 0
            self.rep_stage[exercise] = 'down'

        rep_counted = False

        # Check if moving up
        if angle > threshold_up and self.rep_stage[exercise] == 'down':
            self.rep_stage[exercise] = 'up'

        # Check if completed rep (moving back down)
        elif angle < threshold_down and self.rep_stage[exercise] == 'up':
            self.rep_stage[exercise] = 'down'
            self.rep_counters[exercise] += 1
            rep_counted = True

        return rep_counted

    def process_frame(self, frame: np.ndarray, exercise_type: str) -> Tuple[np.ndarray, Dict]:
        """
        Mock process frame - returns original frame with simulated measurements
        """
        import cv2

        measurements = {}

        # Get measurements based on exercise type
        if exercise_type == 'shoulder_abduction':
            measurements = self.detect_shoulder_abduction()
        elif exercise_type == 'knee_flexion':
            measurements = self.detect_knee_flexion()
        elif exercise_type == 'shoulder_flexion':
            measurements = self.detect_shoulder_flexion()
        else:
            measurements = {'error': f"Unknown exercise type: {exercise_type}"}

        # Simply return the frame with minimal overlay
        try:
            if frame is not None and len(frame.shape) == 3:
                # Make sure we're working with the actual frame
                annotated_frame = frame.copy()
                h, w, c = frame.shape

                print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")  # Debug

                # Draw mock skeleton points - scale based on actual frame size
                skeleton_points = [
                    (w//2, int(h*0.25)),      # head
                    (w//2, int(h*0.35)),      # neck
                    (int(w*0.4), int(h*0.35)), # left shoulder
                    (int(w*0.6), int(h*0.35)), # right shoulder
                    (int(w*0.35), int(h*0.5)), # left elbow
                    (int(w*0.65), int(h*0.5)), # right elbow
                    (w//2, int(h*0.5)),      # center
                    (int(w*0.45), int(h*0.65)), # left hip
                    (int(w*0.55), int(h*0.65)), # right hip
                    (int(w*0.45), int(h*0.85)), # left knee
                    (int(w*0.55), int(h*0.85)), # right knee
                ]

                # Draw skeleton points with transparency
                overlay = annotated_frame.copy()
                for point in skeleton_points:
                    cv2.circle(overlay, point, 8, (0, 255, 0), -1)

                # Draw skeleton connections
                connections = [
                    (0, 1), (1, 2), (1, 3),  # head to shoulders
                    (2, 4), (3, 5),          # shoulders to elbows
                    (1, 6), (6, 7), (6, 8),  # neck to hips
                    (7, 9), (8, 10)          # hips to knees
                ]

                for start_idx, end_idx in connections:
                    cv2.line(overlay, skeleton_points[start_idx],
                            skeleton_points[end_idx], (0, 255, 0), 3)

                # Blend overlay with original frame for transparency
                annotated_frame = cv2.addWeighted(annotated_frame, 0.7, overlay, 0.3, 0)

                # Add text overlay
                cv2.putText(annotated_frame, "MOCK SKELETON - MediaPipe not available",
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Add angle info
                if 'active_angle' in measurements or 'active_flexion' in measurements:
                    angle_value = measurements.get('active_angle', measurements.get('active_flexion', 0))
                    cv2.putText(annotated_frame, f"Angle: {angle_value:.1f}°",
                               (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(annotated_frame, f"Reps: {measurements.get('rep_count', 0)}",
                               (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                return annotated_frame, measurements
            else:
                # If frame is invalid, return a black frame
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(black_frame, "No video feed",
                           (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return black_frame, measurements
        except Exception as e:
            print(f"Error in mock process_frame: {e}")
            # Return original frame if there's an error
            return frame, measurements

    def _draw_exercise_info(self, frame: np.ndarray, measurements: Dict, angle: float):
        """Draw exercise information overlay on frame"""
        import cv2
        h, w = frame.shape[:2]

        # Draw semi-transparent background for info
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h-150), (400, h-10), (0, 0, 0), -1)
        frame_with_overlay = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        # Draw exercise name
        cv2.putText(frame_with_overlay, measurements['exercise'].replace('_', ' ').title(),
                   (20, h-120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw angle
        cv2.putText(frame_with_overlay, f"Angle: {angle:.1f}°",
                   (20, h-90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw rep count
        cv2.putText(frame_with_overlay, f"Reps: {measurements.get('rep_count', 0)}",
                   (20, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw progress bar
        progress = measurements.get('progress_percentage', 0)
        bar_width = int((progress / 100) * 200)
        cv2.rectangle(frame_with_overlay, (20, h-40), (220, h-20), (100, 100, 100), 2)
        cv2.rectangle(frame_with_overlay, (20, h-40), (20 + bar_width, h-20), (0, 255, 0), -1)
        cv2.putText(frame_with_overlay, f"{progress:.0f}%",
                   (230, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame_with_overlay

    def reset_exercise_counters(self, exercise: Optional[str] = None):
        """Reset rep counters for specific exercise or all exercises"""
        if exercise:
            self.rep_counters[exercise] = 0
            self.rep_stage[exercise] = 'down'
        else:
            self.rep_counters = {}
            self.rep_stage = {}
        self.frame_count = 0

    def close(self):
        """Clean up resources"""
        pass