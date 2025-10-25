"""
Kinexis Pose Detection Module
Handles pose detection and angle calculation for PT exercises
Supported exercises: Shoulder Abduction, Knee Flexion, Shoulder Flexion
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Dict, List, Optional
import json
from datetime import datetime

class PoseDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """Initialize MediaPipe Pose detector - Optimized for PT Applications"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Full model - Best balance for PT (0=lite, 1=full, 2=heavy)
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            smooth_landmarks=True,
            enable_segmentation=False  # Disabled for better performance
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Store previous angles for rep counting
        self.previous_angles = {}
        self.rep_counters = {}
        self.rep_stage = {}  # 'up' or 'down' for each exercise

        # Medical accuracy improvements
        self.angle_history = {}  # Store angle history for smoothing
        self.history_size = 3  # Reduced for more responsive tracking at higher FPS
        self.min_visibility_threshold = 0.5  # Minimum landmark visibility for measurements

    def calculate_angle(self, point1: List[float], point2: List[float], point3: List[float]) -> float:
        """
        Calculate angle between three points with sub-degree precision
        point2 is the vertex
        """
        # Convert to numpy arrays with double precision for medical accuracy
        a = np.array(point1, dtype=np.float64)  # First point
        b = np.array(point2, dtype=np.float64)  # Mid point (vertex)
        c = np.array(point3, dtype=np.float64)  # End point

        # Calculate vectors with higher precision
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        # Ensure angle is between 0-180
        if angle > 180.0:
            angle = 360 - angle

        return angle

    def smooth_angle(self, exercise: str, side: str, angle: float) -> float:
        """Apply moving average filter for smoother, more accurate angle readings"""
        key = f"{exercise}_{side}"

        if key not in self.angle_history:
            self.angle_history[key] = []

        self.angle_history[key].append(angle)

        # Keep only recent history
        if len(self.angle_history[key]) > self.history_size:
            self.angle_history[key].pop(0)

        # Return smoothed average
        return np.mean(self.angle_history[key])

    def detect_shoulder_abduction(self, landmarks) -> Dict:
        """
        Detect shoulder abduction (arm out to side)
        Normal range: 0-180°
        Post-surgery typical goal: 80° → 150°
        """
        try:
            # Check landmark visibility for medical accuracy
            right_hip_lm = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            right_shoulder_lm = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_elbow_lm = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            left_hip_lm = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            left_shoulder_lm = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_elbow_lm = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]

            # Check minimum visibility threshold for medical accuracy
            right_visible = all([
                right_hip_lm.visibility > self.min_visibility_threshold,
                right_shoulder_lm.visibility > self.min_visibility_threshold,
                right_elbow_lm.visibility > self.min_visibility_threshold
            ])
            left_visible = all([
                left_hip_lm.visibility > self.min_visibility_threshold,
                left_shoulder_lm.visibility > self.min_visibility_threshold,
                left_elbow_lm.visibility > self.min_visibility_threshold
            ])

            if not (right_visible or left_visible):
                return {'error': 'Insufficient landmark visibility for accurate measurement'}

            # Get coordinates with visibility check
            right_hip = [right_hip_lm.x, right_hip_lm.y]
            right_shoulder = [right_shoulder_lm.x, right_shoulder_lm.y]
            right_elbow = [right_elbow_lm.x, right_elbow_lm.y]
            left_hip = [left_hip_lm.x, left_hip_lm.y]
            left_shoulder = [left_shoulder_lm.x, left_shoulder_lm.y]
            left_elbow = [left_elbow_lm.x, left_elbow_lm.y]

            # Calculate raw angles
            right_angle_raw = self.calculate_angle(right_hip, right_shoulder, right_elbow) if right_visible else 0
            left_angle_raw = self.calculate_angle(left_hip, left_shoulder, left_elbow) if left_visible else 0

            # Apply smoothing for medical accuracy
            right_angle = self.smooth_angle('shoulder_abduction', 'right', right_angle_raw) if right_visible else 0
            left_angle = self.smooth_angle('shoulder_abduction', 'left', left_angle_raw) if left_visible else 0

            # Determine which side is active (higher angle)
            active_side = "right" if right_angle > left_angle else "left"
            active_angle = max(right_angle, left_angle)

            # Count reps based on angle thresholds
            rep_counted = self._count_reps(
                'shoulder_abduction',
                active_angle,
                threshold_up=100,  # Arm raised above 100°
                threshold_down=30  # Arm lowered below 30°
            )

            # Calculate average visibility for medical confidence score
            visibility_scores = []
            if right_visible:
                visibility_scores.extend([right_hip_lm.visibility, right_shoulder_lm.visibility, right_elbow_lm.visibility])
            if left_visible:
                visibility_scores.extend([left_hip_lm.visibility, left_shoulder_lm.visibility, left_elbow_lm.visibility])
            avg_visibility = np.mean(visibility_scores) if visibility_scores else 0

            return {
                'exercise': 'shoulder_abduction',
                'right_angle': round(right_angle, 1),
                'left_angle': round(left_angle, 1),
                'active_side': active_side,
                'active_angle': round(active_angle, 1),
                'rep_count': self.rep_counters.get('shoulder_abduction', 0),
                'rep_stage': self.rep_stage.get('shoulder_abduction', 'down'),
                'rep_counted': rep_counted,
                'target_angle': 150,  # Target for post-surgery recovery
                'progress_percentage': min(100, (active_angle / 150) * 100),
                'measurement_confidence': round(avg_visibility * 100, 1),  # Medical confidence score
                'model_complexity': 1  # Full model - optimized for PT
            }

        except Exception as e:
            return {'error': f"Failed to detect shoulder abduction: {str(e)}"}

    def detect_knee_flexion(self, landmarks) -> Dict:
        """
        Detect knee flexion (bending knee)
        Normal range: 0-135°
        Post-ACL surgery goal: 60° → 120°
        """
        try:
            # Get coordinates for RIGHT knee flexion
            right_hip = [
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y
            ]
            right_knee = [
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y
            ]
            right_ankle = [
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
            ]

            # Get coordinates for LEFT knee flexion
            left_hip = [
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y
            ]
            left_knee = [
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y
            ]
            left_ankle = [
                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y
            ]

            # Calculate angles (180° = straight, less = bent)
            right_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            left_angle = self.calculate_angle(left_hip, left_knee, left_ankle)

            # Convert to flexion angle (0° = straight, higher = more bent)
            right_flexion = 180 - right_angle
            left_flexion = 180 - left_angle

            # Determine which side is active (higher flexion)
            active_side = "right" if right_flexion > left_flexion else "left"
            active_flexion = max(right_flexion, left_flexion)

            # Count reps based on flexion thresholds
            rep_counted = self._count_reps(
                'knee_flexion',
                active_flexion,
                threshold_up=90,   # Knee bent beyond 90°
                threshold_down=20  # Knee straightened below 20°
            )

            return {
                'exercise': 'knee_flexion',
                'right_flexion': round(right_flexion, 1),
                'left_flexion': round(left_flexion, 1),
                'active_side': active_side,
                'active_flexion': round(active_flexion, 1),
                'rep_count': self.rep_counters.get('knee_flexion', 0),
                'rep_stage': self.rep_stage.get('knee_flexion', 'down'),
                'rep_counted': rep_counted,
                'target_angle': 120,  # Target for post-ACL recovery
                'progress_percentage': min(100, (active_flexion / 120) * 100)
            }

        except Exception as e:
            return {'error': f"Failed to detect knee flexion: {str(e)}"}

    def detect_shoulder_flexion(self, landmarks) -> Dict:
        """
        Detect shoulder flexion (arm forward/up)
        Normal range: 0-180°
        Post-surgery typical goal: 80° → 150°
        """
        try:
            # Get coordinates for RIGHT shoulder flexion
            right_hip = [
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y
            ]
            right_shoulder = [
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            ]
            right_elbow = [
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
            ]

            # Get coordinates for LEFT shoulder flexion
            left_hip = [
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y
            ]
            left_shoulder = [
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            ]
            left_elbow = [
                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y
            ]

            # Calculate angles
            right_angle = self.calculate_angle(right_hip, right_shoulder, right_elbow)
            left_angle = self.calculate_angle(left_hip, left_shoulder, left_elbow)

            # Determine which side is active (higher angle)
            active_side = "right" if right_angle > left_angle else "left"
            active_angle = max(right_angle, left_angle)

            # Count reps based on angle thresholds
            rep_counted = self._count_reps(
                'shoulder_flexion',
                active_angle,
                threshold_up=110,  # Arm raised above 110°
                threshold_down=30  # Arm lowered below 30°
            )

            return {
                'exercise': 'shoulder_flexion',
                'right_angle': round(right_angle, 1),
                'left_angle': round(left_angle, 1),
                'active_side': active_side,
                'active_angle': round(active_angle, 1),
                'rep_count': self.rep_counters.get('shoulder_flexion', 0),
                'rep_stage': self.rep_stage.get('shoulder_flexion', 'down'),
                'rep_counted': rep_counted,
                'target_angle': 150,  # Target for post-surgery recovery
                'progress_percentage': min(100, (active_angle / 150) * 100)
            }

        except Exception as e:
            return {'error': f"Failed to detect shoulder flexion: {str(e)}"}

    def _count_reps(self, exercise: str, angle: float, threshold_up: float, threshold_down: float) -> bool:
        """
        Count repetitions based on angle thresholds
        Returns True if a new rep was counted
        """
        # Initialize if not exists
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

    def process_frame(self, frame: np.ndarray, exercise_type: str, is_calibration: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame for pose detection and exercise measurement

        Args:
            frame: Input video frame (BGR format)
            exercise_type: One of 'shoulder_abduction', 'knee_flexion', 'shoulder_flexion'
            is_calibration: If True, check for full body visibility

        Returns:
            Annotated frame and exercise measurements
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Process with MediaPipe
        results = self.pose.process(rgb_frame)

        # Convert back to BGR for OpenCV
        rgb_frame.flags.writeable = True
        annotated_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        measurements = {}
        visible_count = 0  # Initialize here for scope

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Define key landmarks for calibration
            key_landmarks = [
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_HIP,
                self.mp_pose.PoseLandmark.LEFT_KNEE,
                self.mp_pose.PoseLandmark.RIGHT_KNEE,
                self.mp_pose.PoseLandmark.LEFT_ANKLE,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE,
                self.mp_pose.PoseLandmark.LEFT_ELBOW,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW
            ]

            # If calibrating, check for full body visibility
            if is_calibration:
                # Check visibility of all key landmarks
                min_visibility = 1.0
                visibility_details = []

                for landmark_idx in key_landmarks:
                    visibility = landmarks[landmark_idx].visibility
                    landmark_name = landmark_idx.name if hasattr(landmark_idx, 'name') else str(landmark_idx)
                    visibility_details.append(f"{landmark_name}:{visibility:.2f}")

                    if visibility >= 0.15:  # Much more lenient - count landmarks with at least 15% visibility
                        visible_count += 1
                    min_visibility = min(min_visibility, visibility)

                # Much more lenient - allow calibration if at least 4 out of 10 key landmarks are visible
                print(f"Calibration check: {visible_count}/10 landmarks visible (threshold: 0.15)")
                print(f"Visibility details: {', '.join(visibility_details[:4])}...")
                print(f"Min visibility: {min_visibility:.2f}, Average: {sum(landmarks[idx].visibility for idx in key_landmarks)/len(key_landmarks):.2f}")

                if visible_count < 4:  # Much more lenient - only need 4 landmarks
                    measurements = {
                        'error': f'Need more of your body visible - only {visible_count}/10 landmarks detected',
                        'detection_confidence': min_visibility,
                        'calibration_status': 'failed',
                        'visible_landmarks': visible_count
                    }
                    print(f"Calibration failed: Only {visible_count}/10 landmarks visible (need at least 4)")
                    return annotated_frame, measurements

            # If calibrating and successful, return calibration success immediately
            if is_calibration:
                # Draw pose landmarks on frame
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )

                # For calibration, return ONLY calibration data
                total_visibility = sum(landmarks[idx].visibility for idx in key_landmarks)
                measurements = {
                    'calibration_status': 'success',
                    'full_body_detected': True,
                    'visible_landmarks': visible_count,
                    'detection_confidence': total_visibility / len(key_landmarks)
                }
                print(f"Calibration SUCCESS: {visible_count}/10 landmarks, confidence: {measurements['detection_confidence']:.2f}")
                return annotated_frame, measurements

            # For non-calibration, draw pose and get exercise measurements
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Get measurements based on exercise type
            if exercise_type == 'shoulder_abduction':
                measurements = self.detect_shoulder_abduction(results.pose_landmarks.landmark)
            elif exercise_type == 'knee_flexion':
                measurements = self.detect_knee_flexion(results.pose_landmarks.landmark)
            elif exercise_type == 'shoulder_flexion':
                measurements = self.detect_shoulder_flexion(results.pose_landmarks.landmark)
            else:
                measurements = {'error': f"Unknown exercise type: {exercise_type}"}

            # Add pose detection confidence for regular tracking
            if 'error' not in measurements:
                # For regular tracking, use nose visibility as confidence
                measurements['detection_confidence'] = results.pose_landmarks.landmark[0].visibility

            # Draw exercise info on frame
            if 'active_angle' in measurements or 'active_flexion' in measurements:
                angle_value = measurements.get('active_angle', measurements.get('active_flexion', 0))
                self._draw_exercise_info(annotated_frame, measurements, angle_value)
        else:
            # No pose detected at all
            if is_calibration:
                print("Calibration failed: No pose detected at all")
                measurements = {
                    'error': 'No body detected - please step into view',
                    'calibration_status': 'failed',
                    'detection_confidence': 0,
                    'visible_landmarks': 0
                }
            else:
                measurements = {'error': 'No pose detected', 'detection_confidence': 0}

        return annotated_frame, measurements

    def _draw_exercise_info(self, frame: np.ndarray, measurements: Dict, angle: float):
        """Draw exercise information overlay on frame"""
        h, w = frame.shape[:2]

        # Draw semi-transparent background for info
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h-150), (400, h-10), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        # Draw exercise name
        cv2.putText(frame, measurements['exercise'].replace('_', ' ').title(),
                   (20, h-120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw angle
        cv2.putText(frame, f"Angle: {angle:.1f}°",
                   (20, h-90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw rep count
        cv2.putText(frame, f"Reps: {measurements.get('rep_count', 0)}",
                   (20, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw progress bar
        progress = measurements.get('progress_percentage', 0)
        bar_width = int((progress / 100) * 200)
        cv2.rectangle(frame, (20, h-40), (220, h-20), (100, 100, 100), 2)
        cv2.rectangle(frame, (20, h-40), (20 + bar_width, h-20), (0, 255, 0), -1)
        cv2.putText(frame, f"{progress:.0f}%",
                   (230, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def reset_exercise_counters(self, exercise: Optional[str] = None):
        """Reset rep counters for specific exercise or all exercises"""
        if exercise:
            self.rep_counters[exercise] = 0
            self.rep_stage[exercise] = 'down'
        else:
            self.rep_counters = {}
            self.rep_stage = {}

    def close(self):
        """Clean up resources"""
        self.pose.close()