"""
Integrated Pose Detection Module - Combines new ShoulderAbductionDetector with existing interface
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Dict, Optional
from datetime import datetime
from pose_detector_new import ShoulderAbductionDetector

class PoseDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """Initialize integrated pose detector"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize the new shoulder abduction detector
        self.shoulder_detector = ShoulderAbductionDetector()

        # Initialize pose for other exercises
        try:
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
            print("✅ Integrated PoseDetector initialized")
        except Exception as e:
            print(f"⚠️ MediaPipe init error: {e}")
            self.pose = None

        # Rep counting for other exercises
        self.previous_angles = {}
        self.rep_counters = {}
        self.rep_stage = {}

        # Track if we've initialized shoulder assessment
        self._shoulder_initialized = False

    def reset_assessment(self):
        """Reset shoulder abduction assessment"""
        self.shoulder_detector.reset()
        self._shoulder_initialized = True
        print("🔄 Reset shoulder abduction assessment")

    def reset_exercise_counters(self, exercise_type: str):
        """Reset exercise counters"""
        if exercise_type == 'shoulder_abduction':
            self.reset_assessment()
        else:
            # Reset counters for other exercises
            self.rep_counters[exercise_type] = 0
            self.rep_stage[exercise_type] = None
            print(f"🔄 Reset counters for {exercise_type}")

    def process_frame(self, frame: np.ndarray, exercise_type: str, is_calibration: bool = False) -> Tuple[np.ndarray, Dict]:
        """Process frame based on exercise type"""

        # Handle calibration
        if is_calibration:
            return self._handle_calibration(frame)

        # Use new detector for shoulder abduction
        if exercise_type == 'shoulder_abduction':
            return self._process_shoulder_abduction(frame)

        # Handle other exercises (knee flexion, shoulder flexion)
        return self._process_other_exercise(frame, exercise_type)

    def _handle_calibration(self, frame):
        """Handle calibration frame"""
        annotated_frame = frame.copy()

        # Simple pose detection for calibration
        if self.pose:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            if results.pose_landmarks:
                # Count visible landmarks
                visible_count = sum(
                    1 for lm in results.pose_landmarks.landmark
                    if lm.visibility > 0.1
                )

                # Draw skeleton with green color for successful detection
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

                # Check if enough landmarks are visible (at least 10)
                if visible_count >= 10:
                    return annotated_frame, {
                        'calibration_status': 'success',
                        'full_body_detected': True,
                        'visible_landmarks': visible_count,
                        'detection_confidence': 0.9
                    }

        # Return retrying status if not enough landmarks detected
        return annotated_frame, {
            'error': 'Calibrating... please stay in view',
            'calibration_status': 'retrying',
            'visible_landmarks': 0,
            'detection_confidence': 0
        }

    def _process_shoulder_abduction(self, frame):
        """Process shoulder abduction using new detector"""
        # Initialize if needed
        if not self._shoulder_initialized:
            self.reset_assessment()

        # Process frame with shoulder detector - this modifies the frame with skeleton
        annotated_frame = frame.copy()
        result = self.shoulder_detector.process_frame(annotated_frame)

        # Map the new detector format to expected format
        measurements = {
            'assessment_state': result.get('state'),
            'instruction': result.get('instruction', ''),
            'active_angle': result.get('left_angle', 0) if result.get('test_arm') == 'left' else result.get('right_angle', 0),
            'active_side': result.get('test_arm'),
            'left_angle': result.get('left_angle', 0),
            'right_angle': result.get('right_angle', 0),
            'left_max': result.get('left_max', 0),
            'right_max': result.get('right_max', 0),
            'progress': result.get('progress', 0),
            'complete': result.get('complete', False),
            'error': result.get('error', False)
        }

        # The shoulder detector already drew the skeleton on annotated_frame
        return annotated_frame, measurements

    def _process_other_exercise(self, frame, exercise_type):
        """Process other exercises (knee flexion, shoulder flexion)"""
        annotated_frame = frame.copy()
        measurements = {'error': 'Exercise not fully implemented'}

        if not self.pose:
            return annotated_frame, measurements

        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return annotated_frame, {'error': 'No pose detected'}

        # Draw skeleton
        self.mp_drawing.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
            self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        # Basic measurements for other exercises
        landmarks = results.pose_landmarks.landmark

        if exercise_type == 'knee_flexion':
            # Calculate knee angle
            hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y]
            knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].y]
            ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y]

            angle = self._calculate_angle(hip, knee, ankle)

            measurements = {
                'active_flexion': angle,
                'active_side': 'left',
                'rep_count': self.rep_counters.get(exercise_type, 0),
                'instruction': f'Knee flexion: {int(angle)}°'
            }

        elif exercise_type == 'shoulder_flexion':
            # Calculate shoulder flexion angle
            hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y]
            shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]

            angle = self._calculate_angle(hip, shoulder, elbow)

            measurements = {
                'active_flexion': angle,
                'active_side': 'left',
                'rep_count': self.rep_counters.get(exercise_type, 0),
                'instruction': f'Shoulder flexion: {int(angle)}°'
            }

        # Add text overlay
        if 'instruction' in measurements:
            cv2.putText(annotated_frame, measurements['instruction'], (30, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return annotated_frame, measurements

    def _calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle