"""
Integrated Pose Detection Module V2 - Uses enhanced shoulder abduction with proper flow
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Dict, Optional
from datetime import datetime
from shoulder_abduction_v2 import ShoulderAbductionV2

class PoseDetectorV2:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """Initialize integrated pose detector V2"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize the enhanced shoulder abduction detector
        self.shoulder_detector = ShoulderAbductionV2()

        # Initialize pose for other exercises with optimal settings
        # Model complexity: 0=Lite, 1=Full, 2=Heavy
        # We use 1 (Full) for good balance of speed and accuracy
        try:
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # Full model - good balance of speed and accuracy
                smooth_landmarks=True,  # Enable landmark smoothing
                enable_segmentation=False,  # Disable segmentation for speed
                smooth_segmentation=False,
                min_detection_confidence=0.4,  # Balanced threshold
                min_tracking_confidence=0.4   # Balanced threshold for tracking
            )
            print("✅ PoseDetectorV2 initialized with enhanced shoulder abduction")
        except Exception as e:
            print(f"⚠️ MediaPipe init error: {e}")
            self.pose = None

        # Rep counting for other exercises
        self.previous_angles = {}
        self.rep_counters = {}
        self.rep_stage = {}

        # Calibration state
        self.calibration_complete = False

    def reset_assessment(self):
        """Reset shoulder abduction assessment"""
        self.shoulder_detector.reset()
        self.calibration_complete = True  # Mark calibration as done when resetting
        print("🔄 Reset shoulder abduction assessment - ready for ROM test")

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

        # Use enhanced detector for shoulder abduction
        if exercise_type == 'shoulder_abduction':
            return self._process_shoulder_abduction_v2(frame)

        # Handle other exercises (knee flexion, shoulder flexion)
        return self._process_other_exercise(frame, exercise_type)

    def _handle_calibration(self, frame):
        """Handle calibration frame"""
        annotated_frame = frame.copy()

        # Simple pose detection for calibration
        if self.pose:
            try:
                # Ensure frame is the right format
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    print(f"⚠️ Invalid frame shape for calibration: {frame.shape}")
                    return annotated_frame, {
                        'error': 'Invalid frame format',
                        'calibration_status': 'error',
                        'visible_landmarks': 0
                    }

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(image_rgb)

                print(f"🔍 MediaPipe processing: pose_detected={results.pose_landmarks is not None}")
            except Exception as e:
                print(f"❌ MediaPipe error during calibration: {e}")
                return annotated_frame, {
                    'error': f'Detection error: {str(e)}',
                    'calibration_status': 'error',
                    'visible_landmarks': 0
                }

            if results.pose_landmarks:
                # Count visible landmarks with reasonable threshold
                visible_count = sum(
                    1 for lm in results.pose_landmarks.landmark
                    if lm.visibility > 0.3  # Reasonable visibility threshold
                )

                # Also count high confidence landmarks
                high_confidence_count = sum(
                    1 for lm in results.pose_landmarks.landmark
                    if lm.visibility > 0.5
                )

                # Draw skeleton with color based on detection quality
                if visible_count >= 10:
                    skeleton_color = (0, 255, 0)  # Green for good detection
                else:
                    skeleton_color = (255, 255, 0)  # Yellow for partial detection

                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=skeleton_color, thickness=3, circle_radius=4),
                    self.mp_drawing.DrawingSpec(color=skeleton_color, thickness=3, circle_radius=3)
                )

                # Print debug info
                print(f"🎯 Calibration: {visible_count} visible landmarks, {high_confidence_count} high confidence")

                # Check if enough landmarks are visible (require at least 12 for good tracking)
                if visible_count >= 12:
                    # After successful calibration, set shoulder detector to ready state
                    if not self.calibration_complete:
                        self.shoulder_detector.state = 'arms_at_sides_check'
                        self.calibration_complete = True

                    # Don't flip or add text - let frontend handle display
                    return annotated_frame, {
                        'calibration_status': 'success',
                        'full_body_detected': True,
                        'visible_landmarks': visible_count,
                        'detection_confidence': 0.9,
                        'message': 'Calibration complete! Now lower both arms to your sides.'
                    }
                # else: Not enough landmarks yet, keep original frame

            else:
                # No pose landmarks detected at all
                print("⚠️ No pose detected - ensure full body is visible")

        # Return retrying status if not enough landmarks detected
        return annotated_frame, {
            'error': 'Please stand further back and ensure full body is visible',
            'calibration_status': 'retrying',
            'visible_landmarks': 0,
            'detection_confidence': 0,
            'message': 'Move back from camera - need to see full body'
        }

    def _process_shoulder_abduction_v2(self, frame):
        """Process shoulder abduction using enhanced detector"""
        # Process frame with shoulder detector first (this draws skeleton)
        result = self.shoulder_detector.process_frame(frame)

        # Don't draw text on video - let frontend handle it
        # Just return the frame with skeleton only
        annotated_frame = frame.copy()

        # Map the enhanced detector format to expected format
        measurements = {
            'assessment_state': result.get('state'),
            'instruction': result.get('instruction', ''),
            'warning': result.get('warning', ''),
            'active_angle': result.get('left_angle', 0) if result.get('current_test_arm') == 'left' else result.get('right_angle', 0),
            'active_side': result.get('current_test_arm'),
            'left_angle': result.get('left_angle', 0),
            'right_angle': result.get('right_angle', 0),
            'left_max': result.get('left_max', 0),
            'right_max': result.get('right_max', 0),
            'progress': result.get('progress', 0),
            'hold_time_remaining': result.get('hold_time_remaining', 0),
            'test_results': result.get('test_results'),
            'complete': result.get('complete', False),
            'error': result.get('error', False),
            'final_report': result.get('final_report', None)  # Pass the detailed report
        }

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