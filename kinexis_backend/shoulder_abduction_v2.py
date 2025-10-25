"""
Enhanced Shoulder Abduction ROM Test
Implements proper arms-at-sides confirmation, hold timers, and evaluation reports
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Dict, Optional
from datetime import datetime, timedelta
import time

class ShoulderAbductionV2:
    def __init__(self):
        """Initialize the enhanced shoulder abduction detector"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize pose model with optimal settings for accuracy and speed
        # Model complexity: 0=Lite, 1=Full, 2=Heavy
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Full model - best balance of speed and accuracy
            smooth_landmarks=True,  # Enable smoothing for stable tracking
            enable_segmentation=False,  # Disable for better performance
            smooth_segmentation=False,
            min_detection_confidence=0.4,  # Balanced confidence threshold
            min_tracking_confidence=0.4    # Balanced tracking threshold
        )

        # State machine states
        self.state = 'waiting_for_calibration'

        # Arms at sides confirmation
        self.arms_at_sides_frames = 0
        self.arms_at_sides_required = 60  # 2 seconds at 30fps

        # Current test arm
        self.current_test_arm = None

        # Angle tracking
        self.left_current_angle = 0
        self.right_current_angle = 0
        self.left_max_angle = 0
        self.right_max_angle = 0

        # Hold timer for peak detection
        self.peak_start_time = None
        self.peak_angle = 0
        self.stabilization_time = None
        self.hold_duration_required = 5.0  # 5 seconds
        self.angle_threshold_for_peak = 15  # degrees - angle can drop this much from peak
        self.stabilization_time = None  # Time when angle stabilized near peak
        self.stabilization_delay = 1.0  # Wait 1 second after reaching peak before starting countdown
        self.min_angle_to_start_tracking = 15  # Don't track max or trigger holds until angle > 15°

        # Test results
        self.test_results = {
            'left': {
                'max_angle': 0,
                'hold_duration': 0,
                'completed': False,
                'timestamp': None
            },
            'right': {
                'max_angle': 0,
                'hold_duration': 0,
                'completed': False,
                'timestamp': None
            }
        }

        # Visual feedback
        self.instruction_text = "Waiting for calibration..."
        self.warning_text = ""

    def reset(self):
        """Reset for new assessment"""
        self.state = 'arms_at_sides_check'
        self.arms_at_sides_frames = 0
        self.current_test_arm = None

        self.left_current_angle = 0
        self.right_current_angle = 0
        self.left_max_angle = 0
        self.right_max_angle = 0

        self.peak_start_time = None
        self.peak_angle = 0
        self.stabilization_time = None

        self.test_results = {
            'left': {
                'max_angle': 0,
                'hold_duration': 0,
                'completed': False,
                'timestamp': None
            },
            'right': {
                'max_angle': 0,
                'hold_duration': 0,
                'completed': False,
                'timestamp': None
            }
        }

        self.instruction_text = "Please lower both arms to your sides"
        self.warning_text = ""

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def check_arms_at_sides(self, left_angle, right_angle):
        """Check if both arms are at sides"""
        threshold = 25  # degrees - allow some tolerance
        return left_angle < threshold and right_angle < threshold

    def draw_simplified_skeleton(self, image, landmarks, color):
        """Draw simplified skeleton - only essential joints for cleaner visualization"""
        height, width = image.shape[:2]

        # Define simplified connections - only arms and shoulders for shoulder abduction
        connections = [
            # Core body
            (11, 12),  # shoulders connection

            # Left arm
            (11, 13),  # left shoulder to elbow
            (13, 15),  # left elbow to wrist

            # Right arm
            (12, 14),  # right shoulder to elbow
            (14, 16),  # right elbow to wrist

            # Hip reference (optional, for stability)
            (23, 24),  # hips
        ]

        # Draw connections with thinner lines
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]

                # Check visibility
                if start.visibility > 0.5 and end.visibility > 0.5:
                    start_point = (int(start.x * width), int(start.y * height))
                    end_point = (int(end.x * width), int(end.y * height))
                    cv2.line(image, start_point, end_point, color, 2)  # Thinner line

        # Draw only essential key points with smaller circles
        key_points = [11, 12, 13, 14, 15, 16, 23, 24]  # Arms, shoulders, and hips
        for idx in key_points:
            if idx < len(landmarks):
                landmark = landmarks[idx]
                if landmark.visibility > 0.5:
                    cx = int(landmark.x * width)
                    cy = int(landmark.y * height)
                    cv2.circle(image, (cx, cy), 4, color, -1)  # Smaller filled circle
                    cv2.circle(image, (cx, cy), 5, (255, 255, 255), 1)  # Thin white outline

    def process_frame(self, image):
        """Process a frame and return assessment data"""
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        # Default response
        response = {
            'state': self.state,
            'instruction': self.instruction_text,
            'warning': self.warning_text,
            'left_angle': 0,
            'right_angle': 0,
            'left_max': self.left_max_angle,
            'right_max': self.right_max_angle,
            'current_test_arm': self.current_test_arm,
            'progress': 0,
            'hold_time_remaining': 0,
            'test_results': self.test_results,
            'error': False
        }

        if not results.pose_landmarks:
            response['warning'] = 'Please ensure your full body is visible'
            response['error'] = True
            return response

        landmarks = results.pose_landmarks.landmark

        # Calculate shoulder abduction angles
        # Left side
        left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y]
        left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]

        # Right side
        right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y]
        right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y]

        # Calculate angles
        self.left_current_angle = self.calculate_angle(left_hip, left_shoulder, left_elbow)
        self.right_current_angle = self.calculate_angle(right_hip, right_shoulder, right_elbow)

        response['left_angle'] = self.left_current_angle
        response['right_angle'] = self.right_current_angle

        # State machine logic
        if self.state == 'waiting_for_calibration':
            self.instruction_text = "Waiting for calibration..."

        elif self.state == 'arms_at_sides_check':
            # Check if arms are at sides
            if self.check_arms_at_sides(self.left_current_angle, self.right_current_angle):
                self.arms_at_sides_frames += 1

                if self.arms_at_sides_frames >= self.arms_at_sides_required:
                    # Move to left arm test
                    self.state = 'test_left_arm'
                    self.current_test_arm = 'left'
                    self.instruction_text = "Great! Now slowly raise your LEFT arm out to the side as high as you can"
                    self.warning_text = ""
                    self.arms_at_sides_frames = 0
                else:
                    # Don't show progress percentage in instruction - just a simple message
                    self.instruction_text = "Keep both arms at your sides"
                    self.warning_text = ""
                    # Still track progress internally
                    progress = (self.arms_at_sides_frames / self.arms_at_sides_required) * 100
                    response['arms_position_progress'] = progress
            else:
                self.arms_at_sides_frames = 0
                self.instruction_text = "Please lower BOTH arms completely to your sides"
                self.warning_text = f"Left: {self.left_current_angle:.0f}° | Right: {self.right_current_angle:.0f}° (need < 25°)"

        elif self.state == 'test_left_arm':
            self.current_test_arm = 'left'

            # Only start tracking max angles once we exceed the minimum threshold
            if self.left_current_angle > self.min_angle_to_start_tracking:
                # Update max angle if new peak reached (and we're above minimum threshold)
                if self.left_current_angle > self.left_max_angle:
                    self.left_max_angle = self.left_current_angle
                    self.peak_angle = self.left_current_angle
                    # Start stabilization period instead of countdown
                    self.stabilization_time = time.time()
                    self.peak_start_time = None  # Reset countdown timer
                    self.instruction_text = f"Great! New max: {self.left_max_angle:.0f}°"
                    response['new_max_reached'] = True
                    response['hold_time_remaining'] = 0

                # Check if we're in stabilization period or holding near peak
                elif (self.stabilization_time or self.peak_start_time) and \
                   self.left_current_angle >= (self.peak_angle - self.angle_threshold_for_peak):

                    if self.stabilization_time and not self.peak_start_time:
                        # Still in stabilization period
                        stabilization_elapsed = time.time() - self.stabilization_time

                        if stabilization_elapsed >= self.stabilization_delay:
                            # Start the actual countdown
                            self.peak_start_time = time.time()
                            self.instruction_text = f"Hold steady near {self.left_max_angle:.0f}°"
                        else:
                            # Still stabilizing
                            self.instruction_text = f"Good! Hold near {self.left_max_angle:.0f}°"
                            response['hold_time_remaining'] = 0

                    if self.peak_start_time:
                        # In actual countdown phase
                        hold_duration = time.time() - self.peak_start_time
                        remaining = self.hold_duration_required - hold_duration

                        if remaining <= 0:
                            # Completed left arm test
                            self.test_results['left'] = {
                                'max_angle': self.left_max_angle,
                                'hold_duration': hold_duration,
                                'completed': True,
                                'timestamp': datetime.now().isoformat()
                            }

                            # Move to right arm
                            self.state = 'transition_to_right'
                            self.instruction_text = "Excellent! Lower your left arm and prepare to test your right arm"
                            self.peak_start_time = None
                            self.peak_angle = 0
                            self.stabilization_time = None
                            response['hold_complete'] = True
                        else:
                            # Don't show countdown in instruction - the middle timer handles it
                            self.instruction_text = f"Hold near your max angle ({self.left_max_angle:.0f}°)"
                            response['hold_time_remaining'] = remaining
                            response['holding_at_peak'] = True
                else:
                    # Not at peak - lost the hold
                    if self.peak_start_time or self.stabilization_time:
                        self.instruction_text = f"Try to get back near {self.peak_angle:.0f}° (within {self.angle_threshold_for_peak}°)"
                    else:
                        if self.left_max_angle >= 150:
                            self.instruction_text = f"Great range ({self.left_max_angle:.0f}°)! Raise your arm near your max"
                        else:
                            self.instruction_text = f"Raise LEFT arm higher - Best: {self.left_max_angle:.0f}°"

                    self.peak_start_time = None
                    self.stabilization_time = None
                    response['hold_time_remaining'] = 0
            else:
                # Below minimum tracking threshold - just show basic instruction
                self.instruction_text = "Slowly raise your LEFT arm out to the side as high as you can"
                response['hold_time_remaining'] = 0

            response['progress'] = min((self.left_current_angle / 150) * 100, 100)

        elif self.state == 'transition_to_right':
            # Wait for left arm to lower
            if self.left_current_angle < 30:
                self.state = 'test_right_arm'
                self.current_test_arm = 'right'
                self.instruction_text = "Now slowly raise your RIGHT arm out to the side as high as you can"
            else:
                self.instruction_text = "Lower your left arm completely before starting right arm test"

        elif self.state == 'test_right_arm':
            self.current_test_arm = 'right'

            # Only start tracking max angles once we exceed the minimum threshold
            if self.right_current_angle > self.min_angle_to_start_tracking:
                # Update max angle if new peak reached (and we're above minimum threshold)
                if self.right_current_angle > self.right_max_angle:
                    self.right_max_angle = self.right_current_angle
                    self.peak_angle = self.right_current_angle
                    # Start stabilization period instead of countdown
                    self.stabilization_time = time.time()
                    self.peak_start_time = None  # Reset countdown timer
                    self.instruction_text = f"Great! New max: {self.right_max_angle:.0f}°"
                    response['new_max_reached'] = True
                    response['hold_time_remaining'] = 0

                # Check if we're in stabilization period or holding near peak
                elif (self.stabilization_time or self.peak_start_time) and \
                   self.right_current_angle >= (self.peak_angle - self.angle_threshold_for_peak):

                    if self.stabilization_time and not self.peak_start_time:
                        # Still in stabilization period
                        stabilization_elapsed = time.time() - self.stabilization_time

                        if stabilization_elapsed >= self.stabilization_delay:
                            # Start the actual countdown
                            self.peak_start_time = time.time()
                            self.instruction_text = f"Hold steady near {self.right_max_angle:.0f}°"
                        else:
                            # Still stabilizing
                            self.instruction_text = f"Good! Hold near {self.right_max_angle:.0f}°"
                            response['hold_time_remaining'] = 0

                    if self.peak_start_time:
                        # In actual countdown phase
                        hold_duration = time.time() - self.peak_start_time
                        remaining = self.hold_duration_required - hold_duration

                        if remaining <= 0:
                            # Completed right arm test
                            self.test_results['right'] = {
                                'max_angle': self.right_max_angle,
                                'hold_duration': hold_duration,
                                'completed': True,
                                'timestamp': datetime.now().isoformat()
                            }

                            # Test complete
                            self.state = 'complete'
                            self.instruction_text = "Assessment Complete! Lower your arm."
                            self.peak_start_time = None
                            self.peak_angle = 0
                            self.stabilization_time = None
                            response['hold_complete'] = True
                        else:
                            # Don't show countdown in instruction - the middle timer handles it
                            self.instruction_text = f"Hold near your max angle ({self.right_max_angle:.0f}°)"
                            response['hold_time_remaining'] = remaining
                            response['holding_at_peak'] = True
                else:
                    # Not at peak - lost the hold
                    if self.peak_start_time or self.stabilization_time:
                        self.instruction_text = f"Try to get back near {self.peak_angle:.0f}° (within {self.angle_threshold_for_peak}°)"
                    else:
                        if self.right_max_angle >= 150:
                            self.instruction_text = f"Great range ({self.right_max_angle:.0f}°)! Raise your arm near your max"
                        else:
                            self.instruction_text = f"Raise RIGHT arm higher - Best: {self.right_max_angle:.0f}°"

                    self.peak_start_time = None
                    self.stabilization_time = None
                    response['hold_time_remaining'] = 0
            else:
                # Below minimum tracking threshold - just show basic instruction
                self.instruction_text = "Slowly raise your RIGHT arm out to the side as high as you can"
                response['hold_time_remaining'] = 0

            response['progress'] = min((self.right_current_angle / 150) * 100, 100)

        elif self.state == 'complete':
            self.instruction_text = "Assessment Complete! Great job!"
            self.warning_text = ""  # Don't use warning for final report
            response['complete'] = True
            response['final_report'] = self.generate_detailed_report()  # Add detailed report

        # Update response
        response['state'] = self.state
        response['instruction'] = self.instruction_text
        response['warning'] = self.warning_text
        response['left_max'] = self.left_max_angle
        response['right_max'] = self.right_max_angle
        response['test_results'] = self.test_results

        # Draw skeleton on image with color coding
        if results.pose_landmarks:
            # Choose color based on state
            if self.state == 'arms_at_sides_check':
                landmark_color = (255, 255, 0)  # Yellow for positioning
                connection_color = (255, 200, 0)
            elif self.current_test_arm == 'left':
                landmark_color = (0, 255, 0)  # Green for left
                connection_color = (0, 200, 0)
            elif self.current_test_arm == 'right':
                landmark_color = (255, 0, 255)  # Magenta for right
                connection_color = (200, 0, 200)
            else:
                landmark_color = (245, 117, 66)  # Default orange
                connection_color = (200, 100, 50)

            # Draw simplified skeleton - only essential joints for shoulder abduction
            self.draw_simplified_skeleton(image, results.pose_landmarks.landmark, landmark_color)

        return response

    def generate_evaluation_summary(self):
        """Generate final evaluation summary"""
        left_result = self.test_results['left']
        right_result = self.test_results['right']

        # Determine ROM categories
        def get_rom_category(angle):
            if angle >= 150:
                return "Normal"
            elif angle >= 120:
                return "Mild limitation"
            elif angle >= 90:
                return "Moderate limitation"
            elif angle >= 60:
                return "Significant limitation"
            else:
                return "Severe limitation"

        left_category = get_rom_category(left_result['max_angle'])
        right_category = get_rom_category(right_result['max_angle'])

        summary = f"Left: {left_result['max_angle']:.0f}° ({left_category}) | "
        summary += f"Right: {right_result['max_angle']:.0f}° ({right_category})"

        return summary

    def generate_detailed_report(self):
        """Generate comprehensive assessment report with all statistics"""
        left_result = self.test_results['left']
        right_result = self.test_results['right']

        # Determine ROM categories
        def get_rom_category(angle):
            if angle >= 150:
                return {"category": "Normal", "color": "#10b981", "icon": "✓"}
            elif angle >= 120:
                return {"category": "Mild limitation", "color": "#eab308", "icon": "⚠"}
            elif angle >= 90:
                return {"category": "Moderate limitation", "color": "#f97316", "icon": "⚠"}
            elif angle >= 60:
                return {"category": "Significant limitation", "color": "#ef4444", "icon": "✗"}
            else:
                return {"category": "Severe limitation", "color": "#dc2626", "icon": "✗"}

        left_category = get_rom_category(left_result['max_angle'])
        right_category = get_rom_category(right_result['max_angle'])

        # Calculate symmetry
        angle_difference = abs(left_result['max_angle'] - right_result['max_angle'])
        symmetry_percentage = 100 - min(angle_difference * 2, 100)  # 2% deduction per degree difference

        # Determine overall status
        avg_angle = (left_result['max_angle'] + right_result['max_angle']) / 2
        if avg_angle >= 150 and angle_difference <= 10:
            overall_status = "Excellent"
            overall_color = "#10b981"
        elif avg_angle >= 120:
            overall_status = "Good"
            overall_color = "#eab308"
        elif avg_angle >= 90:
            overall_status = "Fair"
            overall_color = "#f97316"
        else:
            overall_status = "Needs Improvement"
            overall_color = "#ef4444"

        report = {
            "timestamp": datetime.now().isoformat(),
            "exercise": "Shoulder Abduction ROM Assessment",
            "left_arm": {
                "max_angle": left_result['max_angle'],
                "hold_duration": left_result.get('hold_duration', 0),
                "completed": left_result['completed'],
                "category": left_category['category'],
                "color": left_category['color'],
                "icon": left_category['icon'],
                "target_achieved": left_result['max_angle'] >= 150
            },
            "right_arm": {
                "max_angle": right_result['max_angle'],
                "hold_duration": right_result.get('hold_duration', 0),
                "completed": right_result['completed'],
                "category": right_category['category'],
                "color": right_category['color'],
                "icon": right_category['icon'],
                "target_achieved": right_result['max_angle'] >= 150
            },
            "symmetry": {
                "difference": angle_difference,
                "percentage": symmetry_percentage,
                "assessment": "Symmetric" if angle_difference <= 10 else "Asymmetric"
            },
            "overall": {
                "status": overall_status,
                "color": overall_color,
                "average_angle": avg_angle,
                "normal_range": "150-180°",
                "both_completed": left_result['completed'] and right_result['completed']
            },
            "recommendations": self.generate_recommendations(avg_angle, angle_difference)
        }

        return report

    def generate_recommendations(self, avg_angle, angle_difference):
        """Generate personalized recommendations based on assessment results"""
        recommendations = []

        # Range-based recommendations
        if avg_angle < 90:
            recommendations.append("Consult with a physical therapist for personalized exercises")
            recommendations.append("Start with gentle range-of-motion exercises daily")
        elif avg_angle < 120:
            recommendations.append("Continue with progressive stretching exercises")
            recommendations.append("Focus on gradual improvement over time")
        elif avg_angle < 150:
            recommendations.append("Good progress! Continue current exercise routine")
            recommendations.append("Add resistance training when comfortable")
        else:
            recommendations.append("Excellent range of motion - maintain with regular exercise")

        # Symmetry recommendations
        if angle_difference > 15:
            recommendations.append(f"Focus on strengthening the weaker side to improve symmetry")

        return recommendations