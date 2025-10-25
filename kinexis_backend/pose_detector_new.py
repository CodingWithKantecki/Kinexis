"""
Simplified Shoulder Abduction Detection
Clean implementation focusing on arms-at-sides check and sequential arm testing
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Dict, Optional
from datetime import datetime

class ShoulderAbductionDetector:
    def __init__(self):
        """Initialize the simplified detector"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize pose model
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )

        # State machine: arms_check -> left_arm -> right_arm -> complete
        self.state = 'arms_check'
        self.arms_down_counter = 0
        self.current_arm_testing = None

        # Peak detection
        self.peak_hold_frames = 0
        self.peak_angle = 0
        self.frames_since_peak = 0

        # Data storage
        self.left_max = 0
        self.right_max = 0

    def reset(self):
        """Reset for new assessment"""
        self.state = 'arms_check'
        self.arms_down_counter = 0
        self.current_arm_testing = None
        self.peak_hold_frames = 0
        self.peak_angle = 0
        self.frames_since_peak = 0
        self.left_max = 0
        self.right_max = 0

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def process_frame(self, image):
        """Process a frame and return assessment data"""
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return {
                'state': self.state,
                'instruction': 'Please ensure your full body is visible',
                'error': True
            }

        landmarks = results.pose_landmarks.landmark

        # Get key landmarks
        left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y]
        left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]

        right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y]
        right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y]

        # Calculate angles
        left_angle = self.calculate_angle(left_hip, left_shoulder, left_elbow)
        right_angle = self.calculate_angle(right_hip, right_shoulder, right_elbow)

        # State machine logic
        if self.state == 'arms_check':
            # Check if both arms are down (< 20 degrees)
            if left_angle < 20 and right_angle < 20:
                self.arms_down_counter += 1
                if self.arms_down_counter >= 30:  # 1 second at 30fps
                    self.state = 'left_arm'
                    self.arms_down_counter = 0
                    return {
                        'state': self.state,
                        'instruction': 'Great! Now raise your LEFT arm out to the side',
                        'left_angle': left_angle,
                        'right_angle': right_angle,
                        'arms_ready': True
                    }
                else:
                    return {
                        'state': self.state,
                        'instruction': 'Hold both arms at your sides...',
                        'left_angle': left_angle,
                        'right_angle': right_angle,
                        'arms_ready': False
                    }
            else:
                self.arms_down_counter = 0
                return {
                    'state': self.state,
                    'instruction': 'Please lower BOTH arms to your sides',
                    'left_angle': left_angle,
                    'right_angle': right_angle,
                    'arms_ready': False
                }

        elif self.state == 'left_arm':
            # Testing left arm
            self.current_arm_testing = 'left'

            # Update max angle
            if left_angle > self.left_max:
                self.left_max = left_angle
                self.peak_angle = left_angle
                self.frames_since_peak = 0
            else:
                self.frames_since_peak += 1

            # Check if we've held the peak for 5 seconds (150 frames at 30fps)
            if self.frames_since_peak >= 150 or self.left_max >= 150:
                self.state = 'right_arm'
                self.frames_since_peak = 0
                self.peak_angle = 0
                return {
                    'state': self.state,
                    'instruction': 'Good! Lower your left arm. Now raise your RIGHT arm',
                    'left_angle': left_angle,
                    'right_angle': right_angle,
                    'left_max': self.left_max,
                    'test_arm': 'left',
                    'progress': (left_angle / 150) * 100
                }

            return {
                'state': self.state,
                'instruction': f'Raise LEFT arm higher - Current: {int(left_angle)}° / Target: 150°',
                'left_angle': left_angle,
                'right_angle': right_angle,
                'left_max': self.left_max,
                'test_arm': 'left',
                'progress': (left_angle / 150) * 100
            }

        elif self.state == 'right_arm':
            # Testing right arm
            self.current_arm_testing = 'right'

            # Update max angle
            if right_angle > self.right_max:
                self.right_max = right_angle
                self.peak_angle = right_angle
                self.frames_since_peak = 0
            else:
                self.frames_since_peak += 1

            # Check if we've held the peak for 5 seconds
            if self.frames_since_peak >= 150 or self.right_max >= 150:
                self.state = 'complete'
                return {
                    'state': self.state,
                    'instruction': 'Assessment Complete!',
                    'left_angle': left_angle,
                    'right_angle': right_angle,
                    'left_max': self.left_max,
                    'right_max': self.right_max,
                    'test_arm': 'right',
                    'progress': (right_angle / 150) * 100,
                    'complete': True
                }

            return {
                'state': self.state,
                'instruction': f'Raise RIGHT arm higher - Current: {int(right_angle)}° / Target: 150°',
                'left_angle': left_angle,
                'right_angle': right_angle,
                'right_max': self.right_max,
                'test_arm': 'right',
                'progress': (right_angle / 150) * 100
            }

        elif self.state == 'complete':
            return {
                'state': self.state,
                'instruction': 'Test Complete',
                'left_max': self.left_max,
                'right_max': self.right_max,
                'complete': True
            }

        # Draw skeleton
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        return {
            'state': self.state,
            'left_angle': left_angle,
            'right_angle': right_angle
        }