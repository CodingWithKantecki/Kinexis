"""
Test client for Kinexis Backend
Use this to test the pose detection without a full frontend
"""

import cv2
import requests
import json
import base64
import socketio
import time

# Configuration
BACKEND_URL = "http://localhost:5001"
EXERCISE_TYPE = "shoulder_abduction"  # Change to test different exercises

# Initialize SocketIO client
sio = socketio.Client()

# Global variables
session_id = None
session_key = None
current_exercise = EXERCISE_TYPE

def create_test_patient():
    """Create a test patient"""
    response = requests.post(f"{BACKEND_URL}/api/patients", json={
        "name": "Test Patient",
        "email": "test@kinexis.com",
        "diagnosis": "Post-surgery recovery"
    })
    return response.json()

def start_session(patient_id, exercise_type):
    """Start a new exercise session"""
    response = requests.post(f"{BACKEND_URL}/api/sessions", json={
        "patient_id": patient_id,
        "exercise_type": exercise_type
    })
    return response.json()

@sio.on('connected')
def on_connected(data):
    print(f"Connected to backend: {data['message']}")

@sio.on('frame_processed')
def on_frame_processed(data):
    """Handle processed frame from backend"""
    measurements = data['measurements']

    # Print measurements
    print("\n" + "="*50)
    print(f"Exercise: {measurements.get('exercise', 'Unknown')}")

    if 'error' in measurements:
        print(f"Error: {measurements['error']}")
    else:
        if 'active_flexion' in measurements:
            print(f"Flexion Angle: {measurements.get('active_flexion', 0):.1f}°")
        else:
            print(f"Angle: {measurements.get('active_angle', 0):.1f}°")

        print(f"Active Side: {measurements.get('active_side', 'Unknown')}")
        print(f"Reps: {measurements.get('rep_count', 0)}")
        print(f"Progress: {measurements.get('progress_percentage', 0):.0f}%")

def test_with_webcam():
    """Test the backend with live webcam feed"""
    global session_id, session_key

    print("Starting Kinexis Backend Test Client")
    print("="*50)

    # Create test patient
    print("Creating test patient...")
    patient = create_test_patient()
    patient_id = patient['id']
    print(f"Patient created with ID: {patient_id}")

    # Start session
    print(f"Starting {current_exercise} session...")
    session_data = start_session(patient_id, current_exercise)
    session_id = session_data['session_id']
    session_key = session_data['session_key']
    print(f"Session started with ID: {session_id}")

    # Connect to WebSocket
    print("Connecting to WebSocket...")
    sio.connect(BACKEND_URL)

    # Start exercise
    sio.emit('start_exercise', {
        'session_id': session_id,
        'exercise_type': current_exercise
    })

    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("\nPress 'q' to quit, 'r' to reset reps, 's' to get session stats")
    print("="*50)

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process every 3rd frame to reduce load
            frame_count += 1
            if frame_count % 3 == 0:
                # Encode frame to base64
                _, buffer = cv2.imencode('.jpg', frame)
                image_base64 = base64.b64encode(buffer).decode('utf-8')

                # Send frame to backend
                sio.emit('process_frame', {
                    'session_key': session_key,
                    'exercise_type': current_exercise,
                    'image': f'data:image/jpeg;base64,{image_base64}'
                })

            # Display frame
            cv2.imshow('Kinexis Test - Press Q to quit', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("\nResetting rep counter...")
                # You could emit a reset event here
            elif key == ord('s'):
                print("\nGetting session stats...")
                sio.emit('get_session_stats', {'session_key': session_key})

    finally:
        # Stop session
        print("\nStopping session...")
        response = requests.post(f"{BACKEND_URL}/api/sessions/{session_id}/stop")
        if response.status_code == 200:
            results = response.json()['results']
            print("\nSession Results:")
            print(f"- Reps completed: {results['reps_completed']}")
            print(f"- Max angle achieved: {results['max_angle_achieved']:.1f}°")
            print(f"- Average angle: {results.get('average_angle', 0):.1f}°")
            print(f"- Duration: {results['duration']}")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        sio.disconnect()

def test_with_image():
    """Test the backend with a static image"""
    print("Testing with static image...")

    # Create a blank test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_image, "Test Image", (200, 240),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Encode to base64
    _, buffer = cv2.imencode('.jpg', test_image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    # Test endpoint
    response = requests.post(f"{BACKEND_URL}/api/test_detection", json={
        'image': f'data:image/jpeg;base64,{image_base64}',
        'exercise_type': 'shoulder_abduction'
    })

    if response.status_code == 200:
        print("Detection result:", response.json())
    else:
        print("Error:", response.text)

if __name__ == "__main__":
    import sys
    import numpy as np

    # Check if backend is running
    try:
        response = requests.get(BACKEND_URL)
        if response.status_code == 200:
            print("Backend is running!")
            print("Available exercises:", response.json()['exercises'])

            # Run webcam test
            test_with_webcam()
        else:
            print("Backend returned unexpected status:", response.status_code)
    except requests.exceptions.ConnectionError:
        print("Error: Backend is not running!")
        print(f"Please start the backend first: python app.py")
        sys.exit(1)