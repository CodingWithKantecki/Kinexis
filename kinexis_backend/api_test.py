"""
Simple API test for Kinexis Backend
Tests all endpoints without requiring a webcam
"""

import requests
import json
import time

BASE_URL = "http://localhost:5001"

def test_health_check():
    """Test the health check endpoint"""
    print("\n1. Testing Health Check...")
    response = requests.get(f"{BASE_URL}/")
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Server Status: {data['status']}")
        print(f"  Version: {data['version']}")
        print(f"  Available Exercises: {', '.join(data['exercises'])}")
        return True
    else:
        print(f"✗ Health check failed: {response.status_code}")
        return False

def test_exercises():
    """Test getting exercise list"""
    print("\n2. Testing Exercise Endpoints...")
    response = requests.get(f"{BASE_URL}/api/exercises")
    if response.status_code == 200:
        exercises = response.json()
        print(f"✓ Found {len(exercises)} exercises:")
        for ex in exercises:
            print(f"  - {ex['name']}: {ex['description']}")
            print(f"    Target: {ex['target_angle']}°, Goal: {ex['post_surgery_goal']}")
        return True
    else:
        print(f"✗ Failed to get exercises: {response.status_code}")
        return False

def test_patient_crud():
    """Test patient creation and retrieval"""
    print("\n3. Testing Patient Operations...")

    # Create patient
    patient_data = {
        "name": "Test Patient",
        "email": f"test_{int(time.time())}@kinexis.com",
        "diagnosis": "Post-surgery recovery testing"
    }

    response = requests.post(f"{BASE_URL}/api/patients", json=patient_data)
    if response.status_code == 201:
        patient = response.json()
        print(f"✓ Patient created with ID: {patient['id']}")

        # Get patient details
        response = requests.get(f"{BASE_URL}/api/patients/{patient['id']}")
        if response.status_code == 200:
            print(f"✓ Retrieved patient: {response.json()['name']}")
            return patient['id']
        else:
            print(f"✗ Failed to retrieve patient: {response.status_code}")
            return None
    else:
        print(f"✗ Failed to create patient: {response.status_code}")
        return None

def test_session_crud(patient_id):
    """Test session creation and management"""
    print("\n4. Testing Session Operations...")

    if not patient_id:
        print("✗ Skipping session tests (no patient ID)")
        return False

    # Create session
    session_data = {
        "patient_id": patient_id,
        "exercise_type": "shoulder_abduction"
    }

    response = requests.post(f"{BASE_URL}/api/sessions", json=session_data)
    if response.status_code == 201:
        session = response.json()
        session_id = session['session_id']
        print(f"✓ Session created with ID: {session_id}")
        print(f"  Session Key: {session['session_key']}")

        # Get session details
        response = requests.get(f"{BASE_URL}/api/sessions/{session_id}")
        if response.status_code == 200:
            print(f"✓ Retrieved session details")

        # Stop session
        time.sleep(1)  # Simulate some activity
        response = requests.post(f"{BASE_URL}/api/sessions/{session_id}/stop")
        if response.status_code == 200:
            results = response.json()['results']
            print(f"✓ Session stopped successfully")
            print(f"  Duration: {results['duration']}")
            print(f"  Max Angle: {results['max_angle_achieved']}°")
            return True
        else:
            print(f"✗ Failed to stop session: {response.status_code}")
            return False
    else:
        print(f"✗ Failed to create session: {response.status_code}")
        return False

def test_all_patients():
    """Test getting all patients"""
    print("\n5. Testing Get All Patients...")
    response = requests.get(f"{BASE_URL}/api/patients")
    if response.status_code == 200:
        patients = response.json()
        print(f"✓ Found {len(patients)} patients in database")
        for p in patients[:3]:  # Show first 3
            print(f"  - {p['name']} (ID: {p['id']})")
        return True
    else:
        print(f"✗ Failed to get patients: {response.status_code}")
        return False

def run_all_tests():
    """Run all API tests"""
    print("="*50)
    print("Kinexis Backend API Tests")
    print("="*50)

    # Check if backend is running
    try:
        requests.get(BASE_URL, timeout=2)
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Backend is not running!")
        print(f"Please start the backend first: python app.py")
        return

    # Run tests
    tests_passed = 0
    tests_total = 5

    if test_health_check():
        tests_passed += 1

    if test_exercises():
        tests_passed += 1

    patient_id = test_patient_crud()
    if patient_id:
        tests_passed += 1

    if test_session_crud(patient_id):
        tests_passed += 1

    if test_all_patients():
        tests_passed += 1

    # Summary
    print("\n" + "="*50)
    print(f"Test Results: {tests_passed}/{tests_total} passed")

    if tests_passed == tests_total:
        print("✓ All tests passed! Backend is working correctly.")
    else:
        print(f"✗ {tests_total - tests_passed} test(s) failed.")

    print("="*50)

if __name__ == "__main__":
    run_all_tests()