"""
Kinexis Backend - Main Flask Application
Real-time PT exercise tracking with MediaPipe
"""

import os
import json
import base64
import numpy as np
import cv2
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy

# Try to import the V2 pose detector with enhanced shoulder abduction
try:
    from pose_detector_v2 import PoseDetectorV2 as PoseDetector
    print("✨ Using enhanced PoseDetectorV2 with improved shoulder abduction ROM test")
except ImportError:
    try:
        from pose_detector_integrated import PoseDetector
        print("Using integrated MediaPipe pose detector with new shoulder abduction")
    except ImportError:
        try:
            from pose_detector import PoseDetector
            print("Using original MediaPipe pose detector")
        except ImportError:
            print("MediaPipe not available, using mock pose detector")
            from pose_detector_mock import PoseDetector

import io

# Initialize Flask app with static folder
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['SECRET_KEY'] = 'kinexis-hackathon-2025'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///kinexis.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
db = SQLAlchemy(app)

# Initialize pose detector
pose_detector = PoseDetector()

# Active sessions storage (in production, use Redis)
active_sessions = {}

# Database Models
class Patient(db.Model):
    """Patient information model"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    date_of_birth = db.Column(db.Date)
    diagnosis = db.Column(db.String(200))
    surgery_date = db.Column(db.Date)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    sessions = db.relationship('Session', backref='patient', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'diagnosis': self.diagnosis,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Session(db.Model):
    """Exercise session model"""
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    exercise_type = db.Column(db.String(50), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    reps_completed = db.Column(db.Integer, default=0)
    max_angle_achieved = db.Column(db.Float, default=0.0)
    average_angle = db.Column(db.Float)
    measurements = db.Column(db.JSON)
    notes = db.Column(db.Text)

    def to_dict(self):
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'exercise_type': self.exercise_type,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'reps_completed': self.reps_completed,
            'max_angle_achieved': self.max_angle_achieved,
            'average_angle': self.average_angle
        }

# Initialize database
with app.app_context():
    db.create_all()

# REST API Endpoints

@app.route('/')
def index():
    """Serve the frontend homepage"""
    try:
        # Try to serve the frontend homepage if it exists
        return send_file('kinexis/frontend/homepage.html')
    except:
        # Fall back to API response if frontend doesn't exist
        return jsonify({
            'status': 'running',
            'service': 'Kinexis Backend',
            'version': '1.0.0',
            'exercises': ['shoulder_abduction', 'knee_flexion']
        })

@app.route('/demo')
def demo():
    """Demo interface for testing"""
    return render_template('index.html')

@app.route('/kinexis/frontend/<path:filename>')
def serve_frontend(filename):
    """Serve frontend files"""
    return send_file(f'kinexis/frontend/{filename}')

@app.route('/homepage.html')
def homepage():
    """Serve homepage"""
    return send_file('kinexis/frontend/homepage.html')

@app.route('/instructions.html')
def instructions():
    """Serve instructions page"""
    return send_file('kinexis/frontend/instructions.html')

@app.route('/goals.html')
def goals():
    """Serve goals page"""
    return send_file('kinexis/frontend/goals.html')

@app.route('/styles.css')
def styles():
    """Serve styles"""
    return send_file('kinexis/frontend/styles.css')

@app.route('/images/<path:filename>')
def serve_images(filename):
    """Serve image files"""
    return send_file(f'kinexis/frontend/images/{filename}')

@app.route('/config.js')
def serve_config():
    """Serve config.js file"""
    return send_file('kinexis/frontend/config.js')

@app.route('/api/patients', methods=['GET', 'POST'])
def handle_patients():
    """Get all patients or create a new patient"""
    if request.method == 'POST':
        data = request.json
        patient = Patient(
            name=data['name'],
            email=data.get('email'),
            diagnosis=data.get('diagnosis', '')
        )
        db.session.add(patient)
        db.session.commit()
        return jsonify(patient.to_dict()), 201

    patients = Patient.query.all()
    return jsonify([p.to_dict() for p in patients])

@app.route('/api/patients/<int:patient_id>')
def get_patient(patient_id):
    """Get specific patient details"""
    patient = Patient.query.get_or_404(patient_id)
    patient_data = patient.to_dict()
    patient_data['session_count'] = len(patient.sessions)
    patient_data['recent_sessions'] = [s.to_dict() for s in patient.sessions[-5:]]
    return jsonify(patient_data)

@app.route('/api/sessions', methods=['POST'])
def create_session():
    """Create a new exercise session"""
    data = request.json
    session = Session(
        patient_id=data['patient_id'],
        exercise_type=data['exercise_type'],
        measurements=[]
    )
    db.session.add(session)
    db.session.commit()

    # Store in active sessions
    session_key = f"session_{session.id}"
    active_sessions[session_key] = {
        'id': session.id,
        'patient_id': session.patient_id,
        'exercise_type': session.exercise_type,
        'start_time': datetime.utcnow().isoformat(),
        'measurements': [],
        'max_angle': 0,
        'reps': 0
    }

    # Reset pose detector counters for this exercise
    pose_detector.reset_exercise_counters(data['exercise_type'])

    return jsonify({
        'session_id': session.id,
        'session_key': session_key,
        'status': 'created'
    }), 201

@app.route('/api/sessions/<int:session_id>/stop', methods=['POST'])
def stop_session(session_id):
    """Stop an active session and save results"""
    session = Session.query.get_or_404(session_id)
    session_key = f"session_{session_id}"

    if session_key in active_sessions:
        session_data = active_sessions[session_key]

        # Update session with collected data
        session.end_time = datetime.utcnow()
        session.reps_completed = session_data['reps']
        session.max_angle_achieved = session_data['max_angle']

        # Calculate average angle from measurements
        if session_data['measurements']:
            angles = [m['angle'] for m in session_data['measurements'] if 'angle' in m]
            session.average_angle = np.mean(angles) if angles else 0

        session.measurements = session_data['measurements'][-100:]  # Keep last 100 measurements

        db.session.commit()

        # Remove from active sessions
        del active_sessions[session_key]

        return jsonify({
            'session_id': session_id,
            'status': 'stopped',
            'results': {
                'reps_completed': session.reps_completed,
                'max_angle_achieved': session.max_angle_achieved,
                'average_angle': session.average_angle,
                'duration': str(session.end_time - session.start_time)
            }
        })

    return jsonify({'error': 'Session not active'}), 400

@app.route('/api/sessions/<int:session_id>')
def get_session(session_id):
    """Get session details"""
    session = Session.query.get_or_404(session_id)
    return jsonify(session.to_dict())

@app.route('/api/test_mediapipe')
def test_mediapipe():
    """Test if MediaPipe is working"""
    import mediapipe as mp
    try:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        # Create a simple test image
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        test_img[:] = (255, 255, 255)  # White background
        result = pose.process(test_img)
        pose.close()
        return jsonify({
            'mediapipe_status': 'working',
            'test_result': 'success',
            'pose_detected': result.pose_landmarks is not None
        })
    except Exception as e:
        return jsonify({
            'mediapipe_status': 'error',
            'error': str(e)
        })

@app.route('/api/exercises')
def get_exercises():
    """Get available exercises and their descriptions"""
    exercises = [
        {
            'id': 'shoulder_abduction',
            'name': 'Shoulder Abduction',
            'description': 'Raise your arm out to the side',
            'target_angle': 150,
            'instructions': 'Stand straight and slowly raise your arm out to the side, keeping it straight. Hold briefly at the top, then lower slowly.',
            'normal_range': '0-180°',
            'post_surgery_goal': '80° → 150°'
        },
        {
            'id': 'knee_flexion',
            'name': 'Knee Flexion',
            'description': 'Bend your knee bringing heel toward buttocks',
            'target_angle': 120,
            'instructions': 'Stand holding a chair for support. Slowly bend your knee, bringing your heel toward your buttocks. Hold briefly, then lower slowly.',
            'normal_range': '0-135°',
            'post_surgery_goal': '60° → 120°'
        }
    ]
    return jsonify(exercises)

# WebSocket Events for Real-time Video Processing

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to Kinexis backend'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")

@socketio.on('start_exercise')
def handle_start_exercise(data):
    """Initialize exercise session via WebSocket"""
    session_id = data.get('session_id')
    exercise_type = data.get('exercise_type')

    # Reset pose detector for new exercise
    pose_detector.reset_exercise_counters(exercise_type)

    # Reset assessment for shoulder abduction
    if exercise_type == 'shoulder_abduction':
        pose_detector.reset_assessment()
        print(f"🔄 Reset PT assessment for {exercise_type}")

    emit('exercise_started', {
        'session_id': session_id,
        'exercise_type': exercise_type,
        'status': 'ready'
    })

@socketio.on('process_frame')
def handle_process_frame(data):
    """Process video frame for pose detection"""
    try:
        session_key = data.get('session_key')
        exercise_type = data.get('exercise_type')
        image_data = data.get('image')
        is_calibration = data.get('is_calibration', False)

        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)

        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Check if frame is valid
        if frame is None or frame.size == 0:
            print("Warning: Invalid frame received")
            emit('error', {'message': 'Invalid frame'})
            return

        # Debug: Check frame properties
        if is_calibration:
            print(f"🎯 CALIBRATION FRAME RECEIVED: shape={frame.shape}, dtype={frame.dtype}")
            print(f"   Frame stats: min={np.min(frame)}, max={np.max(frame)}, mean={np.mean(frame):.1f}")
        # Remove regular frame logging to reduce noise

        # Reset assessment on first shoulder abduction frame (ensure initialization)
        if exercise_type == 'shoulder_abduction' and not is_calibration:
            if not hasattr(pose_detector, '_assessment_initialized'):
                pose_detector.reset_assessment()
                pose_detector._assessment_initialized = True
                print("🎯 Initialized PT assessment for shoulder abduction")

        # Process frame with pose detector (no flip needed - handled in CSS)
        annotated_frame, measurements = pose_detector.process_frame(frame, exercise_type, is_calibration=is_calibration)

        # Log calibration results
        if is_calibration:
            print(f"📊 Calibration result: {measurements}")

        # Debug: Log only calibration results, not every frame
        if is_calibration and 'calibration_status' in measurements:
            print(f"📊 Calibration result: status={measurements.get('calibration_status')}, landmarks={measurements.get('visible_landmarks')}")

        # Update session data if active
        if session_key in active_sessions:
            session_data = active_sessions[session_key]

            # Update measurements
            if 'error' not in measurements:
                angle = measurements.get('active_angle', measurements.get('active_flexion', 0))

                # Store measurement
                session_data['measurements'].append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'angle': angle,
                    'side': measurements.get('active_side', 'unknown')
                })

                # Update max angle
                if angle > session_data['max_angle']:
                    session_data['max_angle'] = angle

                # Update reps
                session_data['reps'] = measurements.get('rep_count', 0)

        # Resize annotated frame while preserving aspect ratio
        height, width = annotated_frame.shape[:2]
        max_dimension = 1280  # Increased resolution for better quality

        if width > max_dimension or height > max_dimension:
            if width > height:
                scale = max_dimension / width
            else:
                scale = max_dimension / height
            new_width = int(width * scale)
            new_height = int(height * scale)
            annotated_frame = cv2.resize(annotated_frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Convert annotated frame back to base64 with maximum quality
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        processed_image = base64.b64encode(buffer).decode('utf-8')

        # Send results back to client
        emit('frame_processed', {
            'processed_image': f'data:image/jpeg;base64,{processed_image}',
            'measurements': measurements,
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        emit('error', {'message': f'Error processing frame: {str(e)}'})

@socketio.on('get_session_stats')
def handle_get_session_stats(data):
    """Get real-time session statistics"""
    session_key = data.get('session_key')

    if session_key in active_sessions:
        session_data = active_sessions[session_key]

        # Calculate statistics
        measurements = session_data['measurements']
        angles = [m['angle'] for m in measurements if 'angle' in m]

        stats = {
            'reps_completed': session_data['reps'],
            'max_angle': session_data['max_angle'],
            'current_average': np.mean(angles[-10:]) if len(angles) >= 10 else np.mean(angles) if angles else 0,
            'measurement_count': len(measurements),
            'session_duration': (datetime.utcnow() - datetime.fromisoformat(session_data['start_time'])).total_seconds()
        }

        emit('session_stats', stats)
    else:
        emit('error', {'message': 'Session not found'})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5001))

    print("=" * 50)
    print("Kinexis Backend Server")
    print("=" * 50)
    print("Supported exercises:")
    print("- Shoulder Abduction")
    print("- Knee Flexion")
    print("=" * 50)
    print(f"Starting server on http://localhost:{port}")
    print(f"WebSocket available on ws://localhost:{port}")
    print("=" * 50)

    # Use debug mode only in development
    is_production = os.environ.get('FLASK_ENV') == 'production'
    socketio.run(app, debug=not is_production, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)