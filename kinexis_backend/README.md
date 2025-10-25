# Kinexis Backend - PT Recovery Tracking System

## Overview
Backend service for the Kinexis hackathon project that provides real-time pose detection and angle measurement for physical therapy exercises using MediaPipe.

## Supported Exercises
1. **Shoulder Abduction** - Arm out to side (0-180°)
2. **Knee Flexion** - Knee bending (0-135°)
3. **Shoulder Flexion** - Arm forward/up (0-180°)

## Quick Start

### 1. Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run the Backend
```bash
python app.py
```

The server will start on:
- HTTP: `http://localhost:5000`
- WebSocket: `ws://localhost:5000`

### 3. Test the Backend
```bash
# In a new terminal
python test_client.py
```

## API Documentation

### REST Endpoints

#### Health Check
```
GET /
```
Returns server status and available exercises.

#### Patients

**Create Patient**
```
POST /api/patients
Body: {
  "name": "John Doe",
  "email": "john@example.com",
  "diagnosis": "Post-ACL surgery"
}
```

**Get All Patients**
```
GET /api/patients
```

**Get Patient Details**
```
GET /api/patients/<patient_id>
```

#### Sessions

**Create Session**
```
POST /api/sessions
Body: {
  "patient_id": 1,
  "exercise_type": "shoulder_abduction"
}
Returns: {
  "session_id": 1,
  "session_key": "session_1",
  "status": "created"
}
```

**Stop Session**
```
POST /api/sessions/<session_id>/stop
Returns: {
  "results": {
    "reps_completed": 10,
    "max_angle_achieved": 145.5,
    "average_angle": 120.3,
    "duration": "0:02:30"
  }
}
```

**Get Session Details**
```
GET /api/sessions/<session_id>
```

#### Exercises

**Get Available Exercises**
```
GET /api/exercises
```

### WebSocket Events

#### Client → Server Events

**Connect**
```javascript
socket.connect('http://localhost:5000')
```

**Start Exercise**
```javascript
socket.emit('start_exercise', {
  session_id: 1,
  exercise_type: 'shoulder_abduction'
})
```

**Process Frame**
```javascript
// Convert frame to base64
const imageData = canvas.toDataURL('image/jpeg')

socket.emit('process_frame', {
  session_key: 'session_1',
  exercise_type: 'shoulder_abduction',
  image: imageData
})
```

**Get Session Stats**
```javascript
socket.emit('get_session_stats', {
  session_key: 'session_1'
})
```

#### Server → Client Events

**Connected**
```javascript
socket.on('connected', (data) => {
  console.log(data.message)  // "Connected to Kinexis backend"
})
```

**Frame Processed**
```javascript
socket.on('frame_processed', (data) => {
  // data.processed_image - annotated frame as base64
  // data.measurements - exercise measurements
  console.log('Angle:', data.measurements.active_angle)
  console.log('Reps:', data.measurements.rep_count)
  console.log('Progress:', data.measurements.progress_percentage)
})
```

**Session Stats**
```javascript
socket.on('session_stats', (stats) => {
  console.log('Reps:', stats.reps_completed)
  console.log('Max Angle:', stats.max_angle)
  console.log('Duration:', stats.session_duration)
})
```

## Frontend Integration Example

```html
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
</head>
<body>
  <video id="webcam" autoplay></video>
  <canvas id="canvas" style="display:none;"></canvas>

  <script>
    const socket = io('http://localhost:5000')
    const video = document.getElementById('webcam')
    const canvas = document.getElementById('canvas')
    const ctx = canvas.getContext('2d')

    // Get webcam
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        video.srcObject = stream
      })

    // Connect to backend
    socket.on('connected', (data) => {
      console.log('Connected to backend')

      // Start exercise session
      socket.emit('start_exercise', {
        session_id: 1,
        exercise_type: 'shoulder_abduction'
      })
    })

    // Process frames
    setInterval(() => {
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      ctx.drawImage(video, 0, 0)

      const imageData = canvas.toDataURL('image/jpeg', 0.8)

      socket.emit('process_frame', {
        session_key: 'session_1',
        exercise_type: 'shoulder_abduction',
        image: imageData
      })
    }, 100)  // Send frame every 100ms

    // Handle processed frames
    socket.on('frame_processed', (data) => {
      console.log('Measurements:', data.measurements)
      // Update UI with angle, reps, etc.
    })
  </script>
</body>
</html>
```

## Data Models

### Patient
- `id`: Integer (Primary Key)
- `name`: String
- `email`: String
- `diagnosis`: String
- `created_at`: DateTime

### Session
- `id`: Integer (Primary Key)
- `patient_id`: Integer (Foreign Key)
- `exercise_type`: String
- `start_time`: DateTime
- `end_time`: DateTime
- `reps_completed`: Integer
- `max_angle_achieved`: Float
- `average_angle`: Float
- `measurements`: JSON

## Measurement Response Format

```json
{
  "exercise": "shoulder_abduction",
  "right_angle": 45.5,
  "left_angle": 120.3,
  "active_side": "left",
  "active_angle": 120.3,
  "rep_count": 5,
  "rep_stage": "up",
  "rep_counted": false,
  "target_angle": 150,
  "progress_percentage": 80.2,
  "detection_confidence": 0.95
}
```

## Exercise-Specific Details

### Shoulder Abduction
- **Motion**: Arm out to the side
- **Target ROM**: 150° (post-surgery)
- **Rep Thresholds**: Up >100°, Down <30°

### Knee Flexion
- **Motion**: Bending knee
- **Target ROM**: 120° (post-ACL)
- **Rep Thresholds**: Up >90°, Down <20°

### Shoulder Flexion
- **Motion**: Arm forward and up
- **Target ROM**: 150° (post-surgery)
- **Rep Thresholds**: Up >110°, Down <30°

## Troubleshooting

### Webcam not detected
- Ensure webcam permissions are granted
- Check if another application is using the webcam

### Low detection confidence
- Ensure good lighting
- Full body should be visible in frame
- Wear contrasting colors

### Backend not starting
- Check if port 5000 is available
- Verify all dependencies are installed

## For Frontend Team

### Key Integration Points
1. Use WebSocket for real-time video processing
2. Send frames as base64 JPEG (quality 0.8 recommended)
3. Process frames at 10-30 FPS (every 33-100ms)
4. Store session_key from create session endpoint
5. Display processed_image from backend for visual feedback

### Recommended Frontend Features
- Progress bar showing angle percentage
- Rep counter display
- Visual angle indicator
- Session timer
- Start/Stop controls
- Exercise selection dropdown

## Contact
Backend Team - Kinexis Hackathon 2025