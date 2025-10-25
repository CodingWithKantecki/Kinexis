# Enhanced Shoulder Abduction ROM Test Workflow

## Overview
The enhanced shoulder abduction test now provides a complete, clinically-oriented Range of Motion (ROM) assessment with proper positioning verification, hold timers, and detailed evaluation reports.

## Test Workflow

### 1. Calibration Phase
- **Purpose**: Ensure full body is visible and pose detection is working
- **Success Criteria**: At least 15 body landmarks detected with good confidence
- **Visual Feedback**: Green skeleton overlay when successful

### 2. Arms-at-Sides Confirmation (NEW)
- **Purpose**: Establish baseline position before ROM testing
- **Requirements**:
  - Both arms must be lowered to < 25° from the body
  - Position must be held for 2 seconds (60 frames)
- **Visual Feedback**:
  - Yellow skeleton during positioning
  - Progress percentage shown while holding
  - Clear warning if arms are not lowered enough

### 3. Left Arm ROM Test
- **Test Sequence**:
  1. Patient raises LEFT arm out to the side (shoulder abduction)
  2. System tracks current angle and maximum achieved angle
  3. Progress bar displays on RIGHT side of screen (mirrored for webcam)
  4. Color coding: Orange (0-100°) → Yellow (100-135°) → Green (135°+)

- **Peak Hold Requirement** (NEW):
  - When patient reaches their maximum angle, they must hold for 5 seconds
  - Timer displays prominently on screen
  - Allows up to 5° drop from peak while still counting as "holding"
  - Automatically progresses after successful 5-second hold OR reaching 150°

### 4. Transition Phase
- Patient lowers left arm completely (< 30°)
- System waits for confirmation before starting right arm test

### 5. Right Arm ROM Test
- **Test Sequence**:
  1. Patient raises RIGHT arm out to the side
  2. Progress bar displays on LEFT side of screen (mirrored)
  3. Same hold requirements as left arm
  4. Color coding matches left arm test

### 6. Final Evaluation Report (NEW)
- **Automatic ROM Classification**:
  - **Normal**: ≥ 150° (full range)
  - **Mild Limitation**: 120-149°
  - **Moderate Limitation**: 90-119°
  - **Significant Limitation**: 60-89°
  - **Severe Limitation**: < 60°

- **Report Includes**:
  - Maximum angle achieved for each arm
  - ROM classification for each arm
  - Hold duration successfully completed
  - Timestamp of assessment

## Visual UI Elements

### Progress Bars
- **Location**: Side of screen opposite to the arm being tested (for webcam mirroring)
- **Features**:
  - Vertical bar showing current angle
  - Target line at 150°
  - Color changes based on progress
  - Current angle displayed below bar
  - Maximum angle tracked separately

### Instructions & Feedback
- **Top of Screen**: Current instruction (green text)
- **Warning Area**: Additional guidance or warnings (yellow text)
- **Hold Timer**: Large centered display during hold phase
- **Final Results**: Centered summary with both arms' results

## Key Improvements from Previous Version

1. **Mandatory Arms-at-Sides Check**: Ensures consistent starting position
2. **5-Second Hold Timer**: Validates true maximum range vs. momentary peaks
3. **Clinical ROM Categories**: Provides meaningful assessment beyond just angles
4. **Better Visual Feedback**: Clear progress bars with side-specific positioning
5. **Smooth State Transitions**: Logical flow from calibration → positioning → left → right → results

## Technical Details

### State Machine
```
waiting_for_calibration
    ↓
arms_at_sides_check (2 sec hold)
    ↓
test_left_arm (5 sec peak hold)
    ↓
transition_to_right
    ↓
test_right_arm (5 sec peak hold)
    ↓
complete (show results)
```

### Angle Calculation
- Uses MediaPipe pose landmarks
- Calculates angle between hip-shoulder-elbow points
- Smoothed tracking with threshold tolerance

## Usage Tips

1. **Patient Positioning**:
   - Stand facing camera with full body visible
   - Maintain upright posture throughout test

2. **Optimal Distance**:
   - 6-8 feet from camera for full body capture
   - Good lighting without shadows on body

3. **Movement Speed**:
   - Slow, controlled movements
   - Pause at maximum range for hold timer

4. **Common Issues**:
   - If skeleton tracking is lost, move closer to camera
   - Ensure arms are fully extended during abduction
   - Avoid rotating body - keep shoulders square to camera