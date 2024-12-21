# Gesture-Controlled Shooting Game

## Overview
This project is a gesture-controlled shooting game implemented in Python using OpenCV and MediaPipe. Players can interact with the game by forming a "finger-gun" gesture to aim and shoot at moving targets on the screen. The program uses hand gesture recognition to detect shooting actions and compute aim trajectories, providing an engaging and intuitive experience.

---

## Features

### 1. **Hand Gesture Detection**
- Utilizes MediaPipe's hand landmark detection to identify hand gestures.
- Detects the "finger-gun" gesture with confidence scoring based on the position and orientation of the index finger and thumb.

### 2. **Shooting Mechanism**
- Recognizes a "shoot" action based on thumb movements relative to other fingers.
- Implements state management for the hand, transitioning between `READY`, `MOVING`, and `SHOT` states.

### 3. **Aim and Trajectory Calculation**
- Calculates the aim direction based on the relative position of the index finger joints.
- Smoothens aiming using a history buffer and weighted averaging to handle noise.

### 4. **Targets and Scoring**
- Randomly spawns moving targets on the screen.
- Allows players to shoot and destroy targets, earning points.

### 5. **Visual Effects**
- Displays real-time feedback, including:
  - Aim lines extending from the player's finger-gun gesture.
  - Animated "BANG!" effect when a shot is fired.
  - Target color change and explosion effect upon hit.

---

## How It Works

### 1. **Gesture Detection**
The program uses MediaPipe to detect hand landmarks and analyze gestures. The following criteria define the "finger-gun" gesture:
- **Index Finger:** Fully extended (measured by comparing joint positions).
- **Middle, Ring, and Pinky Fingers:** Folded (measured by relative distances to the palm).
- **Thumb:** Positioned at an angle that aligns with the finger-gun gesture.

### 2. **Shooting Action**
- Tracks thumb movement over time using a history buffer.
- Detects rapid upward motion of the thumb to trigger a "SHOT" state.
- Implements cooldowns and debounce logic to avoid accidental repeated shots.

### 3. **Aiming and Shooting**
- Calculates aim direction using the midpoint of the index finger's PIP and TIP joints.
- Smooths aim vector using a weighted history of recent movements to improve stability.
- Determines if a target is hit by projecting the aim vector and measuring proximity to targets.

### 4. **Target Management**
- Targets move horizontally across the screen.
- A limited number of targets are active at a time, with a random chance to spawn new targets.
- Upon being hit, targets change color and are removed from the active list.

---

## Installation

### Prerequisites
- Python 3.7+
- OpenCV (`cv2`)
- MediaPipe
- NumPy

### Setup
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. Install dependencies:
   ```bash
   pip install opencv-python mediapipe numpy
   ```
3. Run the game:
   ```bash
   python main.py
   ```

---

## Controls
- **Form a "finger-gun" gesture:** Aim at the targets.
- **Shoot:** Perform a quick upward motion with your thumb.

---

## Configuration
- Modify the constants at the top of the script to adjust gameplay:
  - `SHOT_COOLDOWN`: Minimum time between consecutive shots.
  - `TARGET_SPEED`: Speed of moving targets.
  - `TARGET_SPAWN_RATE`: Probability of spawning new targets.
  - `SHOT_RANGE`: Maximum range for detecting target hits.

---

## Acknowledgments
- **MediaPipe Hands** for robust hand tracking and landmark detection.
- **OpenCV** for video processing and visualization.

---

## Future Improvements
- Add sound effects for shooting and target hits.
- Implement a game timer and leaderboard.
- Support for multiple players and competitive modes.
- Enhance gesture recognition to support additional actions.

