
# Finger Gun Detection and Shooting Game

## Overview

This project implements a real-time computer vision-based game where the user can shoot targets by performing a "finger gun" gesture. The program utilizes the MediaPipe Hands library for hand landmark detection and OpenCV for video processing and rendering.

## Features

-   **Hand Gesture Recognition**: Detects finger gun gestures using MediaPipe hand landmarks.
-   **Target Shooting Mechanism**: Aims and shoots at moving targets by simulating a "bang" effect when the gesture is detected.
-   **Real-Time Feedback**: Visualizes the aim trajectory and gesture status on the camera feed.
-   **Score Tracking**: Updates and displays the score based on successful target hits.

## Requirements

-   Python 3.8+
-   OpenCV
-   MediaPipe
-   NumPy

Install dependencies using pip:

```bash
pip install opencv-python mediapipe numpy

```

## How It Works

1.  **Hand Tracking**:
    
    -   Detects hand landmarks in real-time using MediaPipe.
    -   Analyzes the positions and movements of the thumb and index finger to recognize a finger gun gesture.
2.  **Shooting Mechanism**:
    
    -   Detects when the thumb moves rapidly forward to simulate a "shooting" action.
    -   Calculates the aim vector based on the finger direction and checks if any target is hit along the trajectory.
3.  **Game Logic**:
    
    -   Spawns moving targets randomly on the screen.
    -   Updates the player's score for every successful target hit.
    -   Displays feedback, including the aim line, "BANG!" text, and visual effects when a shot is fired.

## Code Structure

-   **`ThumbPosition` and `ThumbState` Classes**: Handle thumb position and movement states.
-   **`Target` Class**: Represents a target object with properties for movement, position, and hit detection.
-   **Gesture Recognition**: Implements logic for detecting finger gun gestures using landmark positions.
-   **Shooting Detection**: Tracks thumb movements to determine when a shot is fired.
-   **Game Loop**: Handles video capture, hand detection, target spawning, and score updates.

## Usage

1.  Run the script:
    
    ```bash
    python your_script.py
    
    ```
    
2.  Point your camera at your hand and perform a "finger gun" gesture:
    -   Extend your index finger.
    -   Fold your middle, ring, and pinky fingers.
    -   Position your thumb to point up or outwards.
3.  Aim at the targets and "shoot" by moving your thumb forward quickly.
4.  Track your score as you hit the targets!

## Key Configurations

-   **Target Settings**:
    -   `TARGET_SPEED`: Speed of target movement.
    -   `TARGET_SIZE`: Radius of the targets.
    -   `MAX_TARGETS`: Maximum number of targets on screen.
-   **Gesture Detection**:
    -   `INDEX_EXTENSION_THRESHOLD`: Threshold for detecting index finger extension.
    -   `FINGER_FOLD_THRESHOLD`: Threshold for detecting folded fingers.
-   **Shooting Logic**:
    -   `SHOT_COOLDOWN`: Minimum time between shots.
    -   `MIN_SHOT_MOVEMENT`: Minimum thumb movement to trigger a shot.

## Output

-   Real-time display of the camera feed with the following overlays:
    -   Detected landmarks and hand connections.
    -   Targets moving across the screen.
    -   Aim trajectory line and shooting effects.
    -   Score displayed in the top-right corner.

## Future Improvements

-   Add sound effects for shooting and target hits.
-   Include different difficulty levels with varying target speeds and sizes.
-   Implement multi-hand support for dual-hand gameplay.

## Acknowledgments

-   [MediaPipe](https://mediapipe.dev/) for providing an easy-to-use library for hand detection.
-   [OpenCV](https://opencv.org/) for video capture and processing.

## License

This project is licensed under the MIT License.
