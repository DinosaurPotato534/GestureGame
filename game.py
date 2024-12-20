import cv2
import mediapipe as mp
import numpy as np
from tkinter import Tk, Label

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)

root = Tk()
root.title("Finger Gun Game")
score_label = Label(root, text="Score: 0", font=("Arial", 24))
score_label.pack()

score = 0
gesture_detected = False

def is_finger_gun_gesture(landmarks):
    thumb = landmarks[4]
    index_finger = landmarks[8]
    middle_finger = landmarks[12]
    ring_finger = landmarks[16]
    pinky = landmarks[20]

    thumb_extended = np.abs(thumb.y - landmarks[3].y) > 0.05
    index_extended = np.abs(index_finger.y - landmarks[7].y) > 0.05

    other_fingers_folded = (
        np.abs(middle_finger.y - landmarks[9].y) < 0.05 and
        np.abs(ring_finger.y - landmarks[13].y) < 0.05 and
        np.abs(pinky.y - landmarks[17].y) < 0.05
    )

    return thumb_extended and index_extended and other_fingers_folded

def update_score():
    global score
    score += 1
    score_label.config(text=f"Score: {score}")

def main():
    global score, gesture_detected

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                if is_finger_gun_gesture(landmarks.landmark):
                    if not gesture_detected:
                        update_score()
                        gesture_detected = True

                else:
                    gesture_detected = False

        cv2.imshow("Finger Gun Game - Hand Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

root.after(100, main)
root.mainloop()
