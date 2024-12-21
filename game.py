import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from time import time
import random

class ThumbPosition:
    def __init__(self, x, y, z, t, relative_y):
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.relative_y = relative_y

class ThumbState:
    def __init__(self, angle, position_y, t):
        self.angle = angle
        self.position_y = position_y
        self.t = t

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

thumb_states = deque(maxlen=5)
last_shot_time = 0
SHOT_COOLDOWN = 1.0
ANGLE_SPEED_THRESHOLD = 150
POSITION_SPEED_THRESHOLD = 0.3
SMOOTHING_WINDOW = 3

SHOT_STATES = ['READY', 'MOVING', 'SHOT']
current_shot_state = 'READY'
thumb_history = []
THUMB_MOVE_THRESHOLD = 0.008
MIN_SHOT_MOVEMENT = 0.015
MAX_SHOT_TIME = 0.6
MIN_SHOT_TIME = 0.05
DEBOUNCE_TIME = 0.4
MIN_CONFIDENCE = 0.6
INDEX_EXTENSION_THRESHOLD = 0.12
FINGER_FOLD_THRESHOLD = 0.1

last_successful_shot = 0

GESTURE_HISTORY_SIZE = 3
GESTURE_CONFIDENCE_THRESHOLD = 0.7
gesture_history = deque([False] * GESTURE_HISTORY_SIZE, maxlen=GESTURE_HISTORY_SIZE)

STATE_TIMEOUT = 1.0
RECOVERY_TIME = 0.5

last_state_change = time()
last_gesture_loss = 0

SHOT_EFFECT_DURATION = 0.2

last_shot_effect = 0
shot_fired = False
shot_effect_active = False

TARGET_SPEED = 5
TARGET_SIZE = 40
TARGET_SPAWN_RATE = 0.02
MAX_TARGETS = 5
SHOT_RANGE = 30
SCORE_POINTS = 10

class Target:
    def __init__(self, frame_width, frame_height):
        self.size = TARGET_SIZE
        self.y = random.randint(100, frame_height - 100)
        self.from_left = random.choice([True, False])
        self.x = -self.size if self.from_left else frame_width + self.size
        self.active = True
        self.hit = False
        self.color = (0, 0, 255)

    def move(self, frame_width):
        if self.from_left:
            self.x += TARGET_SPEED
            if self.x > frame_width + self.size:
                self.active = False
        else:
            self.x -= TARGET_SPEED
            if self.x < -self.size:
                self.active = False

    def draw(self, frame):
        cv2.circle(frame, (int(self.x), int(self.y)), self.size, self.color, 2)
        cv2.circle(frame, (int(self.x), int(self.y)), self.size//3, self.color, 2)

targets = []
score = 0

def calculate_finger_angle(point1, point2, point3):
    vector1 = np.array([point1.x - point2.x, point1.y - point2.y])
    vector2 = np.array([point3.x - point2.x, point3.y - point2.y])
    
    cosine = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

def get_finger_direction(base, tip):
    return np.array([tip.x - base.x, tip.y - base.y, tip.z - base.z])

def is_finger_extended(finger_vec):
    return abs(np.dot(finger_vec, [0, -1, 0]) / np.linalg.norm(finger_vec)) > 0.3

def detect_shot(landmarks, hand_id):
    state = hand_states[hand_id]
    current_time = time()
    
    if current_time - state.last_state_change > STATE_TIMEOUT:
        state.current_shot_state = 'READY'
        state.thumb_history.clear()
        state.last_state_change = current_time
        state.shot_effect_active = False
    
    if current_time - state.last_successful_shot < DEBOUNCE_TIME:
        return False

    thumb_tip = landmarks[4]
    index_base = landmarks[5]
    relative_pos = thumb_tip.y - index_base.y
    
    state.thumb_history.append((relative_pos, current_time))
    if len(state.thumb_history) > 5:
        state.thumb_history.pop(0)
    
    if len(state.thumb_history) < 2:
        return False

    start_pos, start_time = state.thumb_history[0]
    current_pos, current_time = state.thumb_history[-1]
    movement = current_pos - start_pos
    elapsed_time = current_time - start_time
    speed = movement / elapsed_time if elapsed_time > 0 else 0
    
    if state.current_shot_state == 'READY':
        if movement > THUMB_MOVE_THRESHOLD:
            state.current_shot_state = 'MOVING'
            state.last_state_change = current_time
    
    elif state.current_shot_state == 'MOVING':
        if elapsed_time > MAX_SHOT_TIME:
            state.current_shot_state = 'READY'
            state.last_state_change = current_time
            state.thumb_history.clear()
        elif movement > MIN_SHOT_MOVEMENT:
            positions = [pos for pos, _ in state.thumb_history]
            movements = [b - a for a, b in zip(positions, positions[1:])]
            is_consistent = sum(m > 0 for m in movements) >= len(movements) * 0.7
            
            if is_consistent:
                state.current_shot_state = 'SHOT'
                state.last_successful_shot = current_time
                state.last_state_change = current_time
                state.thumb_history.clear()
                state.shot_effect_active = True
                return True
    
    elif state.current_shot_state == 'SHOT':
        if current_time - state.last_successful_shot >= DEBOUNCE_TIME:
            state.current_shot_state = 'READY'
            state.last_state_change = current_time
            state.thumb_history.clear()
            state.shot_effect_active = False
    
    return False

def is_finger_gun_gesture(landmarks):
    index_mcp = landmarks[5]
    index_tip = landmarks[8]
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]

    palm_size = np.linalg.norm(get_finger_direction(landmarks[0], landmarks[5]))
    
    index_vec = get_finger_direction(index_mcp, index_tip)
    vertical_component = abs(index_vec[1]) / np.linalg.norm(index_vec)
    
    index_extension = np.linalg.norm(get_finger_direction(landmarks[5], landmarks[8])) / palm_size
    middle_fold = np.linalg.norm(get_finger_direction(landmarks[9], landmarks[12])) / palm_size
    ring_fold = np.linalg.norm(get_finger_direction(landmarks[13], landmarks[16])) / palm_size
    pinky_fold = np.linalg.norm(get_finger_direction(landmarks[17], landmarks[20])) / palm_size

    confidence = 0.0
    if index_extension > INDEX_EXTENSION_THRESHOLD:
        confidence += 0.4
        if vertical_component < 0.8:
            confidence += 0.1
    if all(fold < FINGER_FOLD_THRESHOLD for fold in [middle_fold, ring_fold, pinky_fold]):
        confidence += 0.3
    
    thumb_vec = get_finger_direction(thumb_mcp, thumb_tip)
    index_vec = get_finger_direction(index_mcp, index_tip)
    thumb_angle = np.arccos(np.clip(np.dot(thumb_vec, index_vec), -1.0, 1.0))
    if 0.8 < thumb_angle < 2.7:
        confidence += 0.2
        
    gesture_detected = confidence >= MIN_CONFIDENCE
    gesture_history.append(gesture_detected)
    
    return sum(gesture_history) >= GESTURE_CONFIDENCE_THRESHOLD * GESTURE_HISTORY_SIZE

AIM_LINE_LENGTH = 1000
AIM_THRESHOLD = 0.2
SHOW_AIM_LINE = True

AIM_SMOOTHING_FRAMES = 4
AIM_LERP_FACTOR = 0.3
AIM_STEADY_THRESHOLD = 0.015
MOVEMENT_RESET_THRESHOLD = 0.1

class HandState:
    def __init__(self):
        self.aim_history = deque(maxlen=AIM_SMOOTHING_FRAMES)
        self.last_aim_vector = (0, 0)
        self.aim_steadiness = 0.0
        self.last_aim_pos = None
        self.last_hand_position = None
        self.last_frame_landmarks = None
        self.thumb_history = []
        self.current_shot_state = 'READY'
        self.last_successful_shot = 0
        self.last_state_change = time()
        self.shot_effect_active = False
        self.last_gesture_loss = 0

hand_states = {'Left': HandState(), 'Right': HandState()}

def get_smooth_aim_vector(landmarks, hand_id):
    state = hand_states[hand_id]
    
    hand_center = np.mean([(landmarks[i].x, landmarks[i].y) for i in range(21)], axis=0)
    
    if state.last_hand_position is not None:
        movement = np.sqrt(
            (hand_center[0] - state.last_hand_position[0])**2 + 
            (hand_center[1] - state.last_hand_position[1])**2
        )
        if movement > MOVEMENT_RESET_THRESHOLD:
            state.aim_history.clear()
            state.last_aim_vector = (0, 0)
            state.aim_steadiness = 0.0
    
    state.last_hand_position = hand_center
    state.last_frame_landmarks = landmarks
    
    index_mcp = landmarks[5]
    index_pip = landmarks[6]
    index_tip = landmarks[8]
    
    mid_x = (index_pip.x + index_tip.x) / 2
    mid_y = (index_pip.y + index_tip.y) / 2
    
    dx = mid_x - index_mcp.x
    dy = mid_y - index_mcp.y
    length = np.sqrt(dx*dx + dy*dy)
    
    if length > 0:
        current_vector = (dx/length, dy/length)
    else:
        return state.last_aim_vector if state.last_aim_vector[0] != 0 or state.last_aim_vector[1] != 0 else (0, -1)
    
    if len(state.aim_history) == 0:
        state.last_aim_vector = current_vector
    
    state.aim_history.append(current_vector)
    
    if len(state.aim_history) > 0:
        weights = np.linspace(0.5, 1, len(state.aim_history))
        weights = weights / np.sum(weights)
        
        avg_x = sum(v[0] * w for v, w in zip(state.aim_history, weights))
        avg_y = sum(v[1] * w for v, w in zip(state.aim_history, weights))
        
        length = np.sqrt(avg_x*avg_x + avg_y*avg_y)
        if length > 0:
            smooth_vector = (avg_x/length, avg_y/length)
            state.last_aim_vector = (
                smooth_vector[0] * AIM_LERP_FACTOR + state.last_aim_vector[0] * (1 - AIM_LERP_FACTOR),
                smooth_vector[1] * AIM_LERP_FACTOR + state.last_aim_vector[1] * (1 - AIM_LERP_FACTOR)
            )
    
    # Normalize final vector
    length = np.sqrt(state.last_aim_vector[0]**2 + state.last_aim_vector[1]**2)
    if length > 0:
        return (state.last_aim_vector[0]/length, state.last_aim_vector[1]/length)
    return current_vector

def get_aim_vector(landmarks, hand_id):
    return get_smooth_aim_vector(landmarks, hand_id)

def check_trajectory_hit(start_x, start_y, aim_vector, target, hand_id):
    state = hand_states[hand_id]
    dx, dy = aim_vector
    v = (target.x - start_x, target.y - start_y)
    t = max(0, (v[0]*dx + v[1]*dy))
    
    closest_x = start_x + t * dx
    closest_y = start_y + t * dy
    
    effective_range = SHOT_RANGE * (PRECISE_AIM_MULTIPLIER if state.aim_steadiness > 0.8 else 1.0)
    
    distance = np.sqrt((target.x - closest_x)**2 + (target.y - closest_y)**2)
    in_front = (v[0]*dx + v[1]*dy) > 0
    
    max_distance = np.sqrt(frame.shape[1]**2 + frame.shape[0]**2)
    target_distance = np.sqrt((target.x - start_x)**2 + (target.y - start_y)**2)
    distance_factor = 1.0 - (target_distance / max_distance * 0.5)
    
    return distance < (TARGET_SIZE + effective_range) * distance_factor and in_front

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)

    if len(targets) < MAX_TARGETS and random.random() < TARGET_SPAWN_RATE:
        targets.append(Target(frame.shape[1], frame.shape[0]))

    targets = [target for target in targets if target.active]
    for target in targets:
        target.move(frame.shape[1])
        target.draw(frame)

    cv2.putText(frame, f"Score: {score}", (frame.shape[1] - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    gesture_detected = False
    if results.multi_hand_landmarks:
        handedness = results.multi_handedness
        
        for idx, (landmarks, hand_info) in enumerate(zip(results.multi_hand_landmarks, handedness)):
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            hand_id = hand_info.classification[0].label
            
            if is_finger_gun_gesture(landmarks.landmark):
                state = hand_states[hand_id]
                state.last_gesture_loss = 0
                
                status_text = "FINGER GUN DETECTED!"
                cv2.putText(frame, status_text, (20, 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.putText(frame, f"State: {state.current_shot_state}", 
                          (20, frame.shape[0] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                current_time = time()
                did_shoot = detect_shot(landmarks.landmark, hand_id)
                
                if did_shoot or (state.shot_effect_active and current_time - state.last_successful_shot < SHOT_EFFECT_DURATION):
                    index_mcp = landmarks.landmark[5]
                    index_pip = landmarks.landmark[6]
                    index_tip = landmarks.landmark[8]
                    
                    shot_x = int(((index_pip.x + index_tip.x) / 2) * frame.shape[1])
                    shot_y = int(((index_pip.y + index_tip.y) / 2) * frame.shape[0])
                    
                    aim_vector = get_aim_vector(landmarks.landmark, hand_id)
                    
                    if SHOW_AIM_LINE:
                        aim_end_x = int(shot_x + aim_vector[0] * AIM_LINE_LENGTH)
                        aim_end_y = int(shot_y + aim_vector[1] * AIM_LINE_LENGTH)
                        aim_color = (0, 255 * min(1, state.aim_steadiness + 0.2), 255 * (1 - state.aim_steadiness))
                        cv2.line(frame, (shot_x, shot_y), (aim_end_x, aim_end_y), 
                                aim_color, 1 + int(state.aim_steadiness * 2), cv2.LINE_AA)
                    
                    for radius in range(40, 10, -10):
                        color = (0, 255, 255) if radius % 20 == 0 else (255, 255, 255)
                        cv2.circle(frame, (shot_x, shot_y), radius, color, -1)
                    
                    cv2.putText(frame, "BANG!", (frame.shape[1]//2 - 50, frame.shape[0]//2),
                              cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    
                    for target in targets:
                        if not target.hit and check_trajectory_hit(shot_x, shot_y, aim_vector, target, hand_id):
                            target.hit = True
                            target.active = False
                            target.color = (0, 255, 0)
                            score += SCORE_POINTS
                            cv2.circle(frame, (int(target.x), int(target.y)), 
                                     target.size + 10, (0, 255, 255), -1)
                
                elif is_finger_gun_gesture(landmarks.landmark):
                    index_tip = landmarks.landmark[8]
                    aim_x = int(index_tip.x * frame.shape[1])
                    aim_y = int(index_tip.y * frame.shape[0])
                    aim_vector = get_aim_vector(landmarks.landmark, hand_id)
                    
                    if SHOW_AIM_LINE:
                        aim_end_x = int(aim_x + aim_vector[0] * AIM_LINE_LENGTH)
                        aim_end_y = int(aim_y + aim_vector[1] * AIM_LINE_LENGTH)
                        cv2.line(frame, (aim_x, aim_y), (aim_end_x, aim_end_y), 
                                (0, 255, 255), 1, cv2.LINE_AA)
            else:
                state = hand_states[hand_id]
                current_time = time()
                if state.last_gesture_loss == 0:
                    state.last_gesture_loss = current_time
                elif current_time - state.last_gesture_loss > RECOVERY_TIME:
                    state.current_shot_state = 'READY'
                    state.thumb_history.clear()
                    state.last_successful_shot = 0
                    state.last_state_change = current_time
                    state.shot_effect_active = False
                cv2.putText(frame, "No gesture detected", (20, 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                state.current_shot_state = 'READY'
                state.thumb_history.clear()

            thumb_tip = landmarks.landmark[4]
            thumb_x = int(thumb_tip.x * frame.shape[1])
            thumb_y = int(thumb_tip.y * frame.shape[0])
            
            color = (0, 255, 0) if state.current_shot_state == 'READY' else (0, 165, 255)
            cv2.circle(frame, (thumb_x, thumb_y), 10, color, -1)
            
            if state.current_shot_state == 'READY':
                cv2.putText(frame, "Move thumb down to shoot", 
                           (frame.shape[1]//2 - 100, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Finger Gun Detector", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
