# main.py
# This version includes an interactive App Switcher Mode.

import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np
import time
import sys
import config

# --- Helper Functions ---

def get_landmark_pixel_coords(hand_landmarks, landmark_id, frame_width, frame_height):
    """Helper function to get the pixel coordinates of a specific landmark."""
    landmark = hand_landmarks.landmark[landmark_id]
    return int(landmark.x * frame_width), int(landmark.y * frame_height)

def are_fingers_up(hand_landmarks, finger_indices):
    """Checks if specified fingers are extended straight."""
    for i in finger_indices:
        tip = hand_landmarks.landmark[i]
        pip = hand_landmarks.landmark[i - 2]
        if tip.y >= pip.y:
            return False
    return True

def are_fingers_down(hand_landmarks, finger_indices):
    """Checks if specified fingers are curled down."""
    for i in finger_indices:
        tip = hand_landmarks.landmark[i]
        mcp = hand_landmarks.landmark[i - 3]
        if tip.y < mcp.y:
            return False
    return True

# --- Main Application ---

def main():
    """
    The main function that runs the gesture recognition and mouse control.
    """
    # --- CONSTANTS AND SETUP (Loaded from config.py) ---
    SMOOTHING_FACTOR = config.SMOOTHING_FACTOR
    SENSITIVITY_PADDING = config.SENSITIVITY_PADDING
    CLICK_DISTANCE_THRESHOLD = config.CLICK_DISTANCE_THRESHOLD
    ACTION_COOLDOWN = config.ACTION_COOLDOWN
    SWIPE_NAV_DISTANCE_RATIO = config.SWIPE_NAV_DISTANCE_RATIO

    # OS-specific keys
    APP_SWITCH_KEY = 'command' if sys.platform == 'darwin' else 'alt'
    
    # 1. Initialize Webcam & MediaPipe
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # 2. Get Screen Dimensions
    screen_width, screen_height = pyautogui.size()

    # --- STATE VARIABLES ---
    prev_x, prev_y = 0, 0
    is_left_pinching = False
    last_action_time = 0
    status_text = "Initializing..."
    
    # ✨ New state variables for the interactive app switcher
    is_app_switcher_active = False
    swipe_nav_start_x = 0
    last_swipe_nav_time = 0

    print("Starting in 2 seconds... Hand control is active.")
    time.sleep(2)

    # --- MAIN LOOP ---
    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue

        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_time = time.time()
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip, middle_tip, ring_tip, pinky_tip = mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP
            thumb_tip, wrist = mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.WRIST

            is_fist = are_fingers_down(hand_landmarks, [index_tip, middle_tip, ring_tip, pinky_tip])
            
            # --- ✨ INTERACTIVE APP SWITCHER LOGIC ---
            if is_fist and not is_app_switcher_active:
                # 1. Enter App Switcher Mode
                is_app_switcher_active = True
                pyautogui.keyDown(APP_SWITCH_KEY)
                pyautogui.press('tab')
                swipe_nav_start_x = hand_landmarks.landmark[wrist].x
                last_swipe_nav_time = current_time
                status_text = "App Switching"
            
            elif is_fist and is_app_switcher_active:
                # 2. Navigate within the switcher
                status_text = "App Switching"
                can_nav = (current_time - last_swipe_nav_time) > 0.4 # Cooldown for navigation
                
                if can_nav:
                    current_wrist_x = hand_landmarks.landmark[wrist].x
                    delta_x = current_wrist_x - swipe_nav_start_x

                    if delta_x > SWIPE_NAV_DISTANCE_RATIO: # Navigate Right
                        pyautogui.press('tab')
                        swipe_nav_start_x = current_wrist_x
                        last_swipe_nav_time = current_time
                    elif delta_x < -SWIPE_NAV_DISTANCE_RATIO: # Navigate Left
                        pyautogui.hotkey('shift', 'tab')
                        swipe_nav_start_x = current_wrist_x
                        last_swipe_nav_time = current_time
            
            elif not is_fist and is_app_switcher_active:
                # 3. Exit App Switcher Mode and Select
                is_app_switcher_active = False
                pyautogui.keyUp(APP_SWITCH_KEY)
                last_action_time = current_time
                status_text = "App Selected"
            
            # --- OTHER GESTURES (only if not in App Switcher Mode) ---
            elif not is_app_switcher_active:
                is_scroll_gesture = are_fingers_up(hand_landmarks, [index_tip, middle_tip, ring_tip]) and are_fingers_down(hand_landmarks, [pinky_tip])
                is_pointing = are_fingers_up(hand_landmarks, [index_tip]) and are_fingers_down(hand_landmarks, [middle_tip, ring_tip, pinky_tip])

                can_perform_action = (current_time - last_action_time) > ACTION_COOLDOWN

                if is_scroll_gesture:
                    status_text = "Scrolling"
                    scroll_amount = np.interp(hand_landmarks.landmark[middle_tip].y, [0.7, 0.3], [-20, 20])
                    pyautogui.scroll(int(scroll_amount))

                elif is_pointing:
                    status_text = "Mouse Control"
                    target_x = np.interp(hand_landmarks.landmark[index_tip].x, (SENSITIVITY_PADDING/frame_width, 1 - SENSITIVITY_PADDING/frame_width), (0, screen_width))
                    target_y = np.interp(hand_landmarks.landmark[index_tip].y, (SENSITIVITY_PADDING/frame_height, 1 - SENSITIVITY_PADDING/frame_height), (0, screen_height))
                    
                    current_x = prev_x + (target_x - prev_x) / SMOOTHING_FACTOR
                    current_y = prev_y + (target_y - prev_y) / SMOOTHING_FACTOR
                    pyautogui.moveTo(current_x, current_y)
                    prev_x, prev_y = current_x, current_y
                
                else:
                    if status_text != "App Selected": status_text = "Ready"

                # Click Detection
                thumb_px, thumb_py = get_landmark_pixel_coords(hand_landmarks, thumb_tip, frame_width, frame_height)
                index_px, index_py = get_landmark_pixel_coords(hand_landmarks, index_tip, frame_width, frame_height)
                
                if math.hypot(index_px - thumb_px, index_py - thumb_py) < CLICK_DISTANCE_THRESHOLD and can_perform_action:
                    pyautogui.click()
                    last_action_time = current_time
                    status_text = "Click!"
        else:
            # If hand is lost, release the Alt key to avoid getting stuck
            if is_app_switcher_active:
                pyautogui.keyUp(APP_SWITCH_KEY)
                is_app_switcher_active = False
            status_text = "No Hand Detected"

        cv2.putText(frame, f"MODE: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Gesture Control', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()