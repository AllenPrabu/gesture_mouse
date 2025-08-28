# main.py
# This version uses the 'mouse' library for mouse control and 'keyboard' for Alt+Tab navigation.

import cv2
import mediapipe as mp
import math
import numpy as np
import time
import sys
import config
import speech_recognition as sr
import threading
import mouse 
import keyboard 

# --- Helper Functions ---
def get_landmark_pixel_coords(hand_landmarks, landmark_id, frame_width, frame_height):
    landmark = hand_landmarks.landmark[landmark_id]
    return int(landmark.x * frame_width), int(landmark.y * frame_height)

def are_fingers_up(hand_landmarks, finger_indices):
    for i in finger_indices:
        tip = hand_landmarks.landmark[i]; pip = hand_landmarks.landmark[i - 2]
        if tip.y >= pip.y: return False
    return True

def are_fingers_down(hand_landmarks, finger_indices):
    for i in finger_indices:
        tip = hand_landmarks.landmark[i]; mcp = hand_landmarks.landmark[i - 3]
        if tip.y < mcp.y: return False
    return True

# --- Voice Control Function (runs in a separate thread) ---
def listen_for_commands(state):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)

    while state['running']:
        try:
            with microphone as source:
                if not state['listening_active']:
                    print("Listening for wake word 'computer'...")
                    audio = recognizer.listen(source, phrase_time_limit=2)
                    try:
                        wake_word_text = recognizer.recognize_google(audio).lower()
                        if "computer" in wake_word_text:
                            state['listening_active'] = True
                            state['status_text'] = "Computer Active"
                            print("Wake word detected!")
                    except sr.UnknownValueError:
                        continue
                
                if state['listening_active']:
                    print("Listening for command or dictation...")
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    text = recognizer.recognize_google(audio).lower()
                    print(f"Recognized: {text}")

                    if text.startswith("computer"):
                        command = text.replace("computer", "").strip()
                        if command == "press enter" or command == "enter":
                            mouse.press(button='left')
                            mouse.release(button='left')
                            state['status_text'] = "Computer: Enter (simulated left click)"
                        elif command == "backspace":
                            state['status_text'] = "Computer: Backspace (not supported)"
                        elif command == "clear":
                            state['status_text'] = "Computer: Clear (not supported)"
                        elif command == "stop listening":
                            state['status_text'] = "Computer Inactive"
                        else:
                            state['status_text'] = "Unknown Command"
                    else:
                        state['status_text'] = "Typing not supported"
                    
                    state['listening_active'] = False

        except sr.WaitTimeoutError:
            state['listening_active'] = False; state['status_text'] = "Voice Timed Out"
            print("No speech detected, voice control deactivated.")
        except sr.UnknownValueError:
            state['listening_active'] = False; state['status_text'] = "Could not understand"
            print("Could not understand audio, voice control deactivated.")
        except Exception as e:
            print(f"An error occurred: {e}")
            state['listening_active'] = False

# --- Main Application ---
def main():
    # --- CONSTANTS AND SETUP ---
    SMOOTHING_FACTOR = config.SMOOTHING_FACTOR
    SENSITIVITY_PADDING = config.SENSITIVITY_PADDING
    CLICK_DISTANCE_THRESHOLD = config.CLICK_DISTANCE_THRESHOLD
    ACTION_COOLDOWN = config.ACTION_COOLDOWN
    SWIPE_NAV_DISTANCE_RATIO = config.SWIPE_NAV_DISTANCE_RATIO
    SCROLL_SPEED = config.SCROLL_SPEED  # <-- Add this line
    
    # --- INITIALIZE LIBRARIES ---
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Get screen size using mouse library (fallback to pyautogui if needed)
    try:
        import ctypes
        user32 = ctypes.windll.user32
        screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        screen_width, screen_height = 1920, 1080  # fallback

    # --- STATE VARIABLES ---
    shared_state = {'running': True, 'listening_active': False, 'status_text': "Initializing..."}
    prev_x, prev_y = 0, 0
    last_action_time = 0
    is_app_switcher_active = False
    app_switcher_start_x = None
    last_swipe_nav_time = 0
    last_mouse_move_time = 0

    # --- Start Voice Control Thread ---
    voice_thread = threading.Thread(target=listen_for_commands, args=(shared_state,))
    voice_thread.daemon = True
    voice_thread.start()

    print("Starting in 2 seconds... Hand control is active.")
    time.sleep(2)

    # --- MAIN LOOP ---
    while cap.isOpened():
        loop_start = time.time()
        current_time = time.time()

        success, frame = cap.read()
        if not success: continue
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape

        t1 = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        t2 = time.time()

        if results.multi_hand_landmarks:
            if not shared_state['listening_active']:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                index_tip, middle_tip, ring_tip, pinky_tip = mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP
                thumb_tip, wrist = mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.WRIST
                is_fist = are_fingers_down(hand_landmarks, [index_tip, middle_tip, ring_tip, pinky_tip])

                # --- ALT+TAB APP SWITCHER LOGIC ---
                if is_fist and not is_app_switcher_active:
                    keyboard.press('alt')
                    keyboard.press_and_release('tab')
                    is_app_switcher_active = True
                    app_switcher_start_x, _ = get_landmark_pixel_coords(hand_landmarks, wrist, frame_width, frame_height)
                    shared_state['status_text'] = "App Switcher"
                    last_swipe_nav_time = current_time
                elif is_fist and is_app_switcher_active:
                    # While holding fist, check for left/right wrist movement to switch tabs
                    wrist_x, _ = get_landmark_pixel_coords(hand_landmarks, wrist, frame_width, frame_height)
                    if abs(wrist_x - app_switcher_start_x) > SWIPE_NAV_DISTANCE_RATIO * frame_width and (current_time - last_swipe_nav_time) > 0.25:
                        if wrist_x > app_switcher_start_x:
                            keyboard.press_and_release('right')
                            shared_state['status_text'] = "App Switcher: Next"
                        else:
                            keyboard.press_and_release('left')
                            shared_state['status_text'] = "App Switcher: Prev"
                        app_switcher_start_x = wrist_x
                        last_swipe_nav_time = current_time
                elif not is_fist and is_app_switcher_active:
                    keyboard.release('alt')
                    is_app_switcher_active = False
                    app_switcher_start_x = None
                    shared_state['status_text'] = "App Selected"

                if not is_fist and not is_app_switcher_active:
                    is_scroll_gesture = are_fingers_up(hand_landmarks, [index_tip, middle_tip, ring_tip]) and are_fingers_down(hand_landmarks, [pinky_tip])
                    is_pointing = are_fingers_up(hand_landmarks, [index_tip]) and are_fingers_down(hand_landmarks, [middle_tip, ring_tip, pinky_tip])
                    
                    if is_scroll_gesture:
                        shared_state['status_text'] = "Scrolling"
                        scroll_amount = np.interp(
                            hand_landmarks.landmark[middle_tip].y, [0.7, 0.3], [-20, 20]
                        )
                        mouse.wheel(int(scroll_amount * SCROLL_SPEED))  # <-- Use SCROLL_SPEED
                    elif is_pointing:
                        shared_state['status_text'] = "Mouse Control"
                        target_x = np.interp(hand_landmarks.landmark[index_tip].x, (SENSITIVITY_PADDING/frame_width, 1 - SENSITIVITY_PADDING/frame_width), (0, screen_width))
                        target_y = np.interp(hand_landmarks.landmark[index_tip].y, (SENSITIVITY_PADDING/frame_height, 1 - SENSITIVITY_PADDING/frame_height), (0, screen_height))
                        current_x = prev_x + (target_x - prev_x) / SMOOTHING_FACTOR
                        current_y = prev_y + (target_y - prev_y) / SMOOTHING_FACTOR
                        now = time.time()
                        if (now - last_mouse_move_time) > 0.03 and (abs(current_x - prev_x) > 3 or abs(current_y - prev_y) > 3):
                            mouse.move(current_x, current_y, absolute=True, duration=0)
                            prev_x, prev_y = current_x, current_y
                            last_mouse_move_time = now
                    else:
                        if shared_state['status_text'] != "App Selected": shared_state['status_text'] = "Ready"

                    # --- CLICK LOGIC ---
                    can_perform_action = (current_time - last_action_time) > ACTION_COOLDOWN
                    thumb_px, thumb_py = get_landmark_pixel_coords(hand_landmarks, thumb_tip, frame_width, frame_height)
                    index_px, index_py = get_landmark_pixel_coords(hand_landmarks, index_tip, frame_width, frame_height)
                    middle_px, middle_py = get_landmark_pixel_coords(hand_landmarks, middle_tip, frame_width, frame_height)
                    
                    # Left click: index and thumb touch
                    if math.hypot(index_px - thumb_px, index_py - thumb_py) < CLICK_DISTANCE_THRESHOLD and can_perform_action:
                        mouse.click(button='left')
                        last_action_time = current_time
                        shared_state['status_text'] = "Left Click!"
                    # Right click: middle and thumb touch
                    elif math.hypot(middle_px - thumb_px, middle_py - thumb_py) < CLICK_DISTANCE_THRESHOLD and can_perform_action:
                        mouse.click(button='right')
                        last_action_time = current_time
                        shared_state['status_text'] = "Right Click!"
        else:
            if not shared_state['listening_active']:
                shared_state['status_text'] = "No Hand Detected"

        t3 = time.time()
        cv2.putText(frame, f"MODE: {shared_state['status_text']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Gesture & Voice Control', frame)
        t4 = time.time()


        if cv2.waitKey(1) & 0xFF == ord('q'):
            shared_state['running'] = False
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    voice_thread.join()

if __name__ == "__main__":
    main()