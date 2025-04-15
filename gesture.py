import cv2
import mediapipe as mp
import pyautogui
import math
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

cooldown = 0.3
last_action_time = 0
last_screenshot_time = 0
screenshot_cooldown = 2
prev_y = None
font = cv2.FONT_HERSHEY_SIMPLEX

zoom_threshold = 35  # a bit more tolerant for zoom out
# zoom_spread_buffer = 50  # more intentional for zoom in
zoom_cooldown = 0.5
last_zoom_time = 0

def is_finger_up(lm, tip_id, pip_id):
    return lm[tip_id].y < lm[pip_id].y

def fingers_up(lm):
    return {
        'index': is_finger_up(lm, 8, 6),
        'middle': is_finger_up(lm, 12, 10),
        'ring': is_finger_up(lm, 16, 14),
        'pinky': is_finger_up(lm, 20, 18),
        'thumb': lm[4].x < lm[3].x  # thumb to the right
    }

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        h, w, _ = image.shape
        action_text = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                finger_state = fingers_up(lm)

                # Get key finger points
                index_tip = (int(lm[8].x * w), int(lm[8].y * h))
                thumb_tip = (int(lm[4].x * w), int(lm[4].y * h))
                index_tip_y = index_tip[1]
                now = time.time()

                # ---------------------- Zoom Detection ----------------------
                pinch_distance = distance(index_tip, thumb_tip)
                if now - last_zoom_time > zoom_cooldown:
                    if pinch_distance < zoom_threshold:
                        pyautogui.hotkey('ctrl', '-')
                        action_text = "Zoom Out"
                        last_zoom_time = now
                    elif pinch_distance > zoom_threshold + 40:
                        pyautogui.hotkey('ctrl', '+')
                        action_text = "Zoom In"
                        last_zoom_time = now

                # ------------------- Screenshot Gesture --------------------
                if (finger_state['index'] and finger_state['middle'] and finger_state['ring']
                        and not finger_state['pinky'] and not finger_state['thumb']):
                    if now - last_screenshot_time > screenshot_cooldown:
                        pyautogui.screenshot("gesture_screenshot.png")
                        action_text = 'Screenshot Taken'
                        last_screenshot_time = now
                    continue  # Skip other actions

                # ------------------ Scroll or Volume Mode ------------------
                mode = None
                if finger_state['index'] and finger_state['middle'] and not finger_state['ring'] and not finger_state['pinky']:
                    mode = 'scroll'
                elif finger_state['index'] and not finger_state['middle'] and not finger_state['ring'] and not finger_state['pinky']:
                    mode = 'volume'

                if prev_y is not None and mode and now - last_action_time > cooldown:
                    dy = index_tip_y - prev_y
                    if abs(dy) > 20:
                        if mode == 'scroll':
                            pyautogui.scroll(-1000 if dy > 0 else 1000)
                            action_text = 'Scroll Down' if dy > 0 else 'Scroll Up'
                        elif mode == 'volume':
                            pyautogui.press(['volumedown'] * 5 if dy > 0 else ['volumeup'] * 5)
                            action_text = 'Volume Down' if dy > 0 else 'Volume Up'
                        last_action_time = now

                prev_y = index_tip_y

        if action_text:
            cv2.putText(image, action_text, (10, 40), font, 1, (0, 255, 255), 2)

        cv2.imshow('Gesture Control + Zoom', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
