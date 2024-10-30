import cv2
import mediapipe as mp
import webbrowser
import numpy as np
import time
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=5, min_detection_confidence=0.7)
prev_hand_center = None
prev_index_finger_tip = None

def finger_up(finger):
    for i in range(4):
        for j in range(i, 4):
            if finger[i].y - finger[j].y > 0.01:
                return False

    return True


cap = cv2.VideoCapture(0)
while True:
    time.sleep(0.0001)
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            thumb = [
                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP],
                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP],
                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC],
            ]
            index = [
                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
            ]
            middle = [
                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
            ]
            ring = [
                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP],
            ]
            pinky = [
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
            ]

            indexUp = finger_up(index)
            middleUp = finger_up(middle)
            ringUp = finger_up(ring)
            pinkyUp = finger_up(pinky)

            if(indexUp):
                pyautogui.moveTo(index[0].x*pyautogui.size()[0], index[0].y*pyautogui.size()[1])
                print(index[0].x*pyautogui.size()[0], index[0].y*pyautogui.size()[1])
                if(middleUp):
                    pyautogui.click()
           
            # hand_center = np.mean(
            #     [[landmark.x, landmark.y] for landmark in hand_landmarks.landmark],
            #     axis=0,
            # )
            # hand_center = (
            #     int(hand_center[0] * frame.shape[1]),
            #     int(hand_center[1] * frame.shape[0]),
            # )
            # detect_swipe(hand_center)
        # try:
        #     mp_hands.solutions.drawing_utils.draw_landmarks(
        #         frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
        #     )
        # except Exception:
        #     print("")

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
