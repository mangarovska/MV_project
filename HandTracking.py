import cv2
import mediapipe as mp
import time
import pyautogui

wCam, hCam = 640, 480

capture = cv2.VideoCapture(0)  # which webcam to use
capture.set(3, wCam)
capture.set(4, hCam)

# Screen resolution
screen_width, screen_height = pyautogui.size()

x_pokazalec = y_pokazalec = 0
x_palec = y_palec = 0
x_sreden = y_sreden = 0
x_pinky = y_pinky = 0

prev_dist = 0

gesture_text = ""

if not capture.isOpened():  # check if camera running
    print("Error: Could not open webcam.")
    exit()

mpHands = mp.solutions.hands  # to use the model
myHands = mpHands.Hands(min_detection_confidence=0.7,
                        min_tracking_confidence=0.7)  # the default values are False, 2, 0.5, 0.5
mpDraw = mp.solutions.drawing_utils

# FPS
prev_time = 0
curr_time = 0

# thresholds for detecting finger state
OPEN_HAND_THRESHOLD = 80
FINGER_DOWN_THRESHOLD = 60
FIST_THRESHOLD = 30

# cooldown in seconds
COOLDOWN_PERIOD = 3
last_click = 0

prev_y_center = None


def is_hand_open(landmarks, width, height):  # neutral position
    base_x, base_y = int(landmarks[0].x * width), int(landmarks[0].y * height)
    open_fingers = 0

    # Check distance for each fingertip from the base of the palm (wrist)
    for fingertip_id in [4, 8, 12, 16, 20]:  # thumb, index, middle, ring, pinky
        tip_x, tip_y = int(landmarks[fingertip_id].x * width), int(landmarks[fingertip_id].y * height)
        distance = ((tip_x - base_x) ** 2 + (tip_y - base_y) ** 2) ** 0.5
        if distance > OPEN_HAND_THRESHOLD:
            open_fingers += 1

    # If all five fingers are sufficiently away from the palm base, consider the hand open
    return open_fingers == 5


def is_finger_down(landmark, base_x, base_y, width, height):
    finger_x, finger_y = int(landmark.x * width), int(landmark.y * height)
    distance = ((finger_x - base_x) ** 2 + (finger_y - base_y) ** 2) ** 0.5
    return distance < FINGER_DOWN_THRESHOLD


def calculate_distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

# Mouse Moving Mode - 2 fingers up
# Mouse Single Left Click Mode - index down
# Mouse Single Right Click Mode - middle down
# Mouse Double Click Mode - both fingers down


while True:
    success, img = capture.read()  # frame captured
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # covert to RGB
    results = myHands.process(imgRGB)
    # print(results.multi_hand_landmarks)  # value None if no hand in frame
    hands = results.multi_hand_landmarks

    # gesture_text = ""  # Reset gesture text

    if hands:
        if len(hands) == 1:  # if there is 1 hand available
            hand_landmarks = hands[0]
            height, width, channels = img.shape

            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

            # Check if hand is open
            # hand_open = is_hand_open(hand_landmarks.landmark, width, height)
            base_x, base_y = int(hand_landmarks.landmark[0].x * width), int(hand_landmarks.landmark[0].y * height)

            for id, landmark in enumerate(hand_landmarks.landmark):
                # print(id, landmark)  # each landmark(point) has an id and x,y,z values
                height, width, channels = img.shape
                center_x = int(landmark.x * width)
                center_y = int(landmark.y * height)

                # print(id, center_x, center_y)
                # print("-----------")

                if id == 0:  # wrist
                    cv2.circle(img, (center_x, center_y), 13, (255, 0, 255), cv2.FILLED)

                if id == 8:  # pokazalec vrv
                    cv2.circle(img, (center_x, center_y), 8, (0, 255, 0), cv2.FILLED)
                    x_pokazalec = center_x
                    y_pokazalec = center_y

                if id == 4:  # palec vrv
                    cv2.circle(img, (center_x, center_y), 8, (255, 0, 0), cv2.FILLED)
                    x_palec = center_x
                    y_palec = center_y

                if id == 12:  # sreden prst vrv
                    cv2.circle(img, (center_x, center_y), 8, (255, 0, 0), cv2.FILLED)
                    x_middle = center_x
                    y_middle = center_y

            # check finger states
            index_finger_down = is_finger_down(hand_landmarks.landmark[8], x_palec, y_palec, width, height)
            middle_finger_down = is_finger_down(hand_landmarks.landmark[12], x_palec, y_palec, width, height)
            hand_open = is_hand_open(hand_landmarks.landmark, width, height)

            # map webcam coordinates to screen coordinates and mirror along x-axis
            screen_x = screen_width - int(x_pokazalec * screen_width / wCam)
            screen_y = int(y_pokazalec * screen_height / hCam)

            current_time = time.time()

            if not hand_open:

                if not index_finger_down and not middle_finger_down:
                    # both fingers up - move the mouse
                    pyautogui.moveTo(screen_x, screen_y)
                    gesture_text = "Move Mouse"
                elif index_finger_down and not middle_finger_down and (current_time - last_click) > COOLDOWN_PERIOD:
                    # only index finger down - single left click
                    pyautogui.click(button='left')
                    print("Single Left click")
                    gesture_text = "Left Click"
                    last_click = current_time
                elif not index_finger_down and middle_finger_down and (current_time - last_click) > COOLDOWN_PERIOD:
                    # only middle finger down - single right click
                    pyautogui.click(button='right')
                    print("Single Right click")
                    gesture_text = "Right Click"
                    last_click = current_time
                elif index_finger_down and middle_finger_down and (current_time - last_click) > COOLDOWN_PERIOD:
                    # both fingers down - double click
                    pyautogui.doubleClick(button='left')
                    print("Double click")
                    gesture_text = "Double Click"
                    last_click = current_time
            else:
                gesture_text = "Open Hand"

            # Draw circles on the index fingertip and middle fingertip
            cv2.circle(img, (x_pokazalec, y_pokazalec), 8, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x_sreden, y_sreden), 8, (255, 0, 0), cv2.FILLED)

            # draw the 21 points and connections
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

        elif len(hands) == 2:
            hand_landmarks1 = hands[0]
            hand_landmarks2 = hands[1]
            height, width, channels = img.shape

            mpDraw.draw_landmarks(img, hand_landmarks1, mpHands.HAND_CONNECTIONS)
            mpDraw.draw_landmarks(img, hand_landmarks2, mpHands.HAND_CONNECTIONS)

            # with thumb
            p_x1, p_y1 = int(hand_landmarks1.landmark[4].x * width), int(hand_landmarks1.landmark[4].y * height)
            p_x2, p_y2 = int(hand_landmarks2.landmark[4].x * width), int(hand_landmarks2.landmark[4].y * height)

            hand_open_1 = is_hand_open(hand_landmarks1.landmark, width, height)
            hand_open_2 = is_hand_open(hand_landmarks2.landmark, width, height)

            dist = calculate_distance(p_x1, p_y1, p_x2, p_y2)
            cv2.line(img, (p_x1, p_y1), (p_x2, p_y2), (0, 255, 0), 3)

            if prev_dist != 0:
                # calculate the difference in distance
                dist_change = dist - prev_dist

                # adjust volume based on distance change
                if dist_change > 0:
                    for _ in range(int(dist_change // 10)):  # Scale the change
                        pyautogui.press("volumeup")
                    gesture_text = "Volume Up"
                elif dist_change < 0:
                    for _ in range(int(-dist_change // 10)):  # Scale the change
                        pyautogui.press("volumedown")
                    gesture_text = "Volume Down"

            prev_dist = dist

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # black border
    cv2.putText(img, str(int(fps)), (30, 60), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 4, cv2.LINE_AA)
    # white text
    cv2.putText(img, str(int(fps)), (30, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "( press 'q' to quit )", (20, 90), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 1, cv2.LINE_AA)

    # display gesture text
    cv2.putText(img, gesture_text, (30, 130), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('The image', img)
    if cv2.waitKey(1) == ord('q'):  # exit loop by pressing 'q'
        break  # 27 for the escape btn

# release resources
cv2.destroyAllWindows()
capture.release()
