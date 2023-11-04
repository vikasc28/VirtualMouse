import cv2
import numpy as np
import HandTrackingModule as handtracker
import time
import autopy

wCam, hCam = 640, 480
frameR = 100
smoothening = 7

prev_time = 0
plotX, plotY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = handtracker.HandDetector()
wScr, hScr = autopy.screen.size()


while True:
    success, img = cap.read()
    img = detector.find_hands(img=img, draw=True)
    lm_list, allowed_border = detector.find_position(img=img)
    if len(lm_list) != 0:
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]

        fingers = detector.fingers_up()
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            clocX = plotX + (x3 - plotX) / smoothening
            clocY = plotY + (y3 - plotY) / smoothening

            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plotX, plotY = clocX, clocY

        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.find_distance(8, 12, img=img)
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),
                           15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # Calculate fps
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    # 12. Display
    cv2.imshow("Air-Mouse", img)
    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q') or cv2.getWindowProperty("Air-Mouse", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
