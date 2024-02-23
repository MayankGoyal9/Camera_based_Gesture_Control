import math
import cv2
import numpy as np
import mediapipe as mp
import time
import pyautogui


class HandDetector:
    def __init__(self, mode=False, maxh=2, complexity=1, detection=0.5, track=0.5):
        self.mode = mode
        self.maxHands = maxh
        self.complexity = complexity
        self.detectionCon = detection
        self.trackCon = track

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        self.results = None
        self.land_mark_list = []

    def findHands(self, img, draw=True):
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks and draw:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def getLandmarkList(self, img, hand_no=0, draw=True):
        self.land_mark_list = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_no]
            for lm_id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.land_mark_list.append([lm_id, cx, cy])

                if draw and (lm_id == 4 or lm_id == 8 or lm_id == 12 or lm_id == 16 or lm_id == 20):
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        return self.land_mark_list

    def findDistance(self, lm1, lm2):
        x1, y1 = self.land_mark_list[lm1][1], self.land_mark_list[lm1][2]
        x2, y2 = self.land_mark_list[lm2][1], self.land_mark_list[lm2][2]
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    @staticmethod
    def getFingersUp(lmlist):
        tipid = [4, 8, 12, 16, 20]
        fingers = []

        if lmlist[tipid[0]][1] < lmlist[tipid[0] - 2][2]:
            fingers.append(True)
        else:
            fingers.append(False)

        for id in range(1, 5):
            if lmlist[tipid[id]][2] < lmlist[tipid[id] - 2][2]:
                fingers.append(True)
            else:
                fingers.append(False)

        return fingers


def main():
    ptime = 0
    ctime = 0

    wCam, hCam = 640, 480
    wScreen, hScreen = pyautogui.size()
    frameR = 100
    smooth = 2

    prev_x, prev_y = 0, 0
    curr_x, curr_y = 0, 0

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    detector = HandDetector(maxh=1)

    while True:

        _, img = cap.read()

        img = detector.findHands(img, draw=False)
        lmlist = detector.getLandmarkList(img, draw=False)

        if len(lmlist) != 0:
            index_x, index_y = lmlist[8][1:]

            fingers = detector.getFingersUp(lmlist)
            # print(fingers)

            # moving mode
            if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (0, 255, 0), 1)

                screen_x = np.interp(index_x, (frameR, wCam - frameR), (0, wScreen))
                screen_y = np.interp(index_y, (frameR, hCam - frameR), (0, hScreen))

                curr_x = prev_x + (screen_x - prev_x) / smooth
                curr_y = prev_y + (screen_y - prev_y) / smooth

                pyautogui.moveTo(curr_x, curr_y)

                prev_x, prev_y = curr_x, curr_y

                dist = detector.findDistance(4, 5)
                if dist < 35:
                    print(dist)
                    pyautogui.click()

        # if len(lmlist) != 0 and detector.findDistance(4, 8) < 20:
        #    print("Near", lmlist[4], lmlist[8])

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()