import cv2
import numpy as np
import time
import PoseModule as pm

#cap = cv2.VideoCapture('videos/bicep_curl.mp4')

detector = pm.poseDetector()

count = 0
dir = 0
pTime = 0

while True:

    # success, img = cap.read()
    #img = cv2.resize(img, (1280, 720))
    img = cv2.imread('videos/up.png')
    img = detector.findPose(img, True)




    # cTime = time.time()
    # fps = 1 / (cTime - pTime)
    # pTime = cTime
    # cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,(255, 0, 0), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(10)