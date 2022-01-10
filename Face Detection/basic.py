import mediapipe as mp
import cv2
import time

cap = cv2.VideoCapture('videos/6.mp4')
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75, 1)

def ResizeWithAspectRatio(image, width=None, height=0, inter=cv2.INTER_LINEAR):

    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:

        return image

    if width is None:

        r = height / float(h)
        dim = (int(w * r), height)

    else:
        # height is None
        r = width / float(w)
        dim = (width, int(h*r))

    return cv2.resize(image, dim, interpolation=inter)



while True:
    success, img = cap.read()
    img = ResizeWithAspectRatio(img, height=700)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):

            mpDraw.draw_detection(img, detection)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxC= detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20),
                                                                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)