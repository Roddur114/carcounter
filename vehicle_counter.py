from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture(
    "C:/Users/RODDUR GHOSH/Desktop/object_detection/test_vid.mp4")
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("C:/Users/RODDUR GHOSH/Desktop/object_detection/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("C:/Users/RODDUR GHOSH/Desktop/object_detection/mask2.png")

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits1 = [40, 550, 570, 550]
limits2 = [680, 450, 990, 450]

incomingCount = []
outgoingCount = []
while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    for rec in results:
        boxes = rec.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(box.cls[0])
            conf = math.ceil((box.conf[0]*100))/100
            font = cv2.FONT_HERSHEY_PLAIN
            color_for_bbox = (0, 0, 255)
            currentClass = classNames[cls]

            # if currentClass == "car" or currentClass == "bus" or currentClass == "truck" or currentClass == "motorbike":
            # color_for_bbox = (0, 0, 255)
            # cv2.rectangle(img, (x1, y1), (x2, y2),
            #               color=color_for_bbox, thickness=3)
            # cvzone.putTextRect(img, f'{classNames[cls]}', (max(0, x1), max(
            #     35, y1)), scale=1, thickness=1, colorT=(255, 255, 255), colorB=color_for_bbox, colorR=color_for_bbox, offset=5)
            currentArray = np.array([x1, y1, x2, y2, conf])
            detections = np.vstack((detections, currentArray))

            # w, h = x2-x1, y2-y1
            # print(x1, y1, x2, y2)
            # print(conf)
            # cvzone.cornerRect(img, (x1, y1, w, h))
            # cvzone.putTextRect(
            #     img, f"{classNames[cls]}-{conf}", (max(0, x1), max(35, y1)))
    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits1[0], limits1[1]), (limits1[2],
             limits1[3]), color=(255, 0, 0), thickness=2)
    cv2.line(img, (limits2[0], limits2[1]), (limits2[2],
             limits2[3]), color=(255, 255, 0), thickness=2)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        if currentClass == "car" or currentClass == "bus" or currentClass == "truck" or currentClass == "motorbike":

            cv2.rectangle(img, (x1, y1, w, h),
                          color=color_for_bbox, thickness=2)
            cvzone.putTextRect(img, f' {currentClass}', (max(0, x1), max(35, y1)),
                               scale=2, thickness=1, colorR=(0, 0, 255), offset=5)

            cx, cy = x1+w//2, y1+h//2
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            if limits1[0] < cx < limits1[2] and limits1[1]-15 < cy < limits1[3]+15:
                if incomingCount.count(id) == 0:
                    incomingCount.append(id)
                    # print(incomingCount)
                    cv2.line(img, (limits1[0], limits1[1]), (limits1[2],
                                                             limits1[3]), color=(0, 255, 0), thickness=2)

            if limits2[0] < cx < limits2[2] and limits2[1]-15 < cy < limits2[3]+15:
                if outgoingCount.count(id) == 0:
                    outgoingCount.append(id)
                    # print(incomingCount)
                    cv2.line(img, (limits2[0], limits2[1]), (limits2[2],
                                                             limits2[3]), color=(0, 255, 0), thickness=2)
    # cv2.putText(img, str(len(incomingCount)), (255, 100),
                # cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    cvzone.putTextRect(img, f'Total count:{len(incomingCount)+len(
        outgoingCount)}', (50, 50), font=cv2.FONT_HERSHEY_COMPLEX, thickness=1, colorT=(255, 255, 255), colorR=(0, 0, 220), scale=0.9)
    cvzone.putTextRect(img, f'Incoming:{len(incomingCount)}', (50, 95), font=cv2.FONT_HERSHEY_COMPLEX, thickness=1, colorT=(
        255, 255, 255), colorR=(0, 0, 220), scale=0.9)
    cvzone.putTextRect(img, f'Outgoing:{len(outgoingCount)}', (50, 140), font=cv2.FONT_HERSHEY_COMPLEX, thickness=1, colorT=(
        255, 255, 255), colorR=(0, 0, 220), scale=0.9)
    cv2.imshow("image", img)
    cv2.waitKey(1)
