#import library
import cv2
import numpy as np

#load video
cctv = cv2.VideoCapture("Traffic.mp4")

width = 80
height = 80
count_line = 550

#initialize substractor
model = cv2.bgsegm.createBackgroundSubtractorMOG()



def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

detection = []
offset = 6
counter = 0

#object detection
while True:
    retV, frame = cctv.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 5)

    #apply function to frame
    frame_sub = model.apply(blur)
    dilated = cv2.dilate(frame_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilat = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    dilat = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    contour = cv2.findContours(dilat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame, (25, count_line), (1200, count_line), (255, 127, 0), 3)

    #define function for contour coordinates
    for (i, c) in enumerate(contour[0]):
        (x, y, w, h) = cv2.boundingRect(c)
        val_counter = (w >= width) and (h >= height)

        if not val_counter:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Vehicle" + str(counter), (x, y -20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 244, 0), 2)

        center = center_handle(x, y, w, h)
        detection.append(center)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)

        #calculate the object when hit the line
        for (x, y) in detection:
            if y < (count_line + offset) and y > (count_line - offset):
                counter += 1
                cv2.line(frame, (25, count_line), (1200, count_line), (0, 127, 255), 3)
                detection.remove((x, y))
                print("Count of Vehicle :" + str(counter))

    cv2.putText(frame, "COUNT OF VEHICLE :" + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    cv2.imshow("Traffic Supervision", frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

cctv.release()
cv2.destroyAllWindows()
