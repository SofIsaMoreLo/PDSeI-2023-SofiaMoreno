import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from sort import Sort

BLUE_LINE = [(160, 190), (480, 190)]
GREEN_LINE = [(135, 220), (550, 220)]
RED_LINE = [(110, 250), (575, 250)]

if __name__ =='__main__':

    cap = cv2.VideoCapture(0)
    model = YOLO("best.pt")
    tracker = Sort()

    while cap.isOpened():
        status, frame = cap.read()
        
        if not status:
            break
        
        results = model(frame, stream=True)

        for res in results:
            indice_filtro = np.where((np.isin(res.boxes.cls.cpu().numpy(), [0,1,2,4,6,7,8])) & (res.boxes.conf.cpu().numpy() > 0.5)) [0]
            boxes = res.boxes.xyxy.cpu().numpy() [indice_filtro].astype(int)

            tracks = tracker.update(boxes)
            tracks = tracks.astype(int)

            for xmin, ymin, xmax, ymax, track_id in tracks:
                cv2.rectangle(img=frame, pt1=(xmin,ymin), pt2=(xmax, ymax), color=(255, 255, 0), thickness=2)

        cv2.line(frame, BLUE_LINE[0], BLUE_LINE[1], (255,0,0), 3)
        cv2.line(frame, GREEN_LINE[0], GREEN_LINE[1], (0,255,0), 3)
        cv2.line(frame, RED_LINE[0], RED_LINE[1], (0,0,255), 3)
        
        cv2.imshow("frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("x"):
            break
cap.release()