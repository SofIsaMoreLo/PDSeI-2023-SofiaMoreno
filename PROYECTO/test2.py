import cv2
import numpy as np
#from datetime import datetime
from ultralytics import YOLO
from sort import Sort

RED_LINE = [(10, 100), (700, 100)]
BLUE_LINE = [(10, 190), (700, 190)]
GREEN_LINE_1 = [(200, 0), (200, 480)]
GREEN_LINE_2 = [(510, 0), (510, 480)]

cruza_linea_roja = {}
cruza_linea_azul = {}
cruza_linea_verde_1 = {}
cruza_linea_verde_2 = {}

if __name__ =='__main__':

    cap = cv2.VideoCapture("video_test_1.mp4")
    model = YOLO("yolov8n.pt")
    #model = YOLO("best.pt")
    tracker = Sort()

    while cap.isOpened():
        status, frame = cap.read()
        
        if not status:
            break
        
        results = model(frame, stream=True)
        

        for res1 in results:
            indice_filtro1 = np.where((np.isin(res1.boxes.cls.cpu().numpy(), [0])) & (res1.boxes.conf.cpu().numpy() > 0.1)) [0]
            boxes1 = res1.boxes.xyxy.cpu().numpy() [indice_filtro1].astype(int)

            tracks1 = tracker.update(boxes1)
            tracks1 = tracks1.astype(int)

            for xmin, ymin, xmax, ymax, track_id in tracks1:

                xc, yc = int((xmin+xmax)/2), ymin
                if track_id not in cruza_linea_roja:
                    cruza_rojo = (RED_LINE[1][0] - RED_LINE[0][0]) * (yc - RED_LINE[0][1]) - (RED_LINE[1][1] - RED_LINE[0][1]) * (xc - RED_LINE[0][0])
                    if cruza_rojo >= 0:
                        cruza_linea_roja[track_id]={
                        print("Persona detectada, tenga cuidado")                    
                        }
                cv2.circle(img=frame, center=(xc,yc), radius=5, color=(255, 255, 0), thickness=-1)
                cv2.rectangle(img=frame, pt1=(xmin,ymin), pt2=(xmax, ymax), color=(255, 255, 0), thickness=2)

        for res2 in results:
            indice_filtro2 = np.where((np.isin(res2.boxes.cls.cpu().numpy(), [1, 2, 3])) & (res2.boxes.conf.cpu().numpy() > 0.1)) [0]
            boxes2 = res2.boxes.xyxy.cpu().numpy() [indice_filtro2].astype(int)

            tracks2 = tracker.update(boxes2)
            tracks2 = tracks2.astype(int)

            for xmin, ymin, xmax, ymax, track_id in tracks2:

                xc, yc = int((xmin+xmax)/2), ymin
                if track_id not in cruza_linea_azul:
                    cruza_azul = (BLUE_LINE[1][0] - BLUE_LINE[0][0]) * (yc - BLUE_LINE[0][1]) - (BLUE_LINE[1][1] - BLUE_LINE[0][1]) * (xc - BLUE_LINE[0][0])
                    if cruza_azul >= 0:
                        cruza_linea_roja[track_id]={
                        print("Vehículo detectado, tenga cuidado")
                        }
                cv2.circle(img=frame, center=(xc,yc), radius=5, color=(255, 0, 255), thickness=-1)
                cv2.rectangle(img=frame, pt1=(xmin,ymin), pt2=(xmax, ymax), color=(255, 0, 255), thickness=2)
        
        for res3 in results:
            indice_filtro3 = np.where((np.isin(res3.boxes.cls.cpu().numpy(), [6, 7])) & (res3.boxes.conf.cpu().numpy() > 0.1)) [0]
            boxes3 = res3.boxes.xyxy.cpu().numpy() [indice_filtro3].astype(int)

            tracks3 = tracker.update(boxes3)
            tracks3 = tracks3.astype(int)

            for xmin, ymin, xmax, ymax, track_id in tracks3:

                xc, yc = int((xmin+xmax)/2), int(ymin/2)
                if track_id not in cruza_linea_verde_1 & cruza_linea_verde_2:
                    cruza_verde_1 = (GREEN_LINE_1[1][0] - GREEN_LINE_1[0][0]) * (yc - GREEN_LINE_1[0][1]) - (GREEN_LINE_1[1][1] - GREEN_LINE_1[0][1]) * (xc - GREEN_LINE_1[0][0])
                    cruza_verde_2 = (GREEN_LINE_1[1][0] - GREEN_LINE_1[0][0]) * (yc - GREEN_LINE_1[0][1]) - (GREEN_LINE_1[1][1] - GREEN_LINE_1[0][1]) * (xc - GREEN_LINE_1[0][0])
                    if cruza_verde_1 >= 0 & cruza_verde_2 >= 0 :
                        cruza_linea_verde_1[track_id]=True
                        cruza_linea_verde_2[track_id]=True
                        print("Obstaculo adelante")
                        
                cv2.circle(img=frame, center=(xc,yc), radius=5, color=(100, 0, 255), thickness=-1)
                cv2.rectangle(img=frame, pt1=(xmin,ymin), pt2=(xmax, ymax), color=(100, 0, 255), thickness=2)

        cv2.line(frame, RED_LINE[0], RED_LINE[1], (0,0,255), 3)
        cv2.line(frame, BLUE_LINE[0], BLUE_LINE[1], (255,0,0), 3)
        cv2.line(frame, GREEN_LINE_1[0], GREEN_LINE_1[1], (0,255,0), 3)
        cv2.line(frame, GREEN_LINE_2[0], GREEN_LINE_2[1], (0,255,0), 3)
                
        cv2.imshow("Deteccion", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("x"):
            break
cap.release()