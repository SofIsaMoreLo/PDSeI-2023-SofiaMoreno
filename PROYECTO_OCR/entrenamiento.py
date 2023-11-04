from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data = "D:\Procesamiento\PDSeI-2023-SofiaMoreno\PROYECTO_OCR\config.yaml" , epochs=50)
