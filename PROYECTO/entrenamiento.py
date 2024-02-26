from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data="C:/Users/User/Downloads/PDSeI/PDSeI-2023-SofiaMoreno/PROYECTO_OCR/config.yaml",epochs=45, imgsz=640)
