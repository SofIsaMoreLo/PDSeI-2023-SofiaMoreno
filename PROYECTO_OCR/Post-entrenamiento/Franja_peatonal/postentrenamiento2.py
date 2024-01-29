from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data = "Post-entrenamiento\Franja_peatonal\config2.yaml" , epochs=20, imgsz=640)

