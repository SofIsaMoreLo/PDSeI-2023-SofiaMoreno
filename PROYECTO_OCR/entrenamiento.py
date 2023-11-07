from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data = "config.yaml" , epochs=50, imgsz=640)

