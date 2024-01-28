from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data = "config1.yaml" , epochs=15, batch = 64, imgsz=640)

