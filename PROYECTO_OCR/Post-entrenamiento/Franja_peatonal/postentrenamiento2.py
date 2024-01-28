from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data = "config2.yaml" , epochs=20, batch = 64, imgsz=640)

