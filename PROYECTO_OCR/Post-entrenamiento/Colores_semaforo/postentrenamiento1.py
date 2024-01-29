from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data = "Post-entrenamiento\Colores_semaforo\config1.yaml" , epochs=15, imgsz=640)

