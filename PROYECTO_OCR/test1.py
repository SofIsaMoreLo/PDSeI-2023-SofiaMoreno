from ultralytics import YOLO
import cv2

model = YOLO('best.pt')

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    resultados = model.predict(frame)

    anotaciones = resultados[0].plot()

    cv2.imshow('Detector', anotaciones)

    if cv2.waitKey(5) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()