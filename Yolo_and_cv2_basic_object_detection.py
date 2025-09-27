from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator, Colors

# Download YOLO model
model = YOLO('yolov8n.pt')

# Open device's camera
cap = cv2.VideoCapture(0)

# Colors create
colors = Colors()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc khung hình từ camera.")
        break

    # Object prediction
    results = model.predict(source=frame, show=False)

    if len(results[0].boxes) > 0:
        annotator = Annotator(frame, line_width=2, font_size=10)

        for box, cls_id, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
            box = box.cpu().numpy().astype(int)
            cls_id = int(cls_id)
            conf = float(conf)
            color = colors(cls_id)
            label = f"{model.names[cls_id]} {conf:.2f}"  # thêm tên + độ tin cậy
            annotator.box_label(box, label=label, color=color)

        frame = annotator.result()

    # Show the frame
    cv2.imshow("YOLO", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
