from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator, Colors

# Tải mô hình YOLO
model = YOLO('yolov8n.pt')

# Mở camera
cap = cv2.VideoCapture(0)

# Khởi tạo Colors
colors = Colors()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc khung hình từ camera.")
        break

    # Dự đoán đối tượng
    results = model.predict(source=frame, show=False)

    # Kiểm tra xem có đối tượng nào được phát hiện không
    if len(results[0].boxes) > 0:
        # Khởi tạo Annotator
        annotator = Annotator(frame, line_width=2, font_size=10)

        # Vẽ bounding boxes lên khung hình
        for box, cls_id, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
            box = box.cpu().numpy().astype(int)
            cls_id = int(cls_id)
            conf = float(conf)
            color = colors(cls_id)
            label = f"{model.names[cls_id]} {conf:.2f}"  # thêm tên + độ tin cậy
            annotator.box_label(box, label=label, color=color)

        # Lấy khung hình đã được chú thích
        frame = annotator.result()

    # Hiển thị khung hình
    cv2.imshow("YOLO", frame)

    # Thoát khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()