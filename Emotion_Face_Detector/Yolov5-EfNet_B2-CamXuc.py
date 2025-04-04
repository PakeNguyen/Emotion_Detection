import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b2
import torch.nn as nn
import sys
sys.path.append("C:/Hoc_May/All_Project/Predict_CamXuc/yolov5_face")
from yolov5_face.utils.datasets import letterbox
from yolov5_face.utils.general import non_max_suppression, scale_coords


# === Cấu hình đường dẫn ===
YOLO_PATH = "C:/Hoc_May/All_Project/Predict_CamXuc/yolov5s-face.pt"
EFFNET_PATH = "C:/Hoc_May/All_Project/Predict_CamXuc/EfNet_checkpoint/efficientnet_b2/best.pt"

# === Cấu hình thiết bị ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load YOLOv5-face ===
yolo = torch.load(YOLO_PATH, map_location=device)['model'].float().fuse().eval()

# === Load EfficientNet-B2 (dự đoán cảm xúc) ===
eff_model = efficientnet_b2(weights=None)
eff_model.classifier[1] = nn.Linear(eff_model.classifier[1].in_features, 8)
eff_model.load_state_dict(torch.load(EFFNET_PATH, map_location=device)["model_state_dict"])
eff_model.to(device).eval()

# === Nhãn cảm xúc (Tiếng Việt nếu muốn) ===
emotion_labels = ["Tuc Gian", "Khinh bi", "Kinh Tom", "So Hai", "Vui", "Binh Thuong", "Buon", "Ngac Nhien"]

# === Transform cho EfficientNet-B2 ===
transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Mở camera ===
cap = cv2.VideoCapture(0)  # 0 = webcam mặc định

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img0 = frame.copy()
    img = letterbox(img0, new_shape=640)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        pred = yolo(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)[0]

    boxes = []
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], img0.shape).round()

        for *xyxy, conf, cls in pred:
            x1, y1, x2, y2 = map(int, xyxy)
            if (x2 - x1 < 30) or (y2 - y1 < 30):
                continue

            too_close = False
            for bx1, by1, bx2, by2 in boxes:
                inter_area = max(0, min(x2, bx2) - max(x1, bx1)) * max(0, min(y2, by2) - max(y1, by1))
                union_area = (x2 - x1) * (y2 - y1) + (bx2 - bx1) * (by2 - by1) - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                if iou > 0.5:
                    too_close = True
                    break
            if too_close:
                continue
            boxes.append((x1, y1, x2, y2))

    for x1, y1, x2, y2 in boxes:
        face_crop = img0[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = eff_model(face_tensor)
            pred_class = torch.argmax(output, dim=1).item()
            emotion = emotion_labels[pred_class]

        cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{emotion} {conf*100:.2f}%"  # ← thêm conf tại đây
        cv2.putText(img0, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Nhận diện cảm xúc realtime", img0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
