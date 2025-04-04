# Nhận diện cảm xúc khuôn mặt với YOLOv5 và EfficientNet-B2


## Dữ liệu
Dự án này là một hệ thống **Nhận diện cảm xúc khuôn mặt theo thời gian thực**, sử dụng **YOLOv5** để phát hiện khuôn mặt và **EfficientNet-B2** để phân loại cảm xúc. Mô hình có thể phát hiện các cảm xúc như **Tức giận**, **Khinh bỉ**, **Kinh tởm**, **Sợ hãi**, **Vui**, **Bình thường**, **Buồn**, và **Ngạc nhiên** từ các video webcam trực tiếp hoặc ảnh tĩnh.

## Link tải dữ liệu:
```bash
https://drive.google.com/drive/folders/1ODKTmCffALZ2gy4BQPOXG0gqxqu36GYg
```

## Các tính năng:
- **YOLOv5** phát hiện khuôn mặt theo thời gian thực.
- **EfficientNet-B2** phân loại cảm xúc khuôn mặt.
- Hỗ trợ 8 cảm xúc: **Tức giận**, **Khinh bỉ**, **Kinh tởm**, **Sợ hãi**, **Vui**, **Bình thường**, **Buồn**, **Ngạc nhiên**.
- Nhận diện cảm xúc thời gian thực từ webcam.
- Hệ thống có thể được sử dụng trong các ứng dụng như **Tương tác người-máy**, **Giám sát sức khỏe tâm lý**, và **Nâng cao trải nghiệm người dùng**.

## Cấu trúc dự án:

- **Classifier-Effnet_B2/**: Chứa mã của mô hình EfficientNet-B2 cho việc phân loại cảm xúc, bao gồm dataset và các file checkpoint của mô hình.
  - **Dataset_CamXuc.py**: Lớp dataset sử dụng cho huấn luyện và kiểm tra mô hình.
  - **EfficientNet_B2.py**: Script huấn luyện mô hình EfficientNet-B2.
  - **EfNet_checkpoint/**: Thư mục lưu trữ các checkpoint của mô hình.
  - **EfNet_tensorboard**: Thư mục lưu trữ visualize của mô hình.

- **Emotion_Face_Detector/**: Chứa mã phát hiện khuôn mặt và kết hợp nhận diện cảm xúc.
  - **yolov5_face/**: Submodule chứa mã YOLOv5 để phát hiện khuôn mặt.
  - **yolov5s-face.pt**: Mô hình YOLOv5 đã huấn luyện để phát hiện khuôn mặt.
  - **Yolov5-EfNet_B2-CamXuc.py**: Script kết hợp YOLOv5 và EfficientNet-B2 để nhận diện cảm xúc từ khuôn mặt.
  
## Yêu cầu:
- Python 3.x
- PyTorch
- OpenCV
- torchvision
- Pillow
- numpy
- tqdm
- scikit-learn
- matplotlib
- TensorBoard (cho việc ghi log và trực quan hóa)

## Thiết lập:
1. Clone repository:
```bash
git clone https://github.com/deepcam-cn/yolov5-face.git
```

2. Tải mô hình YOLOv5 face:

Nếu bạn không có file yolov5s-face.pt, bạn có thể tải từ repository YOLOv5 Face chính thức.
Ví dụ ở đây tôi clone từ (https://github.com/deepcam-cn/yolov5-face)


3. Tải mô hình EfficientNet-B2:

Nếu bạn không có mô hình đã huấn luyện EfficientNet-B2, bạn có thể huấn luyện nó bằng script EfficientNet_B2.py hoặc sử dụng mô hình best.pt đã huấn luyện sẵn.


4. Chạy hệ thống nhận diện cảm xúc thời gian thực:

Sau khi thiết lập xong, bạn có thể chạy hệ thống nhận diện khuôn mặt và cảm xúc từ webcam:
```bash
python Yolov5-EfNet_B2-CamXuc.py
```
Hệ thống sẽ phát hiện khuôn mặt, phân loại cảm xúc và hiển thị kết quả trên màn hình webcam.

## Huấn luyện mô hình:

Nếu bạn muốn huấn luyện mô hình EfficientNet-B2 trên bộ dữ liệu của riêng mình, bạn có thể sử dụng script EfficientNet_B2.py. Đảm bảo bộ dữ liệu của bạn có cấu trúc như sau:

```markdown
dataset_classification/
├── train/
│   ├── Anger/
│   ├── Contempt/
│   ├── Disgust/
│   ├── Fear/
│   ├── Happy/
│   ├── Neutral/
│   ├── Sad/
│   └── Surprise/
└── valid/
    ├── Anger/
    ├── Contempt/
    ├── Disgust/
    ├── Fear/
    ├── Happy/
    ├── Neutral/
    ├── Sad/
    └── Surprise/
```

Dataset tôi đã để liên kết ở phía trên.

Sau đó, bạn có thể huấn luyện mô hình bằng lệnh:

```bash
python Classifier-Effnet_B2/EfficientNet_B2.py --data_path path/to/your/dataset --epochs 100 --batch_size 16 --lr 1e-4
```

hoặc là bạn có thể vào file EfficientNet_B2.py để điền các tham số mặc định và sau đó bạn chỉ cần chạy lệnh sau:
```bash
python EfficientNet_B2.py
```
Lệnh này sẽ bắt đầu huấn luyện mô hình trên bộ dữ liệu của bạn.
