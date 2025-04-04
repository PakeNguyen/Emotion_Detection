# Nhận diện cảm xúc khuôn mặt với YOLOv5 và EfficientNet-B2

Dự án này là một hệ thống **Nhận diện cảm xúc khuôn mặt theo thời gian thực**, sử dụng **YOLOv5** để phát hiện khuôn mặt và **EfficientNet-B2** để phân loại cảm xúc. Mô hình có thể phát hiện các cảm xúc như **Tức giận**, **Khinh bỉ**, **Kinh tởm**, **Sợ hãi**, **Vui**, **Bình thường**, **Buồn**, và **Ngạc nhiên** từ các video webcam trực tiếp hoặc ảnh tĩnh.

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
