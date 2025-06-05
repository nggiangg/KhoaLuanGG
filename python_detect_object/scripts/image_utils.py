#!/usr/bin/env python3
import cv2
import numpy as np
import jetson.utils

# Biến toàn cục cho việc theo dõi detection
frame_count = 0
last_detections = []
# Thêm vào file image_utils.py
def set_model(model_instance):
    """Cập nhật biến net từ bên ngoài"""
    global net
    net = model_instance

def decode_image(compressed_data):
    """Giải nén ảnh từ ROS message dạng compressed"""
    np_arr = np.frombuffer(compressed_data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image_np

def detect_objects(image_np, force_detection=False):
    """Thực hiện nhận diện đối tượng"""
    global frame_count, last_detections, net
    
    # Kiểm tra model đã được khởi tạo chưa
    if net is None:
        rospy.logerr("Model chưa được khởi tạo!")
        return []
    
    # Tăng frame counter
    frame_count += 1
    
    # Xác định có cần detection không
    detection_frame = force_detection or (frame_count % 3 == 0)
    
    if detection_frame:
        # Chuyển sang RGBA cho jetson-inference
        image_rgba = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGBA)
        cuda_img = jetson.utils.cudaFromNumpy(image_rgba)

        # Nhận diện đối tượng
        last_detections = net.Detect(cuda_img)
        
    return last_detections
