#!/usr/bin/env python3
#import jetson_utils
import jetson.inference
import rospy

# Khởi tạo biến toàn cục
net = None

def initialize_model(model_path=None, labels_path=None, threshold=0.5):
    """Khởi tạo model detectNet"""
    global net
    
    # Sử dụng đường dẫn mặc định nếu không được chỉ định
    if model_path is None:
        model_path = "/home/jetson/jetson-inference/python/training/detection/ssd/models/ViDu/mb2-ssd-lite.onnx"
    if labels_path is None:
        labels_path = "/home/jetson/jetson-inference/python/training/detection/ssd/models/ViDu/labels.txt"
    
    # Khởi tạo model
    net = jetson.inference.detectNet(
        model=model_path,
        labels=labels_path,
        input_blob="input_0",
        output_cvg="scores",
        output_bbox="boxes",
        threshold=threshold
    )
    
    return net

def get_class_name(class_id):
    """Lấy tên lớp từ class ID"""
    global net
    if net is not None:
        return net.GetClassDesc(class_id)
    return "Unknown"
