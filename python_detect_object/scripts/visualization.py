#!/usr/bin/env python3
import cv2
import time
import rospy
from model_utils import get_class_name

# Biến toàn cục để tính FPS
prev_time = 0
fps_filter = 0

def draw_detections(image_np, detections, confidence_threshold=0.6):
    """Vẽ các bounding box lên ảnh"""
    if len(detections) == 0:
        return image_np
        
    for det in detections:
        x1, y1, x2, y2 = int(det.Left), int(det.Top), int(det.Right), int(det.Bottom)
        # Chỉ vẽ rectangle, bỏ text để tăng tốc
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0,255,0), 1)
        
        # Chỉ hiển thị text cho các object có confidence cao
        if det.Confidence > confidence_threshold:
            class_name = get_class_name(det.ClassID)
            # Đơn giản hóa text - chỉ hiển thị tên class
            cv2.putText(image_np, class_name, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    
    return image_np

def calculate_fps(current_time):
    """Tính toán và cập nhật FPS"""
    global prev_time, fps_filter
    
    fps = 1.0 / (current_time - prev_time) if prev_time > 0 else 30.0
    prev_time = current_time
    fps_filter = 0.9 * fps_filter + 0.1 * fps
    
    return fps_filter

def display_info(image_np, fps):
    """Hiển thị thông tin FPS lên ảnh"""
    cv2.putText(image_np, f"FPS: {fps:.1f}", (10, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    return image_np
