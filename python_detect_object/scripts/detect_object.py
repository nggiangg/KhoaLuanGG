#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage
import cv2
import time

# Import từ các module tự tạo
from model_utils import initialize_model
from image_utils import decode_image, detect_objects
from visualization import draw_detections, calculate_fps, display_info

def callback(msg):
    """Hàm callback chính xử lý mỗi frame"""
    # Bắt đầu đo thời gian
    current_time = time.time()
    
    # Giải nén ảnh từ ROS message
    image_np = decode_image(msg.data)
    if image_np is None:
        rospy.logwarn("Không thể giải mã ảnh!")
        return

    # Thực hiện detection - KHÔNG khởi tạo lại model
    detections = detect_objects(image_np)  # Sử dụng model toàn cục
    
    # Vẽ kết quả detection
    if len(detections) > 0:
        image_np = draw_detections(image_np, detections)
    
    # Tính toán FPS
    fps = calculate_fps(current_time)
    
    # Hiển thị thông tin
    image_np = display_info(image_np, fps)

    # Hiển thị kết quả
    cv2.imshow("DetectNet", image_np)
    cv2.waitKey(1)

def main():
    """Hàm chính của chương trình"""
    rospy.init_node('ros_detectnet_node')
    
    # Lấy tham số từ ROS Parameter Server
    model_path = rospy.get_param('~model_path', None)
    labels_path = rospy.get_param('~labels_path', None)
    threshold = rospy.get_param('~threshold', 0.5)
    camera_topic = rospy.get_param('~camera_topic', '/csi_cam_0/image_raw/compressed')
    
    # Khởi tạo model
    model = initialize_model(model_path, labels_path, threshold)
    
    # Chia sẻ model với module image_utils
    from image_utils import set_model
    set_model(model)
    
    # Đăng ký subscriber
    rospy.Subscriber(camera_topic, CompressedImage, callback, queue_size=1, buff_size=2**24)
    rospy.loginfo(f"ROS detectNet node started, subscribing to {camera_topic}")
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
