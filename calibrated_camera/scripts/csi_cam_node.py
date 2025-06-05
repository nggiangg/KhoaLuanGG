#!/usr/bin/env python3
# filepath: ~/catkin_ws/src/calibrated_camera/scripts/csi_cam_node.py

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge, CvBridgeError

class CSICameraNode:
    def __init__(self):
        # Khởi tạo ROS node
        rospy.init_node('csi_cam_node', anonymous=True)
        
        # Lấy tham số từ ROS Parameter Server
        self.camera_id = rospy.get_param('~camera_id', 0)
        self.width = rospy.get_param('~width', 640)
        self.height = rospy.get_param('~height', 480)
        self.frame_rate = rospy.get_param('~frame_rate', 30)
        self.use_gstreamer = rospy.get_param('~use_gstreamer', True)
        self.apply_calibration = rospy.get_param('~apply_calibration', True)
        
        # Thông số calibration (từ kết quả calibration trước đó)
        self.camera_matrix = np.array([
            [381.457376, 0.000000, 313.222494],
            [0.000000, 501.065312, 193.775713],
            [0.000000, 0.000000, 1.000000]
        ])
        self.dist_coeffs = np.array([-0.318161, 0.088749, 0.000998, 0.001712, 0.000000])
        
        # Khởi tạo bridge để chuyển đổi giữa OpenCV và ROS
        self.bridge = CvBridge()
        
        # Khởi tạo publishers với tên topic mới
        self.image_pub = rospy.Publisher('/csi_cam_0/calibrated_camera', Image, queue_size=1)
        self.compressed_pub = rospy.Publisher('/csi_cam_0/calibrated_camera/compressed', 
                                             CompressedImage, queue_size=1)
        self.camera_info_pub = rospy.Publisher('/csi_cam_0/calibrated_camera_info', 
                                              CameraInfo, queue_size=1)
        
        # Tạo camera info message
        self.camera_info_msg = CameraInfo()
        self.camera_info_msg.header.frame_id = "csi_cam_0"
        self.camera_info_msg.height = self.height
        self.camera_info_msg.width = self.width
        self.camera_info_msg.distortion_model = "plumb_bob"
        self.camera_info_msg.D = self.dist_coeffs.tolist()
        self.camera_info_msg.K = self.camera_matrix.flatten().tolist()
        self.camera_info_msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.camera_info_msg.P = [
            281.401428, 0.000000, 313.712444, 0.000000,
            0.000000, 455.640442, 185.818546, 0.000000,
            0.000000, 0.000000, 1.000000, 0.000000
        ]
        
        # Khởi tạo camera
        self.init_camera()
        
        # Tạo timer để publish frame
        self.timer = rospy.Timer(rospy.Duration(1.0/self.frame_rate), self.timer_callback)
        
        rospy.loginfo("CSI camera node initialized and publishing to /csi_cam_0/calibrated_camera")
        
    def init_camera(self):
        """Khởi tạo camera với phương pháp phù hợp"""
        if self.use_gstreamer:
            # Sử dụng CSI camera với GStreamer (tốt nhất cho Jetson)
            gst_str = (f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int){self.width}, "
                      f"height=(int){self.height}, format=(string)NV12, framerate=(fraction){self.frame_rate}/1 ! "
                      f"nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! "
                      f"video/x-raw, format=(string)BGR ! appsink")
            self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
            if self.cap.isOpened():
                rospy.loginfo("CSI camera opened with GStreamer pipeline")
                return
            else:
                self.cap.release()
                rospy.logwarn("Failed to open CSI camera with GStreamer, trying USB camera")
        
        # Thử với USB camera
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.frame_rate)
        
        if not self.cap.isOpened():
            rospy.logerr("Failed to open camera with any method!")
            rospy.signal_shutdown("Failed to open camera")
    
    def timer_callback(self, event):
        """Callback để đọc frame và publish"""
        if not self.cap.isOpened():
            rospy.logwarn("Camera is not open")
            return
        
        ret, frame = self.cap.read()
        if not ret:
            rospy.logwarn("Failed to read frame")
            return
        
        # Áp dụng calibration nếu được yêu cầu
        if self.apply_calibration:
            frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        
        # Tạo timestamp
        now = rospy.Time.now()
        
        try:
            # Publish raw image
            img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            img_msg.header.stamp = now
            img_msg.header.frame_id = "csi_cam_0"
            self.image_pub.publish(img_msg)
            
            # Publish compressed image
            compressed_msg = CompressedImage()
            compressed_msg.header.stamp = now
            compressed_msg.header.frame_id = "csi_cam_0"
            compressed_msg.format = "jpeg"
            compressed_msg.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
            self.compressed_pub.publish(compressed_msg)
            
            # Publish camera info
            self.camera_info_msg.header.stamp = now
            self.camera_info_pub.publish(self.camera_info_msg)
            
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Error publishing image: {e}")
    
    def shutdown(self):
        """Hàm dọn dẹp khi node dừng"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'timer'):
            self.timer.shutdown()
        rospy.loginfo("CSI camera node shutdown")

if __name__ == '__main__':
    try:
        node = CSICameraNode()
        rospy.on_shutdown(node.shutdown)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
