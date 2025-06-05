#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import math

# Biến toàn cục để lưu trữ dữ liệu scan mới nhất
latest_scan = None

# Kích thước xe (cm)
CAR_WIDTH = 20  # chiều rộng xe (cm)
CAR_LENGTH = 30  # chiều dài xe (cm)

# Khởi tạo figure
fig = plt.figure(figsize=(18, 8))

# Tạo subplot polar cho biểu đồ LiDAR dạng radar
ax_polar = fig.add_subplot(121, projection='polar')  # Chú ý projection='polar' ở đây
ax_polar.set_title('LiDAR Polar View')
ax_polar.grid(True)
scan_line, = ax_polar.plot([], [], 'r.', markersize=2)
ax_polar.set_rmax(5)  # 5 mét

# Tạo subplot Cartesian cho biểu đồ xe
ax_cart = fig.add_subplot(122)
ax_cart.set_title('LiDAR Cartesian View (Unit: m)')
ax_cart.grid(True)
ax_cart.set_xlim(-5, 5)
ax_cart.set_ylim(-5, 5)
ax_cart.set_aspect('equal')
cart_line, = ax_cart.plot([], [], 'r.', markersize=2)

# Vẽ xe trên biểu đồ cartesian
def draw_car():
    # Chuyển từ cm sang m
    width = CAR_WIDTH / 100  # 20cm -> 0.2m
    length = CAR_LENGTH / 100  # 30cm -> 0.3m
    
    # Vẽ xe (hình chữ nhật) với tâm ở gốc tọa độ
    car_rect = Rectangle((-width/2, -length/2), width, length, 
                          color='blue', alpha=0.5, label='JetRacer')
    ax_cart.add_patch(car_rect)
    
    # Vẽ tâm LiDAR (gốc tọa độ)
    ax_cart.plot(0, 0, 'go', markersize=8, label='LiDAR Center')
    
    # Vẽ mũi tên chỉ hướng phía trước của xe
    ax_cart.arrow(0, 0, 0, length/2, head_width=0.05, head_length=0.1, 
                 fc='green', ec='green', label='Front')
    
    ax_cart.legend(loc='upper right')

def scan_callback(msg):
    """Callback khi nhận dữ liệu scan mới"""
    global latest_scan
    latest_scan = msg

def update_plot(frame):
    """Cập nhật biểu đồ"""
    global latest_scan
    
    if latest_scan is None:
        return scan_line, cart_line
    
    # Lấy dữ liệu từ scan
    ranges = np.array(latest_scan.ranges)
    angles = np.linspace(latest_scan.angle_min, latest_scan.angle_max, len(ranges))
    
    # Lọc các giá trị không hợp lệ
    valid_indices = np.isfinite(ranges) & (ranges > 0) & (ranges < 10)
    valid_ranges = ranges[valid_indices]
    valid_angles = angles[valid_indices]
    
    # Cập nhật biểu đồ polar
    scan_line.set_data(valid_angles, valid_ranges)
    
    # Chuyển đổi từ polar sang cartesian
    x = valid_ranges * np.cos(valid_angles)
    y = valid_ranges * np.sin(valid_angles)
    
    # Cập nhật biểu đồ cartesian
    cart_line.set_data(x, y)
    
    return scan_line, cart_line
def main():
    # Khởi tạo node
    rospy.init_node('lidar_visualization')

    # Vẽ xe trên biểu đồ
    draw_car()

    # Đăng ký subscriber
    rospy.Subscriber('/scan', LaserScan, scan_callback, queue_size=1)

    # Giải thích vị trí LiDAR
    info_text = "- Tâm LiDAR nằm ở giữa xe (điểm xanh)\n" + \
                "- Hướng 0° của LiDAR là hướng trước của xe\n" + \
                "- Kích thước xe: 20cm × 30cm\n" + \
                "- Vùng xanh biểu thị xe JetRacer"
    plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=12,
                bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})

    # Khởi chạy animation
    ani = FuncAnimation(fig, update_plot, interval=100, blit=True)

    plt.show()

    # Giữ node chạy
    rospy.spin()
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

