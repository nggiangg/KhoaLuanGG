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
    width = CAR_WIDTH / 100
    length = CAR_LENGTH / 100

    # Vẽ xe (hình chữ nhật) với tâm ở gốc tọa độ
    car_rect = Rectangle((-length/2, -width/2), length, width,
                        color='blue', alpha=0.5, label='JetRacer')
    ax_cart.add_patch(car_rect)
    ax_cart.plot(0, 0, 'go', markersize=8, label='LiDAR Center')

    arrow_len = 1.0  # 1 mét

    # 0° (phía trước, Ox dương)
    ax_cart.arrow(0, 0, arrow_len, 0, head_width=0.05, head_length=0.1, fc='red', ec='red')
    ax_cart.text(arrow_len + 0.2, 0, u'0°\nTrước', color='red', ha='left', va='center', fontsize=10, fontweight='bold')

    # 90° (bên trái, Oy dương)
    ax_cart.arrow(0, 0, 0, arrow_len, head_width=0.05, head_length=0.1, fc='blue', ec='blue')
    ax_cart.text(0, arrow_len + 0.2, u'90°\nTrái', color='blue', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 180° (phía sau, Ox âm)
    ax_cart.arrow(0, 0, -arrow_len, 0, head_width=0.05, head_length=0.1, fc='orange', ec='orange')
    ax_cart.text(-arrow_len - 0.2, 0, u'180°\nSau', color='orange', ha='right', va='center', fontsize=10, fontweight='bold')

    # 270° (bên phải, Oy âm)
    ax_cart.arrow(0, 0, 0, -arrow_len, head_width=0.05, head_length=0.1, fc='green', ec='green')
    ax_cart.text(0, -arrow_len - 0.2, u'270°\nPhải', color='green', ha='center', va='top', fontsize=10, fontweight='bold')

    ax_cart.legend(loc='upper right')

def scan_callback(msg):
    """Callback khi nhận dữ liệu scan mới"""
    global latest_scan
    latest_scan = msg

def update_plot(frame):
    global latest_scan

    if latest_scan is None:
        return scan_line, cart_line

    ranges = np.array(latest_scan.ranges)
    angles = np.linspace(latest_scan.angle_min, latest_scan.angle_max, len(ranges))

    # Xoay lại góc về chuẩn ROS: 0° là phía trước
    # Giả sử hiện tại 0° là phía sau, 180° là phía trước
    angles_ros = (angles + np.pi) % (2 * np.pi)  # Xoay 180 độ

    valid_indices = np.isfinite(ranges) & (ranges > 0) & (ranges < 10)
    valid_ranges = ranges[valid_indices]
    valid_angles = angles_ros[valid_indices]

    # Polar plot
    scan_line.set_data(valid_angles, valid_ranges)

    # Cartesian plot
    x = valid_ranges * np.cos(valid_angles)
    y = valid_ranges * np.sin(valid_angles)
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
    info_text = u"- Tâm LiDAR nằm ở giữa xe (điểm xanh)\n" + \
	    u"- Hướng 0° của LiDAR là hướng trước của xe\n" + \
	    u"- Kích thước xe: 20cm × 30cm\n" + \
	    u"- Vùng xanh biểu thị xe JetRacer"
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

