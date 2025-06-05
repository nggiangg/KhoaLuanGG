#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import LaserScan

def scan_callback(msg):
    print("Tan so scan (Hz):", 1.0 / msg.time_increment if msg.time_increment > 0 else "Không xác định")
    print("So luong tia:", len(msg.ranges))
    print("Goc bat dau (rad):", msg.angle_min)
    print("Goc ket thuc (rad):", msg.angle_max)
    print("Buoc goc (rad):", msg.angle_increment)
    print("Thoi gian giua 2 tia (s):", msg.time_increment)
    print("Thoi gian giua 2 ban scan (s):", msg.scan_time)
    print("")
if __name__ == '__main__':
    rospy.init_node('scan_listener')
    rospy.Subscriber('/scan', LaserScan, scan_callback)
    rospy.spin()
