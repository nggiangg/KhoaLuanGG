```sh
cd ~/catkin_ws/src

catkin_create_pkg python_detect_object rospy cv_bridge sensor_msgs std_msgs

#rospy: Thư viện Python để viết node ROS.
#cv_bridge: Dùng để chuyển đổi giữa OpenCV và ROS Image message.
#sensor_msgs: Chứa các định nghĩa message liên quan đến cảm biến (như ảnh, laser, v.v.).
#std_msgs: Chứa các định nghĩa message cơ bản (như String, Int32, v.v.).


```
