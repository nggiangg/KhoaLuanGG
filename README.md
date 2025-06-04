# KhoaLuanGG

Autonomous JetRacer system using Jetson Nano (Linux Embedded), integrates Python, ROS, jetson-inference for sensor data processing and vehicle control, applied in mobile robotics and AI.

---

## ROS Node: read_scan.py

Khi chạy lệnh:
```sh
rosrun python_detect read_scan.py
```
Bạn sẽ nhận được các thông số từ cảm biến LiDAR như sau:

```
('So luong tia:', 1147)
('Goc bat dau (rad):', -3.1415927410125732)
('Goc ket thuc (rad):', 3.1415927410125732)
('Buoc goc (rad):', 0.005482709966599941)
('Thoi gian giua 2 tia (s):', 0.00011767593241529539)
('Thoi gian giua 2 ban scan (s):', 0.13485661149024963)
```

### Giải thích các thông số

- **Số lượng tia:** 1147  
  Mỗi vòng quét trả về 1147 giá trị khoảng cách.
- **Góc bắt đầu:** -3.1415927 rad (~ -180°)
- **Góc kết thúc:** 3.1415927 rad (~ 180°)  
  → Quét toàn bộ 360° (từ -π đến +π).
- **Bước góc:** 0.00548271 rad (~0.314°/tia)  
  Mỗi tia cách nhau khoảng 0.314°.
- **Thời gian giữa 2 tia:** 0.00011768 s
- **Thời gian giữa 2 bản scan:** 0.1348566 s

#### Tính toán tần số quét (scan frequency)
- **Tần số scan (Hz):**  
  ```
  1 / (Thời gian giữa 2 bản scan) = 1 / 0.1348566 ≈ 7.4 Hz
  ```
  → LiDAR quét khoảng 7.4 lần/giây.

---

> **Lưu ý:** Các thông số này giúp bạn hiểu rõ hơn về dữ liệu đầu ra của LiDAR và cấu hình hệ thống phù hợp cho các ứng dụng robot di động, AI, thị giác máy tính.
