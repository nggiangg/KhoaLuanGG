# KhoaLuanGG
Autonomous JetRacer system using Jetson Nano (Linux Embedded), integrates Python, ROS, jetson-inference for sensor data processing and vehicle control, applied in mobile robotics and AI.


###########################################################################
When run code: rosrun python_detect read_scan.py
('So luong tia:', 1147)
('Goc bat dau (rad):', -3.1415927410125732)
('Goc ket thuc (rad):', 3.1415927410125732)
('Buoc goc (rad):', 0.005482709966599941)
('Thoi gian giua 2 tia (s):', 0.00011767593241529539)
('Thoi gian giua 2 ban scan (s):', 0.13485661149024963)

----------------------------------------------------------------------------
Số lượng tia: 1147
Góc bắt đầu: -3.1415927 rad (~ -180°)
Góc kết thúc: 3.1415927 rad (~ 180°)
Bước góc: 0.00548271 rad (~0.314°/tia)
Thời gian giữa 2 tia: 0.00011768 s
Thời gian giữa 2 bản scan: 0.1348566 s

- Tần số scan (Hz) = 1/(Thời gian giữa 2 bản scan)
    1/(thời gian giữa 2 bản scan) = 1 / 0.1348566 ≈ 7.415 Hz
- Ý nghĩa các thông số:
    1147 tia**: Mỗi vòng quét trả về 1147 giá trị khoảng cách.
    Góc quét**: Quét toàn bộ 360° (từ -π đến +π).
    Bước góc**: Mỗi tia cách nhau ~0.314°.
    Tần số scan**: Khoảng 7.4 lần/giây.
