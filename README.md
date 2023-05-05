# Traffic-status-evaluation

# Download data
Vui lòng xem hướng dẫn tại [đây](https://github.com/manhcuong02/traffic-status-evaluation/blob/main/data/videos/Readme.md)
Please refer to the instructions [here](https://github.com/manhcuong02/traffic-status-evaluation/blob/main/data/videos/Readme.md)

# Download weights for yolov5
Vui lòng xem hướng dẫn tại [đây](https://github.com/manhcuong02/traffic-status-evaluation/blob/main/weights/Readme.md)
Please refer to the instructions [here](https://github.com/manhcuong02/traffic-status-evaluation/blob/main/weights/Readme.md)


# Tutorials
- Nếu chạy video mẫu có sẵn ở link trên:
  - Muốn thay đổi đường dẫn video mẫu: thay đổi tên video ở trong file  -> constant.py
  - Chạy file main.py để xem kết quả thu được
- Nếu muốn chạy video khác (custom data) với những video mẫu:
  - Thay đổi các tham số (parameters) trong file constant.py phù hợp với bối cảnh của con đường trong video
  - Chạy file mouse_callback để lấy các điểm của vùng ROI(region of interest - vùng chứa đoạn đường) theo giá trị từ [left_top, right_top, right_bottom, left_bottom] và 2 điểm của đường thẳng để tham chiếu vật thể đi qua (để ở cuối trên hướng di chuyển của vật thể).
  - Thay đổi đường dẫn của video trong file constant.py
  - Chạy file main.py để xem kết quả
     
