# TRAFFIC SURVEILANCE

# Download data
Vui lòng xem hướng dẫn tại [đây](https://github.com/manhcuong02/traffic-status-evaluation/blob/main/data/videos/Readme.md)

Please refer to the instructions [here](https://github.com/manhcuong02/traffic-status-evaluation/blob/main/data/videos/Readme.md)

# Download weights for yolov5
Vui lòng xem hướng dẫn tại [đây](https://github.com/manhcuong02/traffic-status-evaluation/blob/main/weights/Readme.md)

Please refer to the instructions [here](https://github.com/manhcuong02/traffic-status-evaluation/blob/main/weights/Readme.md)

# Download All
Tất cả source code có sẵn tại [đây](https://drive.google.com/drive/folders/1YvIHFhBLs677atnxiLsfoq5cT2z3unM2?usp=sharing](https://doc-9o-9c-drive-data-export.googleusercontent.com/download/659k930vll651jhiq80aktit4v0fmlc8/g4eee6der6de8ver3kle8ojvcljr8uaq/1684804500000/c071794c-0c49-43bd-92bb-5b86d61a4d6f/114029249490242181682/ADt3v-NaeSE_d8iqqsJd4BFnydDk_4NrHimREj30Z_zrBkh6fhigHURA2byXqUa78NJJi05ADwWzz33Emy5kPBnIalYOsntkveP5tbWaLTtaNQ9jMyMetXCS2L6r5Ip22QhAtbS0bhHj3uNsMu9zRlWHKmyEtf94kycZKzu8Eh0dt4Dsm92VsQgtmDOkeghdsHw0EMBFy8XX-R2cVAMgzNgq5DC5eAfZzcLJBJEuhFqKaK1TlMG9CQP3dZ2FdN14WWXzrFRFhoZvs4eIQCPKqR-e9fiQmmijQ1DlNCUoDBHUhJzcvyFZqcqJ1nNnNB3UTtyDIwQnv_rQUjnlv8GCfmPJ_ko6UFd4eQ==?authuser=1&nonce=4goqgo0lnu4bu&user=114029249490242181682&hash=l8e61d9eljjepc804pq2u1fo05vm87in)

All source code is available [here](https://drive.google.com/drive/folders/1YvIHFhBLs677atnxiLsfoq5cT2z3unM2?usp=sharing](https://doc-9o-9c-drive-data-export.googleusercontent.com/download/659k930vll651jhiq80aktit4v0fmlc8/g4eee6der6de8ver3kle8ojvcljr8uaq/1684804500000/c071794c-0c49-43bd-92bb-5b86d61a4d6f/114029249490242181682/ADt3v-NaeSE_d8iqqsJd4BFnydDk_4NrHimREj30Z_zrBkh6fhigHURA2byXqUa78NJJi05ADwWzz33Emy5kPBnIalYOsntkveP5tbWaLTtaNQ9jMyMetXCS2L6r5Ip22QhAtbS0bhHj3uNsMu9zRlWHKmyEtf94kycZKzu8Eh0dt4Dsm92VsQgtmDOkeghdsHw0EMBFy8XX-R2cVAMgzNgq5DC5eAfZzcLJBJEuhFqKaK1TlMG9CQP3dZ2FdN14WWXzrFRFhoZvs4eIQCPKqR-e9fiQmmijQ1DlNCUoDBHUhJzcvyFZqcqJ1nNnNB3UTtyDIwQnv_rQUjnlv8GCfmPJ_ko6UFd4eQ==?authuser=1&nonce=4goqgo0lnu4bu&user=114029249490242181682&hash=l8e61d9eljjepc804pq2u1fo05vm87in)


# Tutorials
- Nếu chạy video mẫu có sẵn ở link trên:
  - Muốn thay đổi đường dẫn video mẫu: thay đổi tên video ở trong file  -> constant.py
  - Chạy file main.py để xem kết quả thu được
- Nếu muốn chạy video khác (custom data) với những video mẫu:
  - Thay đổi các tham số (parameters) trong file constant.py phù hợp với bối cảnh của con đường trong video
  - Chạy file mouse_callback để lấy các điểm của vùng ROI(region of interest - vùng chứa đoạn đường) theo giá trị từ [left_top, right_top, right_bottom, left_bottom] và 2 điểm của đường thẳng để tham chiếu vật thể đi qua (để ở cuối trên hướng di chuyển của vật thể).
  - Thay đổi đường dẫn của video trong file constant.py
  - Chạy file main.py để xem kết quả
     
