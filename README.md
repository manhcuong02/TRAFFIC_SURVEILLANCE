# TRAFFIC SURVEILLANCE

![](https://s1-www.theimagingsource.com/eb006f9a/dist/news/2017/01/16/assets/fig_01.en_US.webp)

Transforming a nation and advancing its development necessitates investment in fundamental elements, with an intelligent traffic monitoring system becoming increasingly indispensable. In a world evolving at a rapid pace, the utilization of advanced technology and the integration of cameras into the management and surveillance of traffic conditions not only contribute to enhancing the quality of urban life but also play a pivotal role in reducing accidents and improving overall road safety. Supported by the continuous development of technology, our project is driven by this critical objective and harnesses the power of Artificial Intelligence (AI) alongside advanced computer vision technology to address these paramount challenges.

## Project Components

In this project, we comprehensively evaluate traffic conditions based on three key criteria: traffic density, traffic flow, and vehicle speed. Here is a detailed breakdown of our approach:

**1. Defining the Monitored Road Section:**
   - Firstly, we identify the specific road segment to be monitored. This step is essential to ensure that unnecessary or off-road vehicle tracking is avoided.

**2. Object Detection and Tracking:**
   - Subsequently, we employ YOLOv5 for vehicle detection, allowing us to identify and track vehicles within the designated area.
   - To facilitate precise tracking, we implement DeepSORT, a sophisticated object tracking algorithm, ensuring accurate vehicle monitoring even when they are at varying distances from the camera.

**3. Vehicle Speed Estimation:**
   - To determine the speed of each vehicle, we perform a transformation from 3D space to 2D space, allowing us to evaluate vehicle speeds accurately. This method works effectively regardless of whether the vehicles are near or far from the camera.

**4. Traffic Density Calculation:**
   - Calculating traffic density is a critical aspect of our project. We assign weighted values to different types of vehicles since each vehicle type occupies varying amounts of space on the road.
   - This approach ensures fair and precise traffic density calculations, accounting for the varying sizes of vehicles.

**5. Traffic Flow Measurement:**
   - To measure traffic flow, we continuously count the number of vehicles passing through the monitored road section over a specified time interval.

By employing these meticulous calculation and evaluation methods, we provide real-time assessments of traffic conditions on the monitored road segment. This comprehensive approach allows us to deliver accurate and valuable insights into the traffic status, enabling better decision-making for traffic management and road safety enhancement.

## Installation

1. Clone the repository
```bash
git clone https://github.com/manhcuong02/TRAFFIC_SURVEILLANCE
cd TRAFFIC_SURVEILLANCE
```

2. Install the required dependencies
```bash
pip install -r requirements.txt
```

## Download
1. Download video. Please refer to the instructions [here](https://github.com/manhcuong02/traffic-status-evaluation/blob/main/data/videos/Readme.md)

2. Download weights for yolov5. Please refer to the instructions [here](https://github.com/manhcuong02/traffic-status-evaluation/blob/main/weights/Readme.md)

3. All source code is available [here](https://doc-9o-9c-drive-data-export.googleusercontent.com/download/659k930vll651jhiq80aktit4v0fmlc8/g4eee6der6de8ver3kle8ojvcljr8uaq/1684804500000/c071794c-0c49-43bd-92bb-5b86d61a4d6f/114029249490242181682/ADt3v-NaeSE_d8iqqsJd4BFnydDk_4NrHimREj30Z_zrBkh6fhigHURA2byXqUa78NJJi05ADwWzz33Emy5kPBnIalYOsntkveP5tbWaLTtaNQ9jMyMetXCS2L6r5Ip22QhAtbS0bhHj3uNsMu9zRlWHKmyEtf94kycZKzu8Eh0dt4Dsm92VsQgtmDOkeghdsHw0EMBFy8XX-R2cVAMgzNgq5DC5eAfZzcLJBJEuhFqKaK1TlMG9CQP3dZ2FdN14WWXzrFRFhoZvs4eIQCPKqR-e9fiQmmijQ1DlNCUoDBHUhJzcvyFZqcqJ1nNnNB3UTtyDIwQnv_rQUjnlv8GCfmPJ_ko6UFd4eQ==?authuser=1&nonce=4goqgo0lnu4bu&user=114029249490242181682&hash=l8e61d9eljjepc804pq2u1fo05vm87in)

## Usage
1. Download the required dependencies as mentioned above
2. Open your terminal. Run the following command
```bash
python3 main.py
```
