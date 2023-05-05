import cv2 as cv
import os

#path
name_video = 'Xuanthuy2.MOV'    
VIDEO_PATH = os.path.join('data/videos/', name_video)
POLYGON_PATH = 'data/polygon_points/polygon_points_{}.txt'.format(name_video.split('.')[0])
LINE_PATH = 'data/line_points/line_points_{}.txt'.format(name_video.split('.')[0])
RESULTS_DIR = 'results'



IMG_SIZE = (1280, 720) # width, height
FPS_CAMERA = 60
STREET_LENGTH = 50. # mét
ROAD_CALIB_SIZE = (300,IMG_SIZE[1]) # ảnh đoạn đường sau khi được hiệu chỉnh (width, height)
MIN_FRAMES_TO_CALCULATE_VELOCITY = 60

# evaluate 
VELOCITY_LOWER_BOUND = 15 # vận tốc nhỏ nhất km/h
VELOCITY_UPPER_BOUND = 40 # v

MAX_SLOW_VEHICLE = 20

VEHICLE_DENSITY_LOWER_BOUND = 10
VEHICLE_DENSITY_UPPER_BOUND = 50

VEHICLE_FLOW_UPPER_BOUND = 200
VEHICLE_FLOW_LOWER_BOUND = 50




#  YOLOv5
CONF_THRESH = 0.4
IOU_THRESH = 0.3
CLASSES = { # các class muốn lấy 
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
}

VEHICLE_WEIGHTS = {
    'bicycle': 1,
    'car': 2,
    'motorcycle': 1,
    'bus': 4,
    'truck': 4,
}

# Draw
ROI_COLOR = (0,255, 0) # BGR
ALPHA = 0.2 # độ trong suốt 
BETA = 1 - ALPHA
GAMMA = 0.0
LINE_SPACING = 20
FONT_TEXT = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE_TEXT = 0.6
LINESTYPE_TEXT = cv.LINE_AA
THICKNESS_TEXT = 1
TEXT_COLOR = (0,0,255)

# Track
MODEL_TYPE = "osnet_x1_0"
MAX_DIST =  0.1 # The matching threshold. Samples with larger distance are considered an invalid match
MAX_IOU_DISTANCE = 0.7 # Gating threshold. Associations with cost larger than this value are disregarded.
MAX_AGE =  30 # Maximum number of missed misses before a track is deleted
N_INIT =  1 # Number of frames that a track remains in initialization phase
NN_BUDGET =  100 # Maximum size of the appearance descriptors gallery
USE_CUDA = True
