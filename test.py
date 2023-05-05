import json 
with open('data1.json') as f:
    data = json.load(f)

from main import remove_vehicle_info
from _utils import *
from constant import *
line_vertices = read_points(LINE_PATH)

import matplotlib.pyplot as plt 
import cv2 as cv
frame = cv.imread('data/images/road_test.jpg')
frame = cv.resize(frame, IMG_SIZE)

frame = draw_roi_and_lines(frame, None, line_vertices)  


for i in data:
    xywh = [int(t) for t in i['xywh']]
    cv.circle(frame, xywh[:2], 5, (0,255,0), -1)

cv.imshow('',frame)
cv.waitKey()