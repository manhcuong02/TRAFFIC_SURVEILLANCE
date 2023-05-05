from audioop import reverse
import cv2 as cv
import numpy as np
from constant import *
from _utils import *

def handle_left_click(event, x, y,flags, points):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append([x, y])        
    # return points

def polygon_callback(filename):
    points = []
    video = cv.VideoCapture(filename)
    fill = False
    while True: 
        ret, frame = video.read()
        frame = cv.resize(frame, IMG_SIZE)
        frame = draw_polylines(frame, points)
         
        cv.imshow('video', frame)
        
        #  Nếu kích chuột trái thì lấy điểm theo tọa độ x,y vào thêm vào mảng point
        cv.setMouseCallback('video', handle_left_click, points) 
        
        key = cv.waitKey(10)
        if key == ord('q'):
            break
        
        if key == ord('d'):
            points.append(points[0])
        
    video.release()
    cv.destroyAllWindows()
    
    return points

def line_callback(filename):
    points = []
    video = cv.VideoCapture(filename)
    fill = False
    while True: 
        ret, frame = video.read()
        frame = cv.resize(frame, (IMG_SIZE))
        frame = draw_polylines(frame, points)

        cv.imshow('video', frame)
        if len(points) < 3:
            #  Nếu kích chuột trái thì lấy điểm theo tọa độ x,y vào thêm vào mảng point
            cv.setMouseCallback('video', handle_left_click, points) 
        
        key = cv.waitKey(10)
        if key == ord('q'):
            break
        
    video.release()
    cv.destroyAllWindows()
    
    return points

if __name__ =="__main__":
    points = polygon_callback(VIDEO_PATH)            
    np.savetxt(POLYGON_PATH, points, delimiter = ' ', fmt = '%s')
    # points = line_callback(VIDEO_PATH)            
    # np.savetxt(LINE_PATH, points, delimiter = ' ', fmt = '%s')
        