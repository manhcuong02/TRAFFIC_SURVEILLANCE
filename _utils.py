import cv2 as cv
import numpy as np
from constant import *
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from typing import *
import torch 
from numpy._typing import *
from utils.plots import Colors

color = Colors()

def check_vehicle_crossed_line(point: ArrayLike, line_points: ArrayLike) -> bool:
    if torch.is_tensor(point):
        if point.device.type == 'cuda':
            point = point.to('cpu')
        point = point.numpy()
    x, y = point
    point1, point2 = line_points
    x1,y1 = point1
    x2,y2 = point2
    y_lines = (x - x1)*(y2-y1)/(x2-x1) + y1
    
    return y > y_lines

def is_inside_polygon(points: ArrayLike, polygon: ArrayLike) -> ArrayLike:
    """Kiểm tra các điểm có nằm trong đa giác hay không"""
    '''points: (center_x ,center_y)'''
    if torch.is_tensor(points):
        if points.device.type == 'cuda':
            points = points.to('cpu')
        points = points.numpy()
    else:
        points = np.array(points)
    
    inside = []
    polygon = Polygon(polygon)
    if len(np.array(points).shape) == 1: 
        point = Point(points)
        inside.append(polygon.contains(point))
    else:    
        for point in points:
            point = Point(point)
            inside.append(polygon.contains(point))
    return inside

def draw_polylines(frame: np.ndarray, polygon_points: ArrayLike) -> np.ndarray:
    '''Vẽ các đường của đa giác'''
    for point in polygon_points: 
        x, y = point
        cv.circle(frame, (x,y), 3, (0,0, 255), -1) # màu đỏ và fill
    cv.polylines(frame, [np.int32(polygon_points)], False, (0,0, 255), thickness=1)
    return frame

def draw_fill_polygon(frame, points):
    '''Tô màu vào đa giác và phủ cho nó độ trong suốt là alpha = 0.2'''
    overlay = frame.copy()
    output = frame.copy()
    
    cv.fillPoly(overlay, [np.array(points)], color = ROI_COLOR)
    output = cv.addWeighted(overlay, ALPHA, output, BETA, GAMMA)
    return output

def read_points(filename):
    points = np.loadtxt(filename, dtype = np.uint32, delimiter = ' ')
    return points.tolist()

def draw_roi_and_lines(frame, polygon_points = None, line_points = None):
    if polygon_points is not None:
        frame = draw_fill_polygon(frame, polygon_points)
        frame = draw_polylines(frame, polygon_points)
    
    if line_points is not None:
        frame = cv.line(frame, line_points[0], line_points[1], (0,255,0), thickness = 2)
    return frame

def show_vehicle_count(frame: np, num_objects = None, vehicle_flow = None, vehicle_density = None, traffic_status = None):
    '''density_count: Dict[str, int], vehicle_flow: int'''
    line = 1
    if num_objects is not None:
        for key, value in num_objects.items():
            text = key + ': ' + str(value)
            frame = cv.putText(frame, text, (10, LINE_SPACING*line), FONT_TEXT, FONT_SCALE_TEXT, TEXT_COLOR, THICKNESS_TEXT, LINESTYPE_TEXT)
            
            line += 1

    if traffic_status is not None:
        frame = cv.putText(frame, 'status: {}'.format(traffic_status), (IMG_SIZE[0] - 220, LINE_SPACING*(line-2)), FONT_TEXT, FONT_SCALE_TEXT, TEXT_COLOR, THICKNESS_TEXT, LINESTYPE_TEXT)

    if vehicle_density is not None:
        frame = cv.putText(frame, '-------------', (10, LINE_SPACING*line), FONT_TEXT, FONT_SCALE_TEXT, TEXT_COLOR, THICKNESS_TEXT, LINESTYPE_TEXT)
        line += 1
        frame = cv.putText(frame, f'vehicle_density: {vehicle_density}', (10, LINE_SPACING*line), FONT_TEXT, FONT_SCALE_TEXT, TEXT_COLOR, THICKNESS_TEXT, LINESTYPE_TEXT)
        line += 1

    if vehicle_flow is not None:
        frame = cv.putText(frame, '-------------', (10, LINE_SPACING*line), FONT_TEXT, FONT_SCALE_TEXT, TEXT_COLOR, THICKNESS_TEXT, LINESTYPE_TEXT)
        line += 1
        frame = cv.putText(frame, f'vehicle_flow_per_minute: {vehicle_flow}', (10, LINE_SPACING*line), FONT_TEXT, FONT_SCALE_TEXT, TEXT_COLOR, THICKNESS_TEXT, LINESTYPE_TEXT)
        line += 1

    
    return frame
        
def xyxy_to_tlwh(boxes: torch.Tensor) -> torch.Tensor:    
    '''cx_cy_w_h to top_left_w_h'''
    return torch.concat(
        [
            boxes[...,:2], 
            boxes[..., 2:] - boxes[...,:2]
        ], dim = -1
    )
    
def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:    
    '''xyxy to cx_cy_w_h'''
    return torch.concat(
        [
            torch.div(boxes[...,:2] + boxes[...,2:],2, rounding_mode = 'trunc'), 
            boxes[..., 2:] - boxes[...,:2]
        ], dim = -1
    )
    
def draw_bboxes(frame, det,vehicle_info = None,  tracker = None):
    for *xyxy, trackid, cls, age in reversed(det):
        class_num = int(cls)  # integer class
        if tracker is not None or tracker is False:
            if vehicle_info is not None or vehicle_info is False:
                idx = next((index for (index, d) in enumerate(vehicle_info) if d["id"] == int(trackid)), None)
                try:
                    vel = vehicle_info[idx]['vel']
                    if vel != -1:
                        label = f'{trackid} {CLASSES[class_num]} {vel}'
                    else: 
                        label = f'{trackid} {CLASSES[class_num]}'
                except:
                    label = f'{trackid} {CLASSES[class_num]}'
            else:
                label = f'{trackid} {CLASSES[class_num]}'
        else: 
            label = f'{CLASSES[class_num]}'
        # label = f'{self.class_names[class_num]} {conf:.2f}'
        plot_box_and_label(frame, 1, xyxy, label, color = color(class_num, True))
    
    return frame
        
def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    # Add one xyxy box to image with label
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv.rectangle(image, p1, p2, color, thickness=lw, lineType=cv.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv.rectangle(image, p1, p2, color, -1, cv.LINE_AA)  # filled
        cv.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
                    thickness=tf, lineType=cv.LINE_AA)