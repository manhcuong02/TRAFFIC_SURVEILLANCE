import cv2 as cv
import numpy as np
from constant import *
from _utils import *
from detect import Detection 
from typing import *
import torch
from track import *
from road_calib import *
from numpy.typing import *
import math
import time
import json 

def init_tracker(model_type) -> DeepSort:
    deepsort = DeepSort(
        model_type = model_type, 
        max_dist = MAX_DIST,
        max_iou_distance = MAX_IOU_DISTANCE,
        max_age = MAX_AGE, 
        n_init = N_INIT,
        nn_budget = NN_BUDGET,
        use_cuda = USE_CUDA 
    )
    return deepsort

def count_vehicle_objects(classes: Dict[int,str], det: torch.Tensor) -> Dict[str,int]:
    """
    Đếm số lượng phương tiện giao thông trong ảnh.
    Args:
    - classes: dictionary, bao gồm các class và số thứ tự tương ứng, vd: {0: "car", 1: "truck", 2: "motorbike", 3: "bus"}
    - det: tensor, tensor chứa thông tin về các object được phát hiện trong ảnh, gồm các thông tin: x_left_top, y_left_top, x_right_bottom, y_right_bottom, trackid, class.
    Return:
    - vehicle_count: dictionary, bao gồm số lượng phương tiện giao thông theo từng class.
    """
    vehicle_count  = {
        v :0 for v in classes.values()
    }
    for *xyxy, trackid, cls, age in det:
        class_num = int(cls)
        vehicle_count [classes[class_num]] += 1

    return vehicle_count 

def calculate_vehicle_flow(prev_count: int, prev_data: ArrayLike ,line_vertices: ArrayLike, det: torch.Tensor) -> ArrayLike:
    '''Tính lưu lượng giao thông'''
    for *xyxy, trackid, cls, age in det:
        xywh = xyxy_to_xywh(torch.tensor(xyxy))
        center_point = xywh[:2].numpy()
        if check_vehicle_crossed_line(center_point, line_vertices) and trackid not in prev_data:
            prev_count += 1
            prev_data.append(trackid)            
    
    return prev_count, prev_data            
    
def calculate_vehicle_density(num_objects: Dict[str, int], weights_object: Dict[str, int]) -> int:
    '''Tính mật độ giao thông trên đường dựa vào loại xe và hệ số của từng loại xe'''
    vehicle_density = 0
    for key, value in num_objects.items():
        vehicle_density += weights_object[key] * value
        
    return vehicle_density
    
def frames_per_minute() -> int:
    '''số frame camera ghi được trong 1 phút'''
    return FPS_CAMERA * 60

def convert_pixel_to_meters(image_space_distance: float, height_img: int) -> float:
    '''Chuyển từ hệ pixel sang hệ mét'''
    return image_space_distance*1.0/height_img*STREET_LENGTH
    
def convert_to_kmh(image_space_distance: float, height_img: int, n_frame:int) -> float:
    '''convert từ pixel/frame to km/h 
        height_img: độ dài đoạn đường trên ảnh sau khi được hiệu chỉnh về đường thẳng,
        n_frame: số frame mà xe đã di chuyển '''
    meters = convert_pixel_to_meters(image_space_distance, height_img)
    kmh = meters*FPS_CAMERA/n_frame*3.6
    return round(float(kmh),2)

def convert_detection_xyxy2xywh(det: ArrayLike) -> torch.Tensor:
    '''Chuyển tọa độ bboxes của detections từ xyxy sang cxcywh'''
    if torch.is_tensor(det):
        if det.device.type == 'cuda':
            det = det.to('cpu')
    else:
        det = torch.tensor(det)
    new_det = det.clone()
        
    new_det[...,:4] = xyxy_to_xywh(new_det[...,:4])
    
    return new_det

def calculate_velocity(det: ArrayLike, vehicle_id: List, vehicle_info: List[Dict[str, Any]]) -> Tuple[List, Dict[str, Any]]:
    '''
    Args:
        det: *xywh in other coordinate, trackid, cls, age
        vehicle_id: List[trackid]
    Output:
        vehicle_id
        vehicle_info: List[Dict[str, int]]
    '''
    for *xywh, trackid, cls, age in det:
        if trackid not in vehicle_id:
            vehicle_id.append(int(trackid))
            vehicle = {
                'id': int(trackid),
                'xywh': [int(x) for x in xywh],
                # 'age': age,
                'distance_moved': 0,
                'vel': -1
            }
            vehicle_info.append(vehicle)
        else:
            index = vehicle_id.index(trackid)
            vehicle_info[index]['distance_moved'] += round(
                math.dist(xywh[:2], vehicle_info[index]['xywh'][:2]), 6
            )
            vehicle_info[index]['age'] = age
            vehicle_info[index]['xywh'] = xywh
            if age >= MIN_FRAMES_TO_CALCULATE_VELOCITY:
                vehicle_info[index]['vel'] = convert_to_kmh(vehicle_info[index]['distance_moved'], ROAD_CALIB_SIZE[1], age)

    return vehicle_id, vehicle_info

def count_velocity_ranges(vehicle_info: List[Dict[str, int]]) -> Dict[str, int]:
    '''Đếm số xe ở mỗi khoảng vận tốc khác nhau'''
    count = {
        'low_vel': 0,
        'medium_vel': 0,
        'high_vel': 0
    }
    for vehicle in vehicle_info:
        vel = vehicle['vel']
        if 0 < vel <= VELOCITY_LOWER_BOUND:
            count['low_vel'] += 1
        elif VELOCITY_LOWER_BOUND < vel < VEHICLE_FLOW_UPPER_BOUND:
            count['medium_vel'] += 1
        else: 
            count['high_vel'] += 1
    return count

def evaluate_traffic_condition(vehicle_density, vehicle_info, vehicle_flow):
    count_vehicle_velocity = count_velocity_ranges(vehicle_info)
    if vehicle_density > VEHICLE_DENSITY_UPPER_BOUND \
        or vehicle_flow > VEHICLE_FLOW_UPPER_BOUND \
            or count_vehicle_velocity['low_vel'] > MAX_SLOW_VEHICLE:
                return 'traffic jam'
    
    elif VEHICLE_DENSITY_LOWER_BOUND < vehicle_density <  VEHICLE_DENSITY_UPPER_BOUND \
        or VEHICLE_FLOW_LOWER_BOUND < vehicle_flow < VEHICLE_FLOW_UPPER_BOUND: 
            return 'heavy traffic'
    else:
        return 'normal traffic'
    
def remove_vehicle_info(vehicle_info, vehicle_id, trackid_out):
    'xóa thông tin xe khi đi biến mất trong ảnh'
    new_vehicle_info = []
    new_vehicle_id = []
    for vehicle in vehicle_info:
        if vehicle['id'] not in trackid_out:
            new_vehicle_info.append(vehicle)

    for i in vehicle_id:
        if i not in trackid_out:
            new_vehicle_id.append(i)
    return new_vehicle_id, new_vehicle_info
    

def main(weights, yaml_filename, device, img_show = True,save_result = False, save_result_path = None, save_fps  = FPS_CAMERA):
    
    deepsort = init_tracker(MODEL_TYPE)
    model = Detection(weights = weights, device = device, yaml = yaml_filename, img_size = 640,half = False, tracker = deepsort)
    vid = cv.VideoCapture(VIDEO_PATH)

    if save_result:
        if save_result_path is None:
            if RESULTS_DIR not in os.listdir():
                os.mkdir(RESULTS_DIR)
            exp_list = [int(i[3:]) for i in os.listdir(RESULTS_DIR)]
            exp_list.sort()

            if len(exp_list) == 0:
                save_result_dir = os.path.join(RESULTS_DIR, 'exp1') 
                
            else:
                save_result_dir = os.path.join(RESULTS_DIR, 'exp' + str(int(exp_list[-1]) + 1))

            os.mkdir(save_result_dir)
            save_result_path = os.path.join(save_result_dir, 'result.mp4')

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out_vid =  cv.VideoWriter(save_result_path, fourcc, save_fps, IMG_SIZE, isColor = True)


    # đọc tọa độ các điểm của 
    polygon_vertices = read_points(POLYGON_PATH)
    line_vertices = read_points(LINE_PATH)
    
    # ma trận chuyển đổi ảnh từ một góc chéo sang góc thẳng đứng
    # chuyển đổi một hình ảnh từ một hệ tọa độ 3D sang hệ tọa độ 2D
    transfromation_matrix = perspectiveTransform(polygon_vertices[:4], ROAD_CALIB_SIZE)
    
    vehicle_flow = 0
    trackid_out_data = []
    
    frame_count = 1

    vehicle_id, vehicle_info = [], []
    while (True):
        start_time = time.time()
        ret, frame = vid.read()
        if ret is False:
            break

        frame = cv.resize(frame, IMG_SIZE)
        
        ### detect
        det = model.infer(frame, conf_thresh =  CONF_THRESH, iou_thresh = IOU_THRESH, classes = CLASSES, polygon_vertices = polygon_vertices, augment = True)

        ### tính lưu lượng giao thông và mật độ giao thông
        num_objects = count_vehicle_objects(CLASSES, det)
        vehicle_density = calculate_vehicle_density(num_objects, VEHICLE_WEIGHTS)

        vehicle_flow, trackid_out_data = calculate_vehicle_flow(vehicle_flow, trackid_out_data, line_vertices, det)
        
        ###  draw roi
        frame = draw_roi_and_lines(frame, polygon_vertices, line_vertices)  

        if len(det) != 0:   
            transformed_det = convert_detection_xyxy2xywh(det)
            transformed_det[..., :2] = convertPoints2OtherCoordinates(transformed_det[...,:2], transfromation_matrix)
            
            vehicle_id, vehicle_info = calculate_velocity(transformed_det, vehicle_id, vehicle_info)
            vehicle_id, vehicle_info = remove_vehicle_info(vehicle_info, vehicle_id, trackid_out_data)
            
            # with open('vehicle_info.json', 'w') as outfile:
            #     json_data = json.dumps(vehicle_info, indent=4)
            #     outfile.write(json_data)
            # break
        
        traffic_status = evaluate_traffic_condition(vehicle_density, vehicle_info, vehicle_flow)
        
        ### show vehicle count
        frame = show_vehicle_count(frame, num_objects, vehicle_flow, vehicle_density = vehicle_density, traffic_status = traffic_status)
        
        ### Vẽ bounding boxes
        frame = draw_bboxes(frame, det, vehicle_info = vehicle_info, tracker = True)
        
        ### Hiển thị FPS
        end_time = time.time()
        fps = int(1.0 / (end_time - start_time))
        cv2.putText(frame, f"FPS: {fps}", (IMG_SIZE[0]-100, 30), FONT_TEXT, FONT_SCALE_TEXT, TEXT_COLOR, THICKNESS_TEXT)
        
        ### Hiển thị kết quả
        if img_show:
            cv.imshow('video', frame)
        
        ## in kết quả
        print('Frame {}: FPS = {}, speed = {} ms, num_objects: {}'.format(frame_count, fps, round((end_time - start_time)*1000, 2), len(det)))
        
        ### lưu kết quả
        if save_result:
            out_vid.write(frame)

        ### Tính lưu lượng giao thông theo từng phút, nên cứ qua mỗi 1 phút ta sẽ reset lại lưu lượng giao thông trước đó
        if frame_count % frames_per_minute() == 0: 
            vehicle_flow = 0
            trackid_out_data = []
        frame_count += 1


        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    if save_result:
        out_vid.release()
    vid.release()
    cv.destroyAllWindows()

if __name__ =='__main__':
    weights = 'weights/yolov5x.pt'
    yaml_filename = 'data/coco128.yaml'
    device = 'cuda'
    
    main(weights, yaml_filename, device, save_result = True, save_fps = 30, img_show= True)