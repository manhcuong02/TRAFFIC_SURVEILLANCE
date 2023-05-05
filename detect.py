import argparse
import os
import platform
import sys
import numpy as np
import cv2
import torch
from _utils import *

from models.common import DetectMultiBackend
from utils.general import (check_img_size,
                            non_max_suppression, scale_boxes, xyxy2xywh, yaml_load)
from utils.torch_utils import select_device
from utils.augmentations import letterbox

class Detection():
    def __init__(self, weights, device, yaml, img_size = 640, half = False, tracker = None) -> None:
        self.weights = weights 
        if device == 'cuda' or device == 'gpu':
            device = 'cuda:0'
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device = self.device, data=yaml, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.img_size = check_img_size(img_size, self.stride)
        self.tracker = tracker
        self.classes = yaml_load(yaml)
        
    @staticmethod
    def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)
        
    def preprocess_image(self,img_src, img_size, stride):
        '''Process image before image inference.'''
        image = letterbox(img_src, img_size, stride, auto=self.pt)[0]
        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = torch.from_numpy(np.ascontiguousarray(image)).to(self.model.device)
        image = image.half() if self.model.fp16 else image.float()   # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0

        return image, img_src
        
    def infer(self, source, conf_thresh=0.25, iou_thresh=0.45, classes=None, agnostic_nms=False, max_det=1000, polygon_vertices = None, augment = False):
        image, img_src = self.preprocess_image(source, img_size=self.img_size, stride=self.stride)
        
        if classes is not None:
            id_classes = list(classes.keys())
        else:
            id_classes = None
        
        if len(image.shape) == 3:
                image = image[None]    
        det = self.model(image, augment=augment)
        det = non_max_suppression(det, conf_thresh, iou_thresh, id_classes, agnostic_nms, max_det=max_det)
        if len(det):
            det = det[0].to('cpu')
            det[:, :4] = scale_boxes(image.shape[2:], det[:, :4], img_src.shape).round()
            if polygon_vertices is not None:
                center_points = torch.stack(
                    [   
                        #  rounding_mode = trunc: làm tròn tiến về 0 cho số âm.
                        torch.div(det[:, 0] + det[:,2], 2, rounding_mode = 'trunc'),
                        torch.div(det[:, 1] + det[:,3], 2, rounding_mode = 'trunc')
                    ], dim = -1
                )
                inside_polygon = is_inside_polygon(center_points, polygon_vertices)
                det = det[inside_polygon]
            
            else: 
                pass
                
            if self.tracker is not None:
                det = self.tracker.update(det, source)
            
            return det
            
        #     for *xyxy, trackid, cls in reversed(det):
        #         # if self.tracker == None -> trackid is conf of object 
                
        #         class_num = int(cls)  # integer class
        #         if self.tracker is not None:
        #             label = '{} {}'.format(trackid, self.classes['names'][class_num])
        #         else: 
        #             label = '{}'.format(self.classes['names'][class_num])
        #         # label = f'{self.class_names[class_num]} {conf:.2f}'
        #         self.plot_box_and_label(img_src, 1, xyxy, label, color = (255, 0, 0))

        #     img_src = np.asarray(img_src)

        # return img_src, det
            