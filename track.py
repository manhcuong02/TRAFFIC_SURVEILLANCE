import numpy as np
import torch

from deep_sort.deep.feature_extractor import Extractor
from deep_sort.sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.sort.detection import Detection
from deep_sort.sort.tracker import Tracker


__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_type, max_dist=0.2, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):

        self.extractor = Extractor(model_type, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, det , ori_img, use_yolo_preds=True):
        bbox_xywh, confidences, classes = self.detection_to_track_metrics(det)
        self.height, self.width = ori_img.shape[:2]

        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i], bbox_xywh[i]) for i, conf in enumerate(
            confidences)]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, classes)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            if use_yolo_preds:
                det = track.get_yolo_pred()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(det.tlwh)
            else:
                box = track.to_tlwh()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            class_id = track.class_id
            age = track.age
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, age], dtype=np.int32))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return torch.tensor(outputs)

    def detection_to_track_metrics(self, det):
        return self._xyxy_to_xywh(det[:, :4]), det[:, 4], det[:, 5]

    @staticmethod
    def _xyxy_to_xywh(boxes_xyxy):
        '''xyxy to cx_cy_w_h'''
        return torch.concat(
        [
            torch.div(boxes_xyxy[...,:2] + boxes_xyxy[...,2:],2, rounding_mode = 'trunc'), 
            boxes_xyxy[..., 2:] - boxes_xyxy[...,:2]
        ], dim = -1
    )

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        '''cx_cy_w_h to top_left_w_h'''
        return torch.cat(
            [
                bbox_xywh[..., :2] - torch.div(bbox_xywh[..., 2:], 2, rounding_mode = 'trunc'),
                bbox_xywh[..., 2:]
            ], dim = -1
        )

    def _xywh_to_xyxy(self, bbox_xywh):
        '''cx_cy_w_h to xyxy'''
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features



