# import necessary libraries
 
import cv2 
import numpy as np
from _utils import *
import torch
 
def convertPoints2OtherCoordinates(points: ArrayLike, matrix) -> torch.Tensor :
    if torch.is_tensor(points):
        if points.device.type == 'cuda':
            points = points.to('cpu')
        points = points.numpy()
    
    new_point = cv2.perspectiveTransform(np.float32(points).reshape(-1, 1, 2), matrix)
    return torch.tensor(new_point.reshape(-1,2))

def perspectiveTransform(polygon_points: ArrayLike, out_size: Tuple[int, int], src = None):
    '''
    polygon_points: tọa độ các đỉnh của đa giác theo thứ tự lần lượt là: trái trên, phải trên, phải dưới, trái dưới
    out_size: Kích thước ảnh đầu ra của đa giác -> List[width, height].
    src: Nếu có ảnh đầu vào thì sẽ trả về ảnh đầu ra
    '''
    transformation_matrix = cv2.getPerspectiveTransform(
        np.float32(polygon_points),
        np.float32(
            [
                [0,0], [out_size[0], 0], [out_size[0], out_size[1]], [0,out_size[1]]
            ]
        )
    )
    if src is not None:
        transformed_region = cv2.warpPerspective(src, transformation_matrix, out_size)
        return transformation_matrix, transformed_region
    
    return transformation_matrix

# cap = cv2.VideoCapture(VIDEO_PATH)

# while True:
     
#     ret, frame = cap.read()
#     if ret == False: 
#         break
#     frame = cv2.resize(frame, IMG_SIZE)
 
#     # Locate points of the documents
#     # or object which you want to transform
#     pts1 = np.float32(read_points(POLYGON_PATH))[:4]
#     pts2 = np.float32([[0, 0], [IMG_SIZE[0], 0],
#                        [IMG_SIZE[0], IMG_SIZE[1]], [0, IMG_SIZE[1]]])   
#     point = np.uint32([[550,100]])
     
#     # Apply Perspective Transform Algorithm
#     matrix, transformed_region = perspectiveTransform(pts1, (300,700), frame)
#     new_point = convertPoints2OtherCoordinates(point, matrix)

#     cv.circle(frame, point[0], 5, (0,255,0), -1)
#     cv.circle(transformed_region, new_point[0].numpy().astype(np.uint32), 5, (0,255,0), -1)
    
    
#     # Wrap the transformed image
#     cv2.imshow('frame', frame) # Initial Capture
#     cv2.imshow('frame1', transformed_region) # Transformed Capture
 
#     if cv2.waitKey(10) == ord('q'):
#         break
 
# cap.release()
# cv2.destroyAllWindows()