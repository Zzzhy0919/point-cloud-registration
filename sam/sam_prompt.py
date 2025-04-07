from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from PIL import Image
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import cv2
import open3d as o3d

def show_mask(image, mask):
    color_ = [ 0, 255, 0] 
    color = np.array(color_) 
    h, w = mask.shape[-2:] 
    imageResult = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) 
    for i in range(0, h ):
        for j in range(0, w ):
            if color_[0] == imageResult[i][j][0] and color_[1] == imageResult[i][j][1] and color_[2] == imageResult[i][j][2]:
                image[i][j] = color_
    return image

sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam = sam.to('cuda')
predictor = SamPredictor(sam)
point_coords = np.array([[306, 308]])
# point_coords = np.array([[270, 347]])
point_labels = np.array([1])
print("Loaded SAM model")



filename = 'realsense_rgb.png'
# filename = 'headcamera_rgb.png'
start_time = time.time()
image = cv2.imread(filename)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 


predictor.set_image(image)
masks, scores, _ = predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    multimask_output=False
)

result = show_mask(image, masks)
for point in point_coords:
    x, y = point
    result[y, x] = [255, 0, 0]
plt.imsave("realsense_bowl.png", result)
# plt.imsave("headcamera_bowl.png", result)
end_time = time.time()
execution_time = end_time - start_time
print("Socre:   ", scores) 
print(f"Processed {filename} in {end_time - start_time:.2f} seconds")


def depth_to_point_cloud(depth_map, mask, K):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    height, width = depth_map.shape

    points = []

    for v in range(height):
        for u in range(width):
            if mask[0, v, u] == 1: 
                d = depth_map[v, u]
                if d > 0: 
                    X = (u - cx) * d / fx
                    Y = (v - cy) * d / fy
                    Z = d
                    points.append([X, Y, Z])

    return np.array(points)


depth_map = np.load('realsense_depth.npy')
# depth_map = np.load('headcamera_depth.npy')
K = np.array([[554.3827128226441, 0, 320.5], [0, 554.3827128226441, 240.5], [0, 0, 1]])  # 相机内参矩阵

points = depth_to_point_cloud(depth_map, masks, K)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
o3d.io.write_point_cloud("realsense_bowl.pcd", pcd)
# o3d.io.write_point_cloud("headcamera_bowl.pcd", pcd)


