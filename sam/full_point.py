import open3d as o3d
import numpy as np
import copy

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])

# 加载点云文件
pcd1 = o3d.io.read_point_cloud("headcamera_bowl.pcd")  # 第一个视角的点云
pcd2 = o3d.io.read_point_cloud("realsense_bowl.pcd")  # 第二个视角的点云
trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
pcd1.transform(trans_init)

# 假设 transformer 是从 RANSAC + ICP 计算得到的 4x4 变换矩阵
transformer = np.array([
    [-0.57753469,  0.81613089,  0.01959718,  0.99207636],
    [ 0.03034627, -0.00252664,  0.99953625, -0.15966223],
    [ 0.81580193,  0.57786156, -0.0233073,   0.36528862],
    [ 0.  ,        0.     ,    0.      ,    1.        ]
])

draw_registration_result(pcd1, pcd2, transformer)
# # 将第二个点云应用变换矩阵
# pcd1.transform(transformer)

# # 为了区分两个点云，设置不同的颜色
# pcd1.paint_uniform_color([1, 0, 0])  # 红色
# pcd2.paint_uniform_color([0, 1, 0])  # 绿色

# # 合并两个点云
# combined_pcd = pcd1 + pcd2

# # 可视化合并后的点云
# o3d.visualization.draw_geometries([combined_pcd])
