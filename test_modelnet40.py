import numpy as np
from data import load_data, farthest_subsample_points, random_Rt
import open3d as o3d
import random
import h5py

def load_h5(path):
    f = h5py.File(path, 'r+')
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    seg = f['seg'][:].astype('int64')
    return data, label, seg

def plot_pc(data1, data2=None):
    if data2 is None:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(data1)
        pc.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 255], (data1.shape[0], 1)))
        pc = pc.voxel_down_sample(voxel_size=0.01)
        o3d.visualization.draw_geometries([pc], width=800, height=600)
    if data2 is not None:
        pc1 = o3d.geometry.PointCloud()
        pc2 = o3d.geometry.PointCloud()
        pc1.points = o3d.utility.Vector3dVector(data1)
        pc2.points = o3d.utility.Vector3dVector(data2)
        pc1.colors = o3d.utility.Vector3dVector(np.tile([255, 0, 0], (data1.shape[0], 1)))
        pc2.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 255], (data2.shape[0], 1)))
        o3d.visualization.draw_geometries([pc1, pc2], width=800, height=600)

def my_plot_pc(src, tgt, R, t):
    pc1 = o3d.geometry.PointCloud()
    pc2 = o3d.geometry.PointCloud()
    pc1.points = o3d.utility.Vector3dVector(src)
    pc2.points = o3d.utility.Vector3dVector(tgt)
    pc1.colors = o3d.utility.Vector3dVector(np.tile([255, 0, 0], (src.shape[0], 1)))
    pc2.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 255], (tgt.shape[0], 1)))

    tgt_pre = np.matmul(R, src.T) + t[:, np.newaxis]
    tgt_pre = tgt_pre.T
    pc3 = o3d.geometry.PointCloud()
    pc3.points = o3d.utility.Vector3dVector(tgt_pre)
    pc3.colors = o3d.utility.Vector3dVector(np.tile([255, 0, 0], (tgt_pre.shape[0], 1)))

    o3d.visualization.draw_geometries([pc1, pc2], width=800, height=600)
    o3d.visualization.draw_geometries([pc3, pc2], width=800, height=600)

if __name__ == '__main__':
    # pcd = o3d.io.read_point_cloud('D:\\Pycharm-Projects\\DGCNN_for_Satellite_seg\\Results_LFA\\2_2_pred.ply')
    # o3d.visualization.draw_geometries([pcd], width=800, height=600)

    path = 'D:\\Pycharm-Projects\\DGCNN_for_Satellite_seg\\dataset\\train_val.h5'
    partition = 'train'
    gaussian_noise = False
    alpha = 0.75
    data, label, seg = load_h5(path)
    # label = label.squeeze()
    idx = random.randint(0, len(data))
    # plot_pc(data[idx, :])
    data = data[:, :2048]
    pc1 = data[idx, :]
    plot_pc(pc1)
    R_ab, translation_ab, euler_ab = random_Rt(np.pi / 4)
    pc2 = np.matmul(R_ab, pc1.T) + translation_ab[:, np.newaxis]
    pc2 = pc2.T
    plot_pc(pc1, pc2)
    pc1, pc2 = farthest_subsample_points(pc1.T, pc2.T, num_subsampled_points=int(data.shape[1]*alpha))
    plot_pc(pc1.T, pc2.T)
