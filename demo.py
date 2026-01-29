# Adapted from FoundationStereo (https://github.com/NVlabs/FoundationStereo/blob/master/scripts/run_demo.py)
import os
import argparse
import imageio.v2 as imageio
import cv2
import numpy as np
import open3d as o3d
import torch

from bridgedepth.bridgedepth import BridgeDepth
from bridgedepth.utils.logger import setup_logger
from bridgedepth.utils import visualization


def depth2xyzmap(depth: np.ndarray, K, uvs: np.array=None, zmin=0.1):
    invalid_mask = (depth < zmin)
    H, W = depth.shape[:2]
    if uvs is None:
        vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W), sparse=False, indexing='ij')
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:, 0]
        vs = uvs[:, 1]
    zs = depth[vs, us] 
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N, 3)
    xyz_map = np.zeros((H, W, 3), dtype=np.float32)
    xyz_map[vs, us] = pts
    if invalid_mask.any():
        xyz_map[invalid_mask] = 0
    return xyz_map


def toOpen3dCloud(points, colors=None, normals=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_file', default='assets/left.png', type=str)
    parser.add_argument('--right_file', default='assets/right.png', type=str)
    parser.add_argument('--intrinsic_file', default='assets/K.txt', type=str, help='camera intrinsic matrix and baseline file')
    parser.add_argument('--model_name', choices=['rvc', 'rvc_pretrain', 'eth3d_pretrain', 'middlebury_pretrain'], default='rvc_pretrain')
    parser.add_argument('--checkpoint_path', default=None, type=str)
    parser.add_argument('--out_dir', default='demo_output', type=str, help='the directory to save results')
    parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
    parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
    args = parser.parse_args()

    logger = setup_logger(name="bridgedepth")
    logger.info("Arguments: " + str(args))
    os.makedirs(args.out_dir, exist_ok=True)

    pretrained_model_name_or_path = args.model_name
    if args.checkpoint_path is not None:
        assert os.path.exists(args.checkpoint_path)
        pretrained_model_name_or_path = args.checkpoint_path

    model = BridgeDepth.from_pretrained(pretrained_model_name_or_path)
    model = model.to(torch.device("cuda"))
    model.eval()

    img1 = imageio.imread(args.left_file)
    img2 = imageio.imread(args.right_file)
    H, W = img1.shape[:2]
    
    W_MAX = 1024
    if W > W_MAX:
        scale = W_MAX / W
        img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_LINEAR)
    else:
        scale = 1.0
    H, W = img1.shape[:2]
    logger.info(f"img1: {img1.shape}")

    viz = visualization.Visualizer(img1)

    sample = {
        'img1': torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2),
        'img2': torch.as_tensor(img2).cuda().float()[None].permute(0, 3, 1, 2),
    }
    with torch.no_grad():
        results_dict = model(sample)
    disp = results_dict['disp_pred'].clamp_min(1e-3).cpu().numpy().reshape(H, W)

    vis = viz.draw_disparity(disp, colormap=cv2.COLORMAP_TURBO, enhance=False).get_image()
    # vis = viz.draw_disparity(disp, colormap='kitti', enhance=False).get_image()
    vis = np.concatenate([img1, vis], axis=1)
    imageio.imwrite(f'{args.out_dir}/vis.png', vis)
    logger.info(f"Outputs saved to {args.out_dir}")

    if args.get_pc:
        with open(args.intrinsic_file, 'r') as f:
            lines = f.readlines()
            K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3, 3)
            baseline = float(lines[1])
        K[:2] *= scale
        depth = K[0, 0] * baseline / disp
        np.save(f'{args.out_dir}/depth_meter.npy', depth)
        xyz_map = depth2xyzmap(depth, K)
        pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), img1.reshape(-1, 3))
        keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (np.asarray(pcd.points)[:, 2] <= args.z_far)
        keep_ids = np.arange(len(np.array(pcd.points)))[keep_mask]
        pcd = pcd.select_by_index(keep_ids)
        o3d.io.write_point_cloud(f'{args.out_dir}/cloud.ply', pcd)
        logger.info(f"PCL saved to {args.out_dir}")

        logger.info("Visualization point cloud. Press ESC to exit.")
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pcd)
            vis.get_render_option().point_size = 1.0
            vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
            vis.run()
            vis.destory_window()
        except:
            logger.info(f"Maybe you are in a headless environment! Please check the results stored under {args.out_dir}!")