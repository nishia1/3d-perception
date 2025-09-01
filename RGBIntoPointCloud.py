import open3d as o3d
import numpy as np
import time

color_raw = o3d.io.read_image(o3d.data.SampleRedwoodRGBDImages().color_paths[0])
depth_raw = o3d.io.read_image(o3d.data.SampleRedwoodRGBDImages().depth_paths[0])
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)
intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=600)
vis.add_geometry(pcd)
pts = np.asarray(pcd.points)
speed = 0.5
sleep_time = 0.05

while True:
    for i in range(0, 360):
        theta_y = np.radians(i * speed)
        R = np.array([[np.cos(theta_y),0,np.sin(theta_y)],[0,1,0],[-np.sin(theta_y),0,np.cos(theta_y)]])
        pcd.points = o3d.utility.Vector3dVector(pts @ R.T)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(sleep_time)
