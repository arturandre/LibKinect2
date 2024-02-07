import time
from libkinect2 import Kinect2
import numpy as np
import cv2
import open3d as o3d

# Global configs
plot_test_points = False

# Init Kinect
kinect = Kinect2(use_sensors=['color', 'depth'], use_mappings=[('color', 'camera')])
kinect.connect()
kinect.wait_for_worker()

# Prepare Open3D Visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()
pcd = o3d.geometry.PointCloud()

# Set some visualizer properties (e.g., point size)
vis.get_render_option().point_size = 3.0  # Adjust the size as needed

def get_camera_pose(vis):
    # Assuming 'vis' is your Open3D visualizer object
    view_ctl = vis.get_view_control()
    cam_params = view_ctl.convert_to_pinhole_camera_parameters()

    # The 'up' vector
    up = cam_params.extrinsic[:3, 1]  # Up is the y-axis in the camera coordinate system

    # The 'front' vector, which is the negative z-axis in camera coordinates
    front = -cam_params.extrinsic[:3, 2]

    # The 'lookat' point is derived from camera location and front vector
    lookat = cam_params.extrinsic[:3, 3] - front

    # Print out the camera parameters
    print("Up vector:", up)
    print("Front vector:", front)
    print("Lookat point:", lookat)


def upview(pcd, vis):
    up = np.array([-1., 1., 0.])
    front = np.array([0.03504746, 0., 0.1609289])
    #lookat = np.array([-1.26748333, -3.91191914, 3.90062454])
    # Set up the view control of the visualizer
    view_ctl = vis.get_view_control()
    view_ctl.set_up(up)
    view_ctl.set_front(front)
    #view_ctl.set_lookat(lookat)
    #view_ctl.set_front(front)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

def center_view(pcd, vis):
    # Assuming pcd is your point cloud and vis is your Open3D visualizer

    # Compute the center of the point cloud
    pcd_center = pcd.get_center()

    # Compute the bounding box of the point cloud to estimate its extent
    #bounding_box = pcd.get_axis_aligned_bounding_box()
    #extent = bounding_box.get_extent()

    # Set up the view control of the visualizer
    view_ctl = vis.get_view_control()

    # Set the front view - this can be adjusted as needed
    front = np.array([0, -1, 0])  # Assuming the front of the point cloud is aligned with the z-axis

    # Set the lookat point to the center of the point cloud
    lookat = pcd_center

    # Set up vector
    up = np.array([0, 0, 1])  # This may need to be adjusted based on the orientation of your point cloud

    # Set the zoom to accommodate the extent of the bounding box
    #zoom = max(extent)

    # Apply the camera settings
    view_ctl.set_front(front)
    view_ctl.set_lookat(lookat)
    view_ctl.set_up(up)
    #view_ctl.set_zoom(zoom)

    # You can now update the geometry and render
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

first_time = True
#vis.add_geometry(pcd)
#while True:
for _, color_img, depth_map, color_cam_map in kinect.iter_frames():
    # Display color image as reference
    #cv2.imshow('sensors', color_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    h, w, _ = color_img.shape

    # Prepare data for Open3D
    points = []
    colors = []

    # Read (every 10th) point pos and color
    for y_pixel in range(0, h, 10):
        for x_pixel in range(0, w, 10):
            x, z, y = color_cam_map[y_pixel, x_pixel]
            if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z):
                continue  # Skip this point
            b, g, r = color_img[y_pixel, x_pixel]/255
            points.append([x, y, z])
            colors.append([r, g, b])
    
    if plot_test_points:
        points_test = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        colors_test = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
        points += points_test
        colors += colors_test

    # Assign points and colors to point cloud
    pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float64))
    
    # Update point cloud
    
    #if not vis.update_geometry(pcd):
    if first_time:
        vis.add_geometry(pcd)
        first_time = False
    center_view(pcd, vis)
    #get_camera_pose(vis)
    # Up vector: [ 0.         -1.          0.98694987]
    # Front vector: [0.03504746 0.         0.1609289 ]
    # Lookat point: [-1.26748333 -3.91191914  3.90062454]
    #center_view(pcd, vis)
    #vis.update_geometry(pcd)
    #vis.poll_events()
    #vis.update_renderer()
    time.sleep(0.001)

kinect.disconnect()
vis.destroy_window()
