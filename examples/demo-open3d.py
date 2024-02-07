from libkinect2 import Kinect2
import numpy as np
import cv2
import open3d as o3d

# Prepare Open3D Visualizer
# Simple test case
vis = o3d.visualization.Visualizer()
vis.create_window()
pcd_test = o3d.geometry.PointCloud()

# Create some points manually
points_test = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
colors_test = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float64)

pcd_test.points = o3d.utility.Vector3dVector(points_test)
pcd_test.colors = o3d.utility.Vector3dVector(colors_test)

vis.add_geometry(pcd_test)
vis.get_render_option().point_size = 10.0
vis.run()  # This will block the thread until the window is closed.
vis.destroy_window()
