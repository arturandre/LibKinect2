"""
Demonstrating basic usage of Kinect2 cameras.
"""
from libkinect2 import Kinect2

from libkinect2.utils import draw_skeleton, depth_map_to_image, ir_to_image
import numpy as np
import cv2
import os
import h5py
from skimage import morphology

import time

from libkinect2.utils import depth_map_to_image
# Init kinect w/all visual sensors
kinect = Kinect2(use_sensors=['color', 'depth', 'ir', 'body'])
kinect.connect()
kinect.wait_for_worker()
mindist = 400
farthresh = 200

# frame rate (e.g., 10 frames per second)
desired_frame_rate = 5
frame_delay = 1.0 / desired_frame_rate


class Hdf5_Dataset():
    def __init__(self, io_file=None, hf_handler=None, readmode=False):
        """
        Either the output_file or the hf_handler need to be provided,
        if both are provided hf_handler is ignored.
        """
        self.readmode = readmode
        if self.readmode:
            self.hf_handler = h5py.File(io_file, "r")
        else:
            if io_file is not None:
                if os.path.exists(io_file):
                    self.hf_handler = h5py.File(io_file, "a")
                else:
                    self.hf_handler = h5py.File(io_file, "w")
            elif hf_handler is not None:
                self.hf_handler = hf_handler

    def get_frames(self, dtname):
        if not self.readmode:
            raise Exception("Datasets can only be read when readmode = True!")
        return self.hf_handler[dtname][:]

    def init_dataset(self, dtname, sample):
        if self.readmode:
            raise Exception("Can't create datasets in readmode!")
        self.hf_handler.create_dataset(
                dtname, (1, *sample.shape), maxshape=(None, *sample.shape),
                chunks=True)

    def append_sample(self, dtname, sample):
        if self.readmode:
            raise Exception("Can't create datasets in readmode!")
        if dtname not in self.hf_handler:
            self.init_dataset(dtname, sample)
            print(f"Appending to inexistent dataset: {dtname}. Created.")
        self.hf_handler[dtname].resize(
            (self.hf_handler[dtname].shape[0] + 1, ) + sample.shape)

    def __enter__(self):
        pass
        return self
    
    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.hf_handler.close()

input_file = "C:\\Users\\NTUG3\\Documents\\LibKinect2\\frames_box.h5"
sq = morphology.square(width=7)

for _, color_img, depth_map, ir_data, bodies in kinect.iter_frames():
    bg_img = color_img
    cv2.imshow('sensors', bg_img)
    # Resizing to present in screen
    closest = depth_map[depth_map > mindist].min()
    mindp = depth_map[depth_map > mindist].min()
    maxdp = depth_map[depth_map > mindist].max()
    rect_size = 10
    tp = depth_map.shape[1]//2-rect_size
    bt = depth_map.shape[1]//2+rect_size
    lf = depth_map.shape[0]//2-rect_size
    rt = depth_map.shape[0]//2+rect_size
    depth_map[:,:100, :] = 0
    #small_rect = depth_map[tp:bt, lf:rt]
    #assert img is not None, "file could not be read, check with os.path.exists()"
    depth_map = morphology.erosion(depth_map[:,:,0], sq)
    depth_map = np.expand_dims(depth_map, -1)
    depth_img = (depth_map > closest) & (depth_map < closest+farthresh)
    print(f"Total - min depth: {mindp} max depth {maxdp}")
    depth_img = depth_img.astype('uint8')*255
    ###########################################################################
    contours, _ = cv2.findContours(depth_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    convex_hull_image = np.zeros_like(depth_img)
    
    for contour in contours:
        convex_hull = cv2.convexHull(contour)
    #    # Draw the convex hull on the empty image
    #    cv2.drawContours(convex_hull_image, [convex_hull], 0, 255, -1)
    if (convex_hull.shape[0] > 1):
        print(convex_hull.shape)
        minx = convex_hull.squeeze()[:,0].min()
        maxx = convex_hull.squeeze()[:,0].max()
        dx = maxx-minx
        miny = convex_hull.squeeze()[:,1].min()
        maxy = convex_hull.squeeze()[:,1].max()
        dy = maxy-miny
        area = (maxx-minx)*(maxy-miny)
        aux = depth_map
        aux[depth_img==0] = 0
        mean_dist = 1 
        min_dist = 0                
        if len(aux[aux>0]) > 0:
            min_dist = aux[aux>0].min()
            mean_dist = aux[aux>0].mean()
        else:
            print(f"depth_img {depth_img.min()} {depth_img.max()}")
        max_dist = aux.max()
        # Kinect gives points in mm ref: https://stackoverflow.com/a/9678900/3562468
        print(f"Closest - min depth: {min_dist} max depth {max_dist} mean_dist {mean_dist}")
        print(f"dx {dx} dy {dy} area {area} ")

        # Known parameters
        # REF: https://stackoverflow.com/a/45481222/3562468
        box_cm = 35 # Referencial
        # Ref: https://www.semanticscholar.org/paper/Calibration-of-Kinect-for-Xbox-One-and-Comparison-Pagliari-Pinto/6efdd37a71f4cd3a7a0c82c89eabcbb223a11ea3
        focal_length = 400 # 365.7
        horizontal_fov_deg = 70.6
        depth_map_width = 512  
        
        # Calculate degrees per pixel in the horizontal direction
        degrees_per_pixel_horizontal = horizontal_fov_deg / depth_map_width
        
        # Calculate object width in pixels based on known width
        px_to_cm = dx*((mean_dist/10)/focal_length) 
        
        print(f"The object's width in pixels is approximately {px_to_cm:.2f} cm.")

    ###########################################################################
    bg_img = bg_img.astype('uint8')
    ir_img = depth_img #depth_img
    print(f"ir_img {ir_img.min()} {ir_img.max()}")
    #ir_img = depth_map #depth_img
    #ir_img[depth_img==0] = 0 #depth_img
    depth_img = depth_map_to_image(depth_map)
    depth_img = depth_img.astype('uint8')
    #ir_img = ir_img.astype('uint8')
    ir_img = cv2.resize(ir_img, (ir_img.shape[1]//2,ir_img.shape[0]//2))
    bg_img = cv2.resize(bg_img, (bg_img.shape[1]//2,bg_img.shape[0]//2))
    if depth_img.shape[2] == 1:
        depth_img = np.dstack([depth_img]*3)
    if (len(ir_img.shape) == 2) or (ir_img.shape[2] == 1):
        ir_img = np.dstack([ir_img]*3)
    depth_img = cv2.resize(depth_img, (depth_img.shape[1]//2,depth_img.shape[0]//2))

    bg_img[:depth_img.shape[0],:depth_img.shape[1], :] = depth_img
    bg_img[-ir_img.shape[0]:,:ir_img.shape[1], :] = ir_img
    
    cv2.imshow('sensors', bg_img)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

