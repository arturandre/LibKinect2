"""
Demonstrating basic usage of Kinect2 cameras.
"""
from libkinect2 import Kinect2
from libkinect2.utils import draw_skeleton, depth_map_to_image, ir_to_image
import numpy as np
import cv2
import os
import h5py

import time
# frame rate (e.g., 10 frames per second)
desired_frame_rate = 1
frame_delay = 1.0 / desired_frame_rate

# Init kinect w/all visual sensors
kinect = Kinect2(use_sensors=['color', 'depth', 'ir', 'body'])
kinect.connect()
kinect.wait_for_worker()

class Hdf5_Dataset():
    def __init__(self, output_file=None, hf_handler=None):
        """
        Either the output_file or the hf_handler need to be provided,
        if both are provided hf_handler is ignored.
        """
        if output_file is not None:
            if os.path.exists(output_file):
                self.hf_handler = h5py.File(output_file, "a")
            else:
                self.hf_handler = h5py.File(output_file, "w")
        elif hf_handler is not None:
            self.hf_handler = hf_handler

    def init_dataset(self, dtname, sample):
        self.hf_handler.create_dataset(
                dtname, (1, *sample.shape), maxshape=(None, *sample.shape),
                chunks=True)
        self.hf_handler[dtname][0] = sample

    def append_sample(self, dtname, sample):
        if dtname not in self.hf_handler:
            self.init_dataset(dtname, sample)
            print(f"Appending to inexistent dataset: {dtname}. Created.")
        self.hf_handler[dtname].resize(
            (self.hf_handler[dtname].shape[0] + 1, ) + sample.shape)
        self.hf_handler[dtname][-1] = sample

    def __enter__(self):
        pass
        return self
    
    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.hf_handler.close()

        
output_file = "E:\\frames.h5"
with Hdf5_Dataset(output_file) as hf:
    frame_index = 0
    start_time = time.time()
    next_frame_time = start_time + frame_delay
    for _, color_img, depth_map, ir_data, bodies in kinect.iter_frames():
        current_time = time.time()
        if current_time >= next_frame_time:
            bg_img = color_img
            depth_img = depth_map_to_image(depth_map)
            ir_img = ir_to_image(ir_data)

            # If it's the first frame, create the HDF5 file and dataset
            hf.append_sample("rgb", bg_img.astype('uint8'))
            hf.append_sample("depth", depth_map.astype('uint8'))
            hf.append_sample("ir", ir_img.astype('uint8'))

            # Resizing to present in screen
            ir_img = cv2.resize(ir_img, (ir_img.shape[1]//2,ir_img.shape[0]//2))
            bg_img = cv2.resize(color_img, (color_img.shape[1]//2,color_img.shape[0]//2))
            depth_img = cv2.resize(depth_img, (depth_img.shape[1]//2,depth_img.shape[0]//2))
            rect_size = 10
            
            small_rect = depth_img[
                depth_img.shape[1]//2-rect_size:depth_img.shape[1]//2+rect_size,
                depth_img.shape[0]//2-rect_size:depth_img.shape[0]//2+rect_size,:]
            print(f"rect - min depth: {small_rect.min()} max depth {small_rect.max()}")
            depth_img[ # Left
                depth_img.shape[1]//2-rect_size:depth_img.shape[1]//2+rect_size,
                depth_img.shape[0]//2-rect_size:depth_img.shape[0]//2-(rect_size-1),:] = 0
            depth_img[ # Right
                depth_img.shape[1]//2-rect_size:depth_img.shape[1]//2+rect_size,
                depth_img.shape[0]//2+(rect_size-1):depth_img.shape[0]//2+rect_size,:] = 0
            depth_img[ # Top
                depth_img.shape[1]//2-rect_size:depth_img.shape[1]//2-(rect_size-1),
                depth_img.shape[0]//2-rect_size:depth_img.shape[0]//2+rect_size,:] = 0
            depth_img[ # Bottom
                depth_img.shape[1]//2+(rect_size-1):depth_img.shape[1]//2+rect_size,
                depth_img.shape[0]//2-rect_size:depth_img.shape[0]//2+rect_size,:] = 0

            bg_img[:depth_img.shape[0],:depth_img.shape[1], :] = depth_img
            bg_img[-ir_img.shape[0]:,:ir_img.shape[1], :] = ir_img

            # Draw simple skeletons
            #body_img = np.zeros(color_img.shape)
            #for body in bodies:
            #    draw_skeleton(body_img, body)
            #bg_img[:424, -512:, :] = cv2.resize(body_img, (512, 424))
            
            cv2.imshow('sensors', bg_img)

            next_frame_time += frame_delay
            frame_index += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

kinect.disconnect()