"""
Demonstrating basic usage of Kinect2 cameras.
"""

import numpy as np
import cv2
import os
import h5py

import time

from libkinect2.utils import depth_map_to_image
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

input_file = "E:\\frames.h5"

with Hdf5_Dataset(input_file, readmode=True) as hf:
    color_imgs = hf.get_frames("rgb")
    depth_maps = hf.get_frames("depth")
    ir_datas = hf.get_frames("ir")
    frame_index = 0
    start_time = time.time()
    next_frame_time = start_time + frame_delay
    frames = zip(color_imgs, depth_maps, ir_datas)
    bg_img, depth_map, ir_img = next(frames)
    while True:
        current_time = time.time()
        if current_time >= next_frame_time:
            # Resizing to present in screen
            bg_img = bg_img.astype('uint8')
            depth_img = depth_img.astype('uint8')
            ir_img = ir_img.astype('uint8')
            ir_img = cv2.resize(ir_img, (ir_img.shape[1]//2,ir_img.shape[0]//2))
            bg_img = cv2.resize(bg_img, (bg_img.shape[1]//2,bg_img.shape[0]//2))
            depth_img = depth_map_to_image(depth_map)
            depth_img = cv2.resize(depth_img, (depth_img.shape[1]//2,depth_img.shape[0]//2))

            bg_img[:depth_img.shape[0],:depth_img.shape[1], :] = depth_img
            bg_img[-ir_img.shape[0]:,:ir_img.shape[1], :] = ir_img

            
            cv2.imshow('sensors', bg_img)
            next_frame_time += frame_delay
            frame_index += 1
            bg_img, depth_img, ir_img = next(frames)


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

