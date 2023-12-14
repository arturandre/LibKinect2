"""
Demonstrating basic usage of Kinect2 cameras.
"""
from libkinect2 import Kinect2
from libkinect2.utils import draw_skeleton, depth_map_to_image, ir_to_image
import numpy as np
import cv2
import os
import h5py


farthresh = 10

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

        


for _, color_img, depth_map, ir_data, bodies in kinect.iter_frames():
    bg_img = color_img
    closest = depth_map.min()
    print(f"Closest: {closest}")
    rect_size = 10
    tp = depth_map.shape[1]//2-rect_size
    bt = depth_map.shape[1]//2+rect_size
    lf = depth_map.shape[0]//2-rect_size
    rt = depth_map.shape[0]//2+rect_size
    
    small_rect = depth_map[tp:bt, lf:rt]
    depth_img = depth_map < closest+farthresh
    print(f"rect - min depth: {small_rect.min()} max depth {small_rect.max()}")

    depth_img = depth_img.astype('uint8')*255
    ir_img = ir_to_image(ir_data)

    depth_img[tp:bt  , lf:lf+1] = 255 # Left
    depth_img[tp:bt  , rt-1:rt] = 255 # Right
    depth_img[tp:tp+1, lf:rt] = 255 # Top
    depth_img[bt-1:bt,lf:rt] = 255 # Bottom


    # Resizing to present in screen
    ir_img = cv2.resize(ir_img, (ir_img.shape[1]//2,ir_img.shape[0]//2))
    bg_img = cv2.resize(color_img, (color_img.shape[1]//2,color_img.shape[0]//2))
    depth_img = cv2.resize(depth_img, (depth_img.shape[1]//2,depth_img.shape[0]//2))

    bg_img[:depth_img.shape[0],:depth_img.shape[1], :] = np.dstack([depth_img]*3)
    bg_img[-ir_img.shape[0]:,:ir_img.shape[1], :] = ir_img

    # Draw simple skeletons
    #body_img = np.zeros(color_img.shape)
    #for body in bodies:
    #    draw_skeleton(body_img, body)
    #bg_img[:424, -512:, :] = cv2.resize(body_img, (512, 424))
    
    cv2.imshow('sensors', bg_img)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

kinect.disconnect()