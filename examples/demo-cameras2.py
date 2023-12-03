"""
Demonstrating basic usage of Kinect2 cameras.
"""
from libkinect2 import Kinect2
from libkinect2.utils import draw_skeleton, depth_map_to_image, ir_to_image
import numpy as np
import cv2

# Init kinect w/all visual sensors
kinect = Kinect2(use_sensors=['color', 'depth', 'ir', 'body'])
kinect.connect()
kinect.wait_for_worker()

for _, color_img, depth_map, ir_data, bodies in kinect.iter_frames():

    # Use the color image as the background
    bg_img = color_img
    # resize (y, x)
    bg_img = cv2.resize(color_img, (color_img.shape[1]//2,color_img.shape[0]//2))
    

    # Paste on the depth and ir images
    depth_img = depth_map_to_image(depth_map)
    depth_img = cv2.resize(depth_img, (depth_img.shape[1]//2,depth_img.shape[0]//2))
    bg_img[:depth_img.shape[0],:depth_img.shape[1], :] = depth_img
    
    #bg_img[-424:, :512, :] = depth_map_to_image(depth_map)

    #bg_img[-424:, -512:, :] = ir_to_image(ir_data)
    ir_img = ir_to_image(ir_data)
    ir_img = cv2.resize(ir_img, (ir_img.shape[1]//2,ir_img.shape[0]//2))
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