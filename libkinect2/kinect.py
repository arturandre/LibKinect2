from pkg_resources import resource_filename
import numpy as np
import ctypes

F_SENSOR_COLOR = 0x00000001
F_SENSOR_DEPTH = 0x00000010

COLOR_WIDTH = 1920
COLOR_HEIGHT = 1080
COLOR_CHANNELS = 4
DEPTH_WIDTH = 512
DEPTH_HEIGHT = 424

kinectDLL = ctypes.cdll.LoadLibrary(resource_filename(__name__, 'data/Kinect2-API.dll'))

kinectDLL.init_kinect.argtypes = [ctypes.c_int]
kinectDLL.init_kinect.restype = ctypes.c_bool

kinectDLL.get_color_data.argtypes = [np.ctypeslib.ndpointer(dtype=np.uint8)]
kinectDLL.get_color_data.restype = ctypes.c_bool

kinectDLL.get_depth_data.argtypes = [np.ctypeslib.ndpointer(dtype=np.uint16)]
kinectDLL.get_depth_data.restype = ctypes.c_bool

class Kinect2:

    def __init__(self, use_color=True, use_depth=True):
        self.sensor_flags = 0
        if use_color:
            self.sensor_flags |= F_SENSOR_COLOR
        if use_depth:
            self.sensor_flags |= F_SENSOR_DEPTH

    def connect(self):
        if not kinectDLL.init_kinect(self.sensor_flags):
            raise IOError('Unable to init Kinect2 Sensor.')
        return True

    def get_color_image(self):
        color_ary = np.zeros((COLOR_HEIGHT, COLOR_WIDTH, COLOR_CHANNELS), np.uint8)
        if kinectDLL.get_color_data(color_ary):
            return color_ary
        return None

    def get_depth_image(self):
        depth_ary = np.zeros((DEPTH_HEIGHT, DEPTH_WIDTH, 1), np.uint16)
        if kinectDLL.get_depth_data(depth_ary):
            return depth_ary
        return None