import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from tqdm import tqdm
import pandas as pd


class dataset_handler():
    def __init__(self, dir = ''):
        self.dir = dir
        self.data = pd.read_csv('../../Datasets/calib.txt', delimiter=' ', header=None)
        # these 2 have the left & right images files names  
        self.left_file_path = os.path.join(dir, 'image_0')
        self.right_file_path = os.path.join(dir, 'image_1')
        #! for the kitti we'll have 2 camera projection matrices
        #~change it for our case
        self.P0 = np.array(self.data.iloc[0, 1:], dtype=np.float32).reshape((3, 4)) #this has left cam projection matrix
        self.P1 = np.array(self.data.iloc[1, 1:], dtype=np.float32).reshape((3, 4)) #this has right cam projection matrix
        self.num_frames = len(os.listdir(self.left_file_path)) #this has number of frames
        self.gt = None # ground truth trajectory
        self.imheight, self.imwidth = 370, 1226 #~ image size (change it later)
        self.low_memory = False #~ remove it later
        
        
    def load_image(self, filepath):
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in tqdm(image_paths, desc="Loading images")]