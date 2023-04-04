import os

mydir = os.getcwd() # would be the MAIN folder
mydir_tmp = mydir + "/pyflow/" # add the testA folder name
mydir_new = os.chdir(mydir_tmp) # change the current working directory
mydir = os.getcwd() # set the main directory again, now it calls testA

import numpy as np
import pyflow
from PIL import Image
import time
import cv2

class PyFlowOptFlow:
    def __init__(self, alpha = 0.012, ratio = 0.75, minWidth = 20, nOuterFPIterations = 7,
                 nInnerFPIterations = 1, nSORIterations = 30, colType = 1):
        
        # Flow Options:
        self.alpha = alpha
        self.ratio = ratio
        self.minWidth = minWidth
        self.nOuterFPIterations = nOuterFPIterations
        self.nInnerFPIterations = nInnerFPIterations
        self.nSORIterations = nSORIterations
        self.colType = colType  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
        
    def inference(self, im1, im2):
        im1 = im1.astype(float) / 255.
        im2 = im2.astype(float) / 255.
        
        
        # Get opt flow        
        u, v, im2W = pyflow.coarse2fine_flow(
            im1, im2, self.alpha, self.ratio, self.minWidth, self.nOuterFPIterations, 
            self.nInnerFPIterations, self.nSORIterations, self.colType)
        
        flow = np.concatenate((u[..., None], v[..., None]), axis=2)
        
        return flow

    