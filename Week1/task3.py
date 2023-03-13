import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import seaborn as sns


def decodeOpticalFlow(img):
    flow_x = (img[:,:,0].astype(float) - 2**15) / 128
    flow_y = (img[:,:,1].astype(float) - 2**15) / 128 
    valid = img[:,:,2].astype(bool)
    return flow_x, flow_y, valid


def calculateMetrics(img, img_gt):
    # decode the images
    x_img, y_img, valid_img = decodeOpticalFlow(img)
    x_gt, y_gt, valid_gt = decodeOpticalFlow(img_gt)

    # is calculated for every pixel
    motion_vectors = np.sqrt( (x_img - x_gt)**2 + (y_img - y_gt)**2 )

    # erroneous pixels are the ones where motion_vector > 3 and are valid pixels 
    err_pixels = (motion_vectors[valid_gt == 1] > 3).sum()

    # calculate metrics
    msen = np.mean(np.sqrt(motion_vectors[valid_gt == 1]))
    pepn = (err_pixels / (valid_gt == 1).sum()) * 100 # erroneous pixels / total valid pixels from the ground truth

    return msen, pepn




path = ".\\results" # path for Windows
# predicted
img45_name = "LKflow_000045_10.png"
img157_name = "LKflow_000157_10.png"
# ground truth
img45_GT_name = "000045_10.png"
img157_GT_name = "000157_10.png"

# get the images
img45 = cv2.cvtColor(cv2.imread(os.path.join(path, img45_name), cv2.IMREAD_UNCHANGED).astype(np.uint16), cv2.COLOR_BGR2RGB)
img157 = cv2.cvtColor(cv2.imread(os.path.join(path, img157_name), cv2.IMREAD_UNCHANGED).astype(np.uint16), cv2.COLOR_BGR2RGB)
img45_GT = cv2.cvtColor(cv2.imread(os.path.join(path, img45_GT_name), cv2.IMREAD_UNCHANGED).astype(np.uint16), cv2.COLOR_BGR2RGB)
img157_GT = cv2.cvtColor(cv2.imread(os.path.join(path, img157_GT_name), cv2.IMREAD_UNCHANGED).astype(np.uint16), cv2.COLOR_BGR2RGB)


# calculate metrics
msen45, pepn45 = calculateMetrics(img45, img45_GT)
print(msen45, pepn45)

msen157, pepn157 = calculateMetrics(img157, img157_GT)
print(msen157, pepn157)


