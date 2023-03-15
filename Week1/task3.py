import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import seaborn as sns


def decodeOpticalFlow(img):
    """
    Determine the optical flow in x and y direction
    
    Args: The numpy ndarrray of the image
    Return:
        Optical flow estimation in X and Y direction.
    """
    flow_x = (img[:,:,0].astype(float) - 2. ** 15) / 64.0
    flow_y = (img[:,:,1].astype(float) - 2. ** 15) / 64.0
    valid = img[:,:,2].astype(bool)
    return flow_x, flow_y, valid


def calculateMetrics(img, img_gt, outputFile):
    """
    Calculating The Mean Square Error and Percentage of Erroneous Pixels in non-ocluded areas
    
    Args:
        img: the predicted numpy.ndarray
        img_gt: the groundtruth of the prediction. It is also a numpy.ndarray
        ouputFile: Where you want to save the result.
    Return:
        msen: Mean Square Error in non-ocluded areas.
        pepn: Percentage of Erroneous Pixels in non-ocluded areas.
    """
    # decode the images
    x_img, y_img, valid_img = decodeOpticalFlow(img)
    x_gt, y_gt, valid_gt = decodeOpticalFlow(img_gt)

    # is calculated for every pixel
    motion_vectors = np.sqrt( np.square(x_img - x_gt) + np.square(y_img - y_gt) )

    plotError(motion_vectors, outputFile)
    plotValid(valid_gt, outputFile)
    plotErrorHistogram(motion_vectors[valid_gt == 1], outputFile)

    # erroneous pixels are the ones where motion_vector > 3 and are valid pixels 
    err_pixels = (motion_vectors[valid_gt == 1] > 3).sum()

    # calculate metrics
    msen = np.mean((motion_vectors)[valid_gt == 1])
    pepn = (err_pixels / (valid_gt == 1).sum()) * 100 # erroneous pixels / total valid pixels from the ground truth

    return msen, pepn


def plotError(img, outputFile):
    path = ".\\plots"
    
    plt.figure(figsize=(9, 3))
    plt.title('Squared error for ' + outputFile)
    plt.imshow(img)
    plt.colorbar()
    plt.savefig(path + '\\' + outputFile +'.png')


def plotValid(img, outputFile):
    path = ".\\plots"
    
    plt.figure(figsize=(12,9))
    plt.title("Valid pixels for " + outputFile)
    plt.imshow(img, cmap='gray')
    plt.savefig(path + '\\' + outputFile +'_valid_pixels_GT.png')


def plotErrorHistogram(img, outputFile):
    path = ".\\plots"
    print(img.shape)
    # create the histogram
    histogram, bin_edges = np.histogram(img, bins=50, density=True)
    center = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.figure()
    plt.title("Image " + outputFile + " Squared Error Histogram")
    plt.xlabel("Squared error")
    plt.ylabel("Pixel percentage")
    plt.xlim([0, 50])

    plt.bar(center, histogram)
    #plt.plot(bin_edges[0:-1], histogram) 
    plt.savefig(path + '\\' + outputFile +'_error_histogram.png')


#The main function of the Task3    
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
msen45, pepn45 = calculateMetrics(img45, img45_GT, "45")
print(msen45, pepn45)

msen157, pepn157 = calculateMetrics(img157, img157_GT, "157")
print(msen157, pepn157)



