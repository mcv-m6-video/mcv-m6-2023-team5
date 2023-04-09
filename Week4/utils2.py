import cv2
import numpy as np
import itertools
from collections import OrderedDict
import matplotlib.pyplot as plt
import copy
import xml.etree.ElementTree as ET
import random


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


def calculateMetrics(img, img_gt, valid_gt, plot = False, outputFile = "hist"):
    """
    Calculating The Mean Square Error and Percentage of Erroneous Pixels in non-ocluded areas
    
    Args:
        img: the predicted numpy.ndarray
        img_gt: the groundtruth of the prediction. It is also a numpy.ndarray
        valid_gt: the valid pixels
        ouputFile: Where you want to save the result.
    Return:
        msen: Mean Square Error in non-ocluded areas.
        pepn: Percentage of Erroneous Pixels in non-ocluded areas.
    """
    # Get x and y values
    x_gt = img_gt[:,:,0]
    y_gt = img_gt[:,:,1]
    x_img = img[:,:,0]
    y_img = img[:,:,1]

    # is calculated for every pixel
    motion_vectors = np.sqrt( np.square(x_img - x_gt) + np.square(y_img - y_gt) )

    if plot:
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

# Read the XML file of week 1 GT annotations
def readXMLtoAnnotation(annotationFile, remParked = False):
    # Read XML
    file = ET.parse(annotationFile)
    root = file.getroot()
    
    annotations = {}
    image_ids = []
    # Find objects
    for child in root:
        if child.tag == "track":
            # Get class
            className = child.attrib["label"]
            #if className != classObj:
            #    continue
            for obj in child:
                if className == "car":
                    objParked = obj[0].text
                    # Do not store if it is parked and we want to remove parked objects
                    if objParked=="true" and remParked:
                        continue
                frame = obj.attrib["frame"]
                xtl = float(obj.attrib["xtl"])
                ytl = float(obj.attrib["ytl"])
                xbr = float(obj.attrib["xbr"])
                ybr = float(obj.attrib["ybr"])
                bbox = [xtl, ytl, xbr, ybr]
                if frame in image_ids:
                    annotations[frame].append({"name": className, "bbox": bbox})
                else:
                    image_ids.append(frame)
                    annotations[frame] = [{"name": className, "bbox": bbox}]
    
    
    return annotations, image_ids

def removeFirstAnnotations(stopFrame, annots, imageIds):
    """
    This function removes the annotations until a certain number of frame

    Parameters
    ----------
    stopFrame : int
        Until what number of frame remove the annotations.
    annots : dict
        Dictionary of annotations.
    imageIds : list
        List of frames of annotations.

    Returns
    -------
    newAnnots : dict
        Annotations with the removed frames.
    newImageIds : list
        Annotation frame ids with removed frames.

    """
    newAnnots = {}
    newImageIds = []
    
    # Store only next frame annotations
    for frameNum in imageIds:
        num = int(frameNum)
        if num > stopFrame:
            newImageIds.append(frameNum)
            newAnnots[frameNum] = annots[frameNum]
            
    return newAnnots, newImageIds

# Read txt detection lines to annot
def readTXTtoDet(txtPath):
    # Read file
    file = open(txtPath, 'r')
    lines = file.readlines()
    # Init values
    imageIds = []
    confs = []
    BB = np.zeros((0,4))
    # Insert every detection
    for line in lines:
        #frame,-1,left,top,width,height,conf,-1,-1,-1
        splitLine = line.split(",")
        # Frame
        imageIds.append(str(int(splitLine[0])-1))
        # Conf
        confs.append(float(splitLine[6]))
        # BBox
        left = float(splitLine[2])
        top = float(splitLine[3])
        width = float(splitLine[4])
        height = float(splitLine[5])
        xtl = left
        ytl = top
        xbr = left + width - 1
        ybr = top + height - 1
        BB = np.vstack((BB, np.array([xtl,ytl,xbr,ybr])))
    
    file.close()
    
    return (imageIds, np.array(confs), BB)


# Parse from annotations format to detection format
def annoToDetecFormat(annot, className):
    
    imageIds = []
    BB = np.zeros((0,4))
    
    for imageId in annot.keys():
        for obj in annot[imageId]:
            
            if obj["name"] == className:
                imageIds.append(imageId)
                BB = np.vstack((BB, obj["bbox"]))
    
    return imageIds, BB

# Draw detection and annotation boxes in image
def drawBoxes(img, det, annot, colorDet, colorAnnot):
    img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Draw annotations
    for obj in annot:
            # Draw box
            bbox = obj["bbox"]
            img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), 
                               (int(bbox[2]), int(bbox[3])), colorAnnot, 3)
    
    # Draw detections
    for i in range(det.shape[0]):
        # Draw box
        bbox = det[i,:]
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), 
                           (int(bbox[2]), int(bbox[3])), colorDet, 3)
    
    return img


def randomFrame(videoPath):
    """
    This functions reads a video and returns a random frame and the number.

    Parameters
    ----------
    videoPath : str
        video path.

    Returns
    -------
    image : numpy array
        random frame.
    randomFrameNumber : int
        random frame number.

    """
    vidcap = cv2.VideoCapture(videoPath)
    # get total number of frames
    totalFrames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    randomFrameNumber=random.randint(0, totalFrames)
    # set frame position
    vidcap.set(cv2.CAP_PROP_POS_FRAMES,randomFrameNumber)
    success, image = vidcap.read()
    
    
    return image, randomFrameNumber
