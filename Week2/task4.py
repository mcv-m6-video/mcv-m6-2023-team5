from backgroundEstimation import gaussianModel, estimateForeground, objDet
from metrics import voc_eval, mIoU
import cv2
from matplotlib import pyplot as plt
from utils import readXMLtoAnnotation, drawBoxes, removeFirstAnnotations
import numpy as np
import imageio

# Paths
videoPath = "../AICity_data/train/S03/c010/vdo.avi"
annotsPath = "../ai_challenge_s03_c010-full_annotation.xml"
# Parameters
# Set parameters (found with optuna search)
alpha = 2
openKernelSize = 3
closeKernelSize = 81
minContourSize = 5000

# Load annotations
annots, imageNames = readXMLtoAnnotation(annotsPath, remParked = True)
annots, imageNames = removeFirstAnnotations(552, annots, imageNames)

# Create single gaussian
backgroundMean, backgroundStd, cap = gaussianModel(videoPath, False)

# Init detections
BB = np.zeros((0, 4))
imgIds = []

# Init gif frames
int_id = 1
gif_filtered = []
gif_foreground = []
gif_original = []
gif_boxes = []

while True:
    int_id += 1
    
    # Read frame
    ret, frame = cap.read()
    
    # Check if frame was successfully read
    if not ret:
        break
    
    # Convert to grayscale
    # frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frameGray = frame
    # frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 0]
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 2]
    
    
    # Get frame
    imageId = str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)

    # Estimate foreground
    foreground = estimateForeground(frameGray, backgroundMean, backgroundStd, alpha)
    foreground = (foreground*255).astype(np.uint8)
    
    # # view
    # cv2.imshow("win", foreground)
    # if cv2.waitKey(25) == ord("q"):
    #     break
    
    # Detect bboxes
    boxes, imageIds, foregroundFiltered = objDet(foreground, imageId, openKernelSize, closeKernelSize, minContourSize)
    
    # Plot
    if int_id % 10 == 0:
        print(imageId)
        # Resize images
        plotOrig = cv2.resize(frameGray, (500, 250))
        plotForeground = cv2.resize(foreground, (500, 250))
        plotFiltered = cv2.resize(foregroundFiltered, (500, 250))
        # Plot found bboxes
        if imageId in annots.keys():
            plotBoxes = drawBoxes(frameGray, boxes, annots[imageId], [255, 0, 0], [0, 255, 0])
        else:
            plotBoxes = drawBoxes(frameGray, boxes, [], [255, 0, 0], [0, 255, 0])
        plotBoxes = cv2.resize(plotBoxes, (500, 250))
        
        # Store plots
        gif_original.append(plotOrig)
        gif_foreground.append(plotForeground)
        gif_filtered.append(plotFiltered)
        gif_boxes.append(plotBoxes)
        
    # Store predictions
    imgIds = imgIds + imageIds
    BB = np.vstack((BB,boxes))

# No confidence values, repeat N times with random values
N = 10
apSum = 0
for i in range(N):
    conf = np.random.rand(len(imgIds))
    #print((imageIds, conf, BB))
    _,_, ap = voc_eval((imgIds, conf, BB), annots, imageNames)
    apSum += ap
print("mAP: ", apSum/N)

# Estimate miou
miou = mIoU((imgIds, conf, BB), annots, imageNames)
print("mIoU: ", miou)

# Save gifs
imageio.mimsave('orig' + str(alpha) + '.gif', gif_original, fps=2)
imageio.mimsave('foreground' + str(alpha) + '.gif', gif_foreground, fps=2)
imageio.mimsave('filtered' + str(alpha) + '.gif', gif_filtered, fps=2)
imageio.mimsave('boxes' + str(alpha) + '.gif', gif_boxes, fps=2)