from backgroundEstimation import gaussianModel, estimateForeground, objDet
from metrics import voc_eval
import cv2
from matplotlib import pyplot as plt
from utils import readXMLtoAnnotation, drawBoxes
import numpy as np

videoPath = "../AICity_data/train/S03/c010/vdo.avi"
annotsPath = "../ai_challenge_s03_c010-full_annotation.xml"
annots, imageNames = readXMLtoAnnotation(annotsPath)
backgroundMean, backgroundStd, cap = gaussianModel(videoPath)


BB = np.zeros((0, 4))
imgIds = []

while True:
    # Read frame
    ret, frame = cap.read()
    
    # Check if frame was successfully read
    if not ret:
        break
    
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    
    
    imageId = str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
    print(imageId)
    foreground = estimateForeground(frameGray, backgroundMean, backgroundStd, 5)
    foreground = (foreground*255).astype(np.uint8)
    boxes, imageIds = objDet(foreground, imageId)
    #plot = drawBoxes(foreground, boxes, annots[imageId], [255, 0, 0], [0, 255, 0])
    imgIds = imgIds + imageIds
    BB = np.vstack((BB,boxes))
    #plt.imshow(plot)
    #plt.show()

# No confidence values, repeat N times with random values
N = 10
apSum = 0
for i in range(N):
    conf = np.random.rand(len(imgIds))
    #print((imageIds, conf, BB))
    _,_, ap = voc_eval((imgIds, conf, BB), annots, imageNames)
    apSum += ap
print("mAP: ", apSum/N)