from backgroundEstimation import gaussianModel, estimateForeground
import cv2
from matplotlib import pyplot as plt

videoPath = "../AICity_data/train/S03/c010/vdo.avi"
backgroundMean, backgroundStd, cap = gaussianModel(videoPath)

while True:
    # Read frame
    ret, frame = cap.read()
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    # Check if frame was successfully read
    if not ret:
        break
    
    foreground = estimateForeground(frameGray, backgroundMean, backgroundStd, 1.5)
    plt.imshow(foreground)
    plt.show()