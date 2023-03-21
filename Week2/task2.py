import cv2
import numpy as np
from backgroundEstimation import adaptiveModel,estimateForeground, objDet
from utils import readXMLtoAnnotation, drawBoxes, removeFirstAnnotations
from metrics import voc_eval
# initialize video capture
videoPath="/home/michell/Documents/M6/AICity_data/AICity_data/train/S03/c010/vdo.avi"
annotsPath = "/home/michell/Documents/M6/AICity_data/AICity_data/train/ai_challenge_s03_c010-full_annotation.xml"



rho=0.008
alpha=3
openKernelSize = 3
closeKernelSize = 81
minContourSize = 5000


bg_model, backgroundStd, cap =adaptiveModel(videoPath,rho)
# Load annotations
annots, imageNames = readXMLtoAnnotation(annotsPath, remParked = True)
annots, imageNames = removeFirstAnnotations(552, annots, imageNames)
BB = np.zeros((0, 4))
imgIds = []
imageId =0


num_training_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.25)
# # reset video capture to beginning of video



# loop through each frame of the video from .25% to the end
for i in range(num_training_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    # read a frame from the video
    ret, frame = cap.read()
    
    if ret:
        # convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imageId = str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        # update background model using current frame

        bg_model = rho * gray + (1 - rho) * bg_model
        backgroundStd= np.sqrt(rho*((gray-bg_model) ** 2) + (1-rho)* (backgroundStd ** 2))
        
        
        foreground = estimateForeground(gray, bg_model, backgroundStd, alpha)
        foreground = (foreground*255).astype(np.uint8)
        # display the foreground mask
        boxes, imageIds, foregroundFiltered = objDet(foreground, imageId, openKernelSize, closeKernelSize, minContourSize)
        if imageId in annots.keys():
            plot = drawBoxes(foreground, boxes, annots[imageId], [255, 0, 0], [0, 255, 0])
        else:
            plot = drawBoxes(foreground, boxes, [], [255, 0, 0], [0, 255, 0])
        plot = cv2.resize(plot, (500, 250))





        
        #foreground= cv2.resize(foreground, (0,0), fx=0.25, fy=0.25)
        cv2.imshow('plot', plot)
        #cv2.imshow('Foreground Mask', foreground)
        
        # wait for key press and exit if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
# No confidence values, repeat N times with random values
N = 10
apSum = 0
for i in range(N):
    conf = np.random.rand(len(imgIds))
    #print((imageIds, conf, BB))
    _,_, ap = voc_eval((imgIds, conf, BB), annots, imageNames)
    apSum += ap
    print("mAP_: ", apSum/N) 
print("mAP: ", apSum/N)    

# release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()
