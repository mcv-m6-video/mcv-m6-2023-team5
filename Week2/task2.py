import cv2
import numpy as np
from backgroundEstimation import adaptiveModel,estimateForeground
from utils import readXMLtoAnnotation, drawBoxes
# initialize video capture
videoPath="/home/michell/Documents/M6/AICity_data/AICity_data/train/S03/c010/vdo.avi"
annotsPath = "/home/michell/Documents/M6/AICity_data/AICity_data/train/ai_challenge_s03_c010-full_annotation.xml"



alpha=0.01




bg_model, backgroundStd, cap =adaptiveModel(videoPath,alpha)
annots, imageNames = readXMLtoAnnotation(annotsPath)
BB = np.zeros((0, 4))
mgIds = []
imageId =0


num_training_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.25)
# # reset video capture to beginning of video
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


# loop through each frame of the video from .25% to the end
for i in range(num_training_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    # read a frame from the video
    ret, frame = cap.read()
    
    if ret:
        # convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # update background model using current frame

        bg_model = alpha * gray.astype(np.float32) + (1 - alpha) * bg_model
        backgroundStd= np.sqrt(alpha*((gray.astype(np.float32)-bg_model) ** 2) + (1-alpha)* (backgroundStd ** 2))
        
        
        #check here, how do we use the STD to get the diff?
        # compute absolute difference between current frame and background model
        diff = cv2.absdiff(gray.astype(np.float32), bg_model.astype(np.float32))
        
        # apply threshold to foreground mask
        thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)[1]
        
        # perform morphological operations to remove noise
        # pending test with the basic morph operations and find the best conbination with parameter search
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # resize foreground mask to 1/4 of original size
        fg_mask = cv2.resize(fg_mask, (0,0), fx=0.25, fy=0.25)
        
        # display the foreground mask
        cv2.imshow('Foreground Mask', fg_mask)
        
        # wait for key press and exit if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()