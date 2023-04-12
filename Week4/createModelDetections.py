import torchvision.transforms as T
import cv2
import numpy as np
from detrModel import initDetr, inferDetr
import matplotlib.pyplot as plt
import time
import os



# Set device    
device = "cuda"

# Paths
seq = "./seqs/train/S04/"
modelWeights = "./challengeData3_output/checkpoint.pth"
resFolder = "./challengeData3_res/"

# Load model
model = initDetr(device, modelWeights)

for seqSub in os.listdir(seq):
    # Path
    videoPath = seq + seqSub + "/vdo.avi"

    # Load video
    cap = cv2.VideoCapture(videoPath)
    
    # Open file
    f = open(resFolder + "det_detr_" + seqSub + ".txt", "w")
    
    # Init detections
    BB = np.zeros((0, 4))
    imgIds = []
    confs = np.array([])
    
    # Not line jump in first line
    initial = True
    
    while True:
        
        # Read frame
        ret, frame = cap.read()
        
        # Check if frame was successfully read
        if not ret:
            break
        
        # Get frame
        imageId = str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        print(imageId)
        
        # Inference model
        conf, BBoxes = inferDetr(model, frame, device, 0.5)
        BBoxes = BBoxes.cpu().detach().numpy()
        conf = conf.cpu().detach().numpy()
        
        for i in range(conf.shape[0]):
            x = BBoxes[i, 0]
            y = BBoxes[i, 1]
            w = BBoxes[i, 2] - BBoxes[i, 0]
            h = BBoxes[i, 3] - BBoxes[i, 1]
            if initial:
                line = imageId + ",-1,{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},-1,-1,-1".format(x, y, w, h, conf[i])
                initial = False
            else:
                line = "\n" + imageId + ",-1,{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},-1,-1,-1".format(x, y, w, h, conf[i])
    
            f.write(line)
    # Close txt
    f.close()
    

    
    
