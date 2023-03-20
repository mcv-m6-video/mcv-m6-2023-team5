from backgroundEstimation import gaussianModel, estimateForeground, objDet
import optuna
from metrics import voc_eval
import cv2
from matplotlib import pyplot as plt
from utils import readXMLtoAnnotation, drawBoxes
import numpy as np
    
videoPath = "/home/michell/Documents/M6/AICity_data/AICity_data/train/S03/c010/vdo.avi"
annotsPath = "/home/michell/Documents/M6/AICity_data/AICity_data/train/ai_challenge_s03_c010-full_annotation.xml"




def objective(trial):

    n = trial.suggest_int('n', 3, 7, step=2)
    nn = trial.suggest_int('nn', 3, 7, step=2)
    itn=  trial.suggest_int('iterations erotion', 1, 2,step=1)
    itnn=  trial.suggest_int('iterations dilation', 1, 4,step=1)
    contourSize = trial.suggest_int('c_size', 0, 2000,step=50)
    alpha = trial.suggest_int('alpha',5,10,step=1)
    
    
    
    annots, imageNames = readXMLtoAnnotation(annotsPath)
    backgroundMean, backgroundStd, cap = gaussianModel(videoPath)

    BB = np.zeros((0, 4))
    imgIds = []
    imageId =0

    while True:
        # Read frame
        ret, frame = cap.read()
    
        # Check if frame was successfully read
        if not ret:
            break
    
    
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    
    
        imageId = str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        #print(imageId)

        foreground = estimateForeground(frameGray, backgroundMean, backgroundStd, alpha)
        foreground = (foreground*255).astype(np.uint8)
        boxes, imageIds, img1 = objDet(foreground, imageId,n,nn,itn,itnn,contourSize)
        if imageId in annots.keys():
            plot = drawBoxes(img1, boxes, annots[imageId], [255, 0, 0], [0, 255, 0])
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
    score=apSum/N
    return score


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# print the best hyperparameters and score
print(f'Best score: {study.best_value:.3f}')
print(f'Best params: {study.best_params}')


