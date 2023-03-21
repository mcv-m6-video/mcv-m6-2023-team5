from backgroundEstimation import gaussianModel, estimateForeground, objDet
import optuna
from metrics import voc_eval
import cv2
from matplotlib import pyplot as plt
from utils import readXMLtoAnnotation, drawBoxes, removeFirstAnnotations
import numpy as np
import copy

videoPath = "../AICity_data/train/S03/c010/vdo.avi"
annotsPath = "../ai_challenge_s03_c010-full_annotation.xml"

def objective(trial):

    n = trial.suggest_int('n', 1, 7, step=2)
    nn = trial.suggest_int('nn', 11, 81, step=10)
    #nn = 11#51
    contourSize = trial.suggest_int('c_size', 5000, 20000,step=5000)
    
    #♠contourSize = 5000#17000
    alpha = trial.suggest_int('alpha', 5, 14,step=1)
    #alpha = 7
    
    # Open video and set frame
    capNew = cv2.VideoCapture(videoPath)
    capNew.set(1, 553)
    
    BB = np.zeros((0, 4))
    imgIds = []
    imageId =0
    int_id = 0

    while True:
        int_id += 1
        # Read frame
        ret, frame = capNew.read()
    
        # Check if frame was successfully read
        if not ret:
            break
    
    
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    
    
        imageId = str(int(capNew.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        #print(imageId)

        foreground = estimateForeground(frameGray, backgroundMean, backgroundStd, alpha)
        foreground = (foreground*255).astype(np.uint8)
        boxes, imageIds, img1 = objDet(foreground, imageId,n,nn,contourSize)
        # if int_id % 5 == 0:
        #     if imageId in annots.keys():
        #         plot = drawBoxes(foreground, boxes, annots[imageId], [255, 0, 0], [0, 255, 0])
        #         plt.imshow(plot)
        #         plt.show()
        #         plot = drawBoxes(img1, boxes, annots[imageId], [255, 0, 0], [0, 255, 0])
        #         plt.imshow(plot)
        #         plt.show()
        imgIds = imgIds + imageIds
        BB = np.vstack((BB,boxes))
        
    
    # plt.imshow(img1)
    # plt.show()
    capNew.release()
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

annots, imageNames = readXMLtoAnnotation(annotsPath, classObj = "car", remParked = True)
annots, imageNames = removeFirstAnnotations(552, annots, imageNames)
backgroundMean, backgroundStd, cap = gaussianModel(videoPath)
cap.release()
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)

# print the best hyperparameters and score
print(f'Best score: {study.best_value:.3f}')
print(f'Best params: {study.best_params}')


