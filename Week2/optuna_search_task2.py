from backgroundEstimation import gaussianModel, estimateForeground, objDet,updateBackground
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

    newBackgroundMean = backgroundMean.copy()
    newBackgroundStd = backgroundStd.copy()
    
    openKernelSize = 3
    closeKernelSize = 81
    minContourSize = 1000
    rho = trial.suggest_float('rho', 0.001, 0.05)
    alpha = trial.suggest_int('alpha', 2, 10, step=1)
    
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
    
    
        # Convert to grayscale
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get frame
        imageId = str(int(capNew.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        #print(imageId)
        # Estimate foreground
        foreground = estimateForeground(frameGray, newBackgroundMean, newBackgroundStd, alpha)
        foreground = (foreground*255).astype(np.uint8)
        
        # Adapt background
        newBackgroundMean, newBackgroundStd = updateBackground(frameGray, foreground, newBackgroundMean, newBackgroundStd, rho)
        
        # Detect bboxes
        boxes, imageIds, foregroundFiltered = objDet(foreground, imageId, openKernelSize, closeKernelSize, minContourSize)
        
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

# Load annotations
annots, imageNames = readXMLtoAnnotation(annotsPath, remParked = True)
annots, imageNames = removeFirstAnnotations(552, annots, imageNames)
# Create model
backgroundMean, backgroundStd, cap = gaussianModel(videoPath)
cap.release()
# Do parameter search
study_name = 'adaptative_search3'  # Unique identifier of the study.
study = optuna.create_study(study_name=study_name, storage='sqlite:///example.db', direction='maximize')
study.optimize(objective, n_trials=50, n_jobs = 10)

fig = optuna.visualization.plot_contour(study, params=["rho", "alpha"])
fig.show()
fig.write_image("adapatative_search3.png")


# print the best hyperparameters and score
print(f'Best score: {study.best_value:.3f}')
print(f'Best params: {study.best_params}')


