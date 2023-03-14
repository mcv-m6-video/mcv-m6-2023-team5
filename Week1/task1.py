from utils import readXMLtoAnnotation, annoToDetecFormat, readTXTtoDet, randomFrame, drawBoxes
from metrics import voc_eval, mIoU
from noise import add_noise
import numpy as np
import copy
from matplotlib import pyplot as plt

if __name__ == "__main__":
    annotationFile = "../../ai_challenge_s03_c010-full_annotation.xml"
    videoPath = "../../AICity_data/train/S03/c010/vdo.avi"
    className = "car"
    
    
    # Task 1
    
    # Read GT annotations
    annot, imageNames = readXMLtoAnnotation(annotationFile)
    
    noiseMean = 5.0
    noiseStd = 50.0
    probRem = 0.0
    probGen = 0.0
    
    # Get noisy annotations
    annot_noise = copy.deepcopy(annot)
    annot_noise = add_noise(annot_noise, imageNames, noiseMean, noiseStd, probRem, probGen)
    # Get imageIDs + BB
    imageIds, BB = annoToDetecFormat(annot_noise, className)
    
    # Get random frame
    frame, frameNumber = randomFrame(videoPath)
    
    # Plot noise
    colorDet = (255, 0, 0) # Red
    colorAnnot = (0, 0, 255) # Blue
    img_noise = drawBoxes(frame, BB[np.array(imageIds) == str(frameNumber),:], annot[str(frameNumber)], colorDet, colorAnnot, className)
    plt.imshow(img_noise)
    plt.show()
    
    # No confidence values, repeat N times with random values
    N = 10
    apSum = 0
    for i in range(N):
        conf = np.random.rand(len(imageIds))
        #print((imageIds, conf, BB))
        _,_, ap = voc_eval((imageIds, conf, BB), annot, imageNames, className)
        apSum += ap
    print("mAP: ", apSum/N)
    
    miou = mIoU((imageIds, conf, BB), annot, imageNames, className)
    print("mIoU: ", miou)
    
    # Task 1.2
    
    detFolder = "../../AICity_data/train/S03/c010/det/"
    detMaskRcnn = "det_mask_rcnn.txt"
    detSSD = "det_ssd512.txt"
    detYolo3 = "det_yolo3.txt"
    
    imageIds, conf, BB = readTXTtoDet(detFolder + detMaskRcnn)
    _,_, ap = voc_eval((imageIds, conf, BB), annot, imageNames, "car")
    print("Mask RCNN mAP: ", ap)
    imageIds, conf, BB = readTXTtoDet(detFolder + detSSD)
    _,_, ap = voc_eval((imageIds, conf, BB), annot, imageNames, "car")
    print("SSD512 mAP: ", ap)
    imageIds, conf, BB = readTXTtoDet(detFolder + detYolo3)
    _,_, ap = voc_eval((imageIds, conf, BB), annot, imageNames, "car")
    print("Yolo3 mAP: ", ap)
    
