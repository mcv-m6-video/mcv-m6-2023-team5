from utils import readXMLtoAnnotation, annoToDetecFormat, readTXTtoDet
from metrics import voc_eval, mIoU
import numpy as np

annotationFile = "../ai_challenge_s03_c010-full_annotation.xml"
className = "car"


# Task 1

annot, imageNames = readXMLtoAnnotation(annotationFile)
imageIds, BB = annoToDetecFormat(annot, className)


# No confidence values, repeat N times with random values
N = 1#10
apSum = 0
for i in range(N):
    conf = np.random.rand(len(imageIds))
    _,_, ap = voc_eval((imageIds, conf, BB), annot, imageNames, className)
    apSum += ap
print("mAP: ", apSum/N)

miou = mIoU((imageIds, conf, BB), annot, imageNames, className)
print("mIoU: ", miou)

# Need to add noise

# Task 1.2

detFolder = "../AICity_data/train/S03/c010/det/"
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

# Task 2
