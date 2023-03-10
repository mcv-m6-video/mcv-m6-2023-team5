from utils import readXMLtoAnnotation, annoToDetecFormat
from metrics import voc_eval, mIoU
import numpy as np

annotationFile = "../ai_challenge_s03_c010-full_annotation.xml"
className = "car"

annot, imageNames = readXMLtoAnnotation(annotationFile)
imageIds, BB = annoToDetecFormat(annot, className)


# No confidence values, repeat N times with random values
N = 10
apSum = 0
for i in range(N):
    conf = np.random.rand(len(imageIds))
    _,_, ap = voc_eval((imageIds, conf, BB), annot, imageNames, className)
    apSum += ap
print("mAP: ", apSum/N)

miou = mIoU((imageIds, conf, BB), annot, imageNames, className)
print("mIoU: ", miou)