import cv2
import utils as u
import imageio
from matplotlib.animation import FuncAnimation
from utils import readXMLtoAnnotation, annoToDetecFormat, readTXTtoDet, randomFrame, drawBoxes
from metrics import voc_eval, mIoU, mIoU_one_image
from noise import add_noise
import numpy as np
import copy
from matplotlib import pyplot as plt


#########################################################################
green=(0, 255, 0)
red=(255, 0, 0)
blue=(0, 0, 255)
#get the gt and the detector Bboxes
vec_list, obj_list=u.reader("/home/michell/Documents/M6/AICity_data/AICity_data/train/S03/c010/gt/gt.txt")
vec_list_yolo, obj_list_yolo=u.reader("/home/michell/Documents/M6/AICity_data/AICity_data/train/S03/c010/det/det_yolo3.txt")
vec_list_ssd512, obj_list_ssd512=u.reader("/home/michell/Documents/M6/AICity_data/AICity_data/train/S03/c010/det/det_ssd512.txt")
vec_list_rcnn, obj_list_rcnn=u.reader("/home/michell/Documents/M6/AICity_data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt")

#grouped
od=u.group(obj_list)
od_yolo=od=u.group(obj_list_yolo)
od_ssd512=od=u.group(obj_list_ssd512)
od_rcnn=od=u.group(obj_list_rcnn)

#########################################################################






annotationFile = "/home/michell/Documents/M6/AICity_data/AICity_data/train/ai_challenge_s03_c010-full_annotation.xml"
videoPath = "/home/michell/Documents/M6/AICity_data/AICity_data/train/S03/c010/vdo.avi"
className = "car"
detFolder = "/home/michell/Documents/M6/AICity_data/AICity_data/train/S03/c010/det/"
detMaskRcnn = "det_mask_rcnn.txt"
detSSD = "det_ssd512.txt"
detYolo3 = "det_yolo3.txt"    
    
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


det = readTXTtoDet(detFolder + detMaskRcnn)  
print(imageNames[0])  
#miou = mIoU(det, annot, imageNames, className)

    
import cv2

# open the video file
cap = cv2.VideoCapture("/home/michell/Documents/M6/AICity_data/AICity_data/train/S03/c010/vdo.avi")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
imageIds= det[0]

frames_gif = [] 
iou_gif = []   
Index=1
while True:
    # Read frame
    ret, frame = cap.read()
    # Check if frame was successfully read
    if not ret:
        break
    frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
    u.drawBB(frame,frame_num,od,green,0)
    u.drawBB(frame,frame_num,od_yolo,red,0.5)
    if frame_num % 5 == 0:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_gif.append(img)
    
 
    miou = mIoU_one_image(det, annot, imageNames[int(frame_num)], className)
    iou_gif.append(miou)
    print("mIoU:", miou) 
     

    frame = cv2.resize(frame, (int(width/6), int(height/6)))

    cv2.imshow("Frame", frame)

    # Check for quit command
    if cv2.waitKey(1) == ord('q'):
        break

# Save the video frames as a GIF
imageio.mimsave('output.gif', frames_gif, duration=0.05)

# Release video file and close windows
cap.release()
cv2.destroyAllWindows()
    



