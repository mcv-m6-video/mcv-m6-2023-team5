import cv2
import matplotlib
matplotlib.use('agg')
import imageio
from utils import readXMLtoAnnotation, readTXTtoDet, drawBoxes
from metrics import mIoU
from noise import add_noise
import numpy as np
import copy
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure



green=(0, 255, 0)
red=(255, 0, 0)
blue=(0, 0, 255)

annotationFile = "../ai_challenge_s03_c010-full_annotation.xml"
videoPath = "../AICity_data/train/S03/c010/vdo.avi"
className = "car"
detFolder = "../AICity_data/train/S03/c010/det/"
detMaskRcnn = "det_mask_rcnn.txt"
detSSD = "det_ssd512.txt"
detYolo3 = "det_yolo3.txt"    
    

# Read GT annotations
annot, imageNames = readXMLtoAnnotation(annotationFile)
    
noiseMean = 5.0
noiseStd = 50.0
probRem = 0.0
probGen = 0.0
    
# Get noisy annotations
annot_noise = copy.deepcopy(annot)
annot_noise = add_noise(annot_noise, imageNames, noiseMean, noiseStd, probRem, probGen)
    
detModel = "mask_rcnn"
detValues = detMaskRcnn
det = readTXTtoDet(detFolder + detValues)  

# open the video file
cap = cv2.VideoCapture("../AICity_data/train/S03/c010/vdo.avi")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
imageIds= det[0]

frames_gif = [] 
iou_gif = []   
frame_idx = []
while True:
    # Read frame
    ret, frame = cap.read()
    # Check if frame was successfully read
    if not ret:
        break
    frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
    # Get frame
    annotFrame = {str(frame_num): annot[str(frame_num)]}
    frameIndex = np.array(det[0]) == str(frame_num)
    detFrame = (np.array(det[0])[frameIndex], det[1][frameIndex], det[2][frameIndex,:])
    # Draw box
    frameDraw = drawBoxes(frame, detFrame[2], annotFrame[str(frame_num)], red, blue, className)
    frameDraw = cv2.resize(frameDraw, (500,250))
    
    
    # Compute mIoU
    miou = mIoU(detFrame, annotFrame, [str(frame_num)], className)
    iou_gif.append(miou)
    frame_idx.append(frame_num)
    
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.set(xlim=(0, int(det[0][-1])))
    ax.set_xlabel("Frame")
    ax.set_ylabel("mIoU")
    ax.plot(frame_idx, iou_gif, linewidth=1)
    
    canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(canvas.get_width_height()[::-1] + (3,))
    image = cv2.resize(image, (500,250))
    
    # Concatenate two images
    finalImage = np.vstack((frameDraw, image))
    frames_gif.append(finalImage)
    

# Save the video frames as a GIF
imageio.mimsave(detModel + '.gif', frames_gif, fps=10)

# Release video file and close windows
cap.release()
    



