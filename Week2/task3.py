import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from PIL import Image
from utils import readDetectionsXML, getBBmask, drawBBs
from task3_eval import *

def plot_prec_recall_curve(prec, rec, title='Precision-Recall curve', xAxis='Recall', yAxis='Precision'):
    # plotting the points
    plt.plot(rec, prec)
    
    # naming the x axis
    plt.xlabel(xAxis)
    # naming the y axis
    plt.ylabel(yAxis)
    
    # giving a title to my graph
    plt.title(title)
    
    # function to show the plot
    plt.show()

def drawbboxes(img, boxes, color=(0, 255, 0)):
    for box in boxes:
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 3)
    return img

def generate_gif(videoPath, fgbg, videoName='video'):
    fig, ax = plt.subplots()
    plt.axis('off')
    
    vidcap = cv2.VideoCapture(videoPath)
    ims = []
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame in tqdm(range(1,num_frames//4)):
        for i in range(4):
            _, image = vidcap.read()
        fgmask = fgbg.apply(image)
        
        im = ax.imshow(fgmask, animated=True)
        ims.append([im])
    # break

    ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True, repeat_delay=10000)
    ani.save(videoName + ".gif", writer=animation.PillowWriter(fps=24))

def generate_BB_comparison(videoPath, gt, predicted, videoName='videoBoundingBox'):
    fig, ax = plt.subplots()
    plt.axis('off')

    vidcap = cv2.VideoCapture(videoPath)
    ims = []
    green = (0,255,0)
    red = (255,0,0)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame in tqdm(range(1,num_frames//4)):
        for i in range(4):
            _, image = vidcap.read()
        frame = frame*4
        if str(frame) in gt:
            image = drawBBs(image, gt[str(frame)], green)

            if str(frame) in predicted:
                image = drawBBs(image, predicted[str(frame)], red)
        im = ax.imshow(image, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True, repeat_delay=10000)
    ani.save(videoName + ".gif", writer=animation.PillowWriter(fps=24))


def closing(mask, kernel_w=3, kernel_h=3):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element)

def opening(mask, kernel_w=3, kernel_h=3):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)

def gaussianBlur(mask, kernel_w=3, kernel_h=3):
    return cv2.GaussianBlur(mask, (kernel_w,kernel_h),0)

def remove_background3(videoPath, ROIpath, fgbg):
    roi = cv2.imread(ROIpath, cv2.IMREAD_GRAYSCALE)
    
    vidcap = cv2.VideoCapture(videoPath)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    # num_frames = 1000
    detections = {}
    for frame in tqdm(range(num_frames)):
        _, image = vidcap.read()
        if frame >= num_frames // 4:

            # image = cv2.medianBlur(image, 7)
            fgmask = fgbg.apply(image)
            fgmask[fgmask==127]=0 # remove shadows
            
            roi_applied = cv2.bitwise_and(fgmask, roi)
            cleaned = opening(roi_applied, 5, 5) #initial removal of small noise
            cleaned = closing(cleaned, 50, 20) #vertical filling of areas [SWITCH TO HORIZONTAL?]
            cleaned = closing(cleaned, 20, 50) #vertical filling of areas [SWITCH TO HORIZONTAL?]
            cleaned = opening(cleaned, 7, 7) #final removal of noise

            roi_applied = cv2.bitwise_and(cleaned, roi)

            # cv2.imwrite(f'./masks/mask_{frame}.png', roi_applied)

            detections[str(frame)] = getBBmask(cleaned)
    return detections


mog2 = cv2.createBackgroundSubtractorMOG2()
lsbp = cv2.bgsegm.createBackgroundSubtractorLSBP()


data_path = 'AICity_data/AICity_data/train/S03/c010/'
gt_path = readDetectionsXML('AICity_data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml')

#MOG2
MOG2_detections = remove_background3(data_path + 'vdo.avi', data_path + 'roi.jpg', mog2)
rec, prec, ap = voc_eval(gt_path, MOG2_detections, 0.5, False)
plot_prec_recall_curve(prec, rec, f'Precision-Recall curve for MOG - AP {ap}')
generate_BB_comparison(data_path + 'vdo.avi', gt_path, MOG2_detections, videoName='videoBoundingBox_MOG2')

#LSBP
LSBP_detections = remove_background3(data_path + 'vdo.avi', data_path + 'roi.jpg', lsbp)
rec, prec, ap = voc_eval(gt_path, LSBP_detections, 0.5, False)
plot_prec_recall_curve(prec, rec, f'Precision-Recall curve for MOG2 - AP {ap}')
generate_BB_comparison(data_path + 'vdo.avi', gt_path, MOG2_detections, videoName='videoBoundingBox_LSBP')

