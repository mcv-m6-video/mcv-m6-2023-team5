import cv2
import numpy as np
import itertools
from collections import OrderedDict
import matplotlib.pyplot as plt
import copy
import xml.etree.ElementTree as ET
import random
import xmltodict
from task3_aux import DetectedVehicle

# Read the XML file of week 1 GT annotations
def readXMLtoAnnotation(annotationFile):
    # Read XML
    file = ET.parse(annotationFile)
    root = file.getroot()
    
    # Remove static objs
    staticObjs = [0, 1, 2, 3, 4, 5, 6, 8]
    
    annotations = {}
    image_ids = []
    # Find objects
    for child in root:
        if child.tag == "track":
            # Do not take into account static objs
            if not int(child.attrib["id"]) in staticObjs: 
                # Get class
                className = child.attrib["label"]
                for obj in child:
                    frame = obj.attrib["frame"]
                    xtl = float(obj.attrib["xtl"])
                    ytl = float(obj.attrib["ytl"])
                    xbr = float(obj.attrib["xbr"])
                    ybr = float(obj.attrib["ybr"])
                    bbox = [xtl, ytl, xbr, ybr]
                    if frame in image_ids:
                        annotations[frame].append({"name": className, "bbox": bbox})
                    else:
                        image_ids.append(frame)
                        annotations[frame] = [{"name": className, "bbox": bbox}]
    
    
    return annotations, image_ids

# Read txt detection lines to annot
def readTXTtoDet(txtPath):
    # Read file
    file = open(txtPath, 'r')
    lines = file.readlines()
    # Init values
    imageIds = []
    confs = []
    BB = np.zeros((0,4))
    # Insert every detection
    for line in lines:
        #frame,-1,left,top,width,height,conf,-1,-1,-1
        splitLine = line.split(",")
        # Frame
        imageIds.append(str(int(splitLine[0])-1))
        # Conf
        confs.append(float(splitLine[6]))
        # BBox
        left = float(splitLine[2])
        top = float(splitLine[3])
        width = float(splitLine[4])
        height = float(splitLine[5])
        xtl = left
        ytl = top
        xbr = left + width - 1
        ybr = top + height - 1
        BB = np.vstack((BB, np.array([xtl,ytl,xbr,ybr])))
    
    file.close()
    
    return (imageIds, np.array(confs), BB)


# Parse from annotations format to detection format
def annoToDetecFormat(annot, className):
    
    imageIds = []
    BB = np.zeros((0,4))
    
    for imageId in annot.keys():
        for obj in annot[imageId]:
            
            if obj["name"] == className:
                imageIds.append(imageId)
                BB = np.vstack((BB, obj["bbox"]))
    
    return imageIds, BB

# Draw detection and annotation boxes in image
def drawBoxes(img, det, annot, colorDet, colorAnnot):
    img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Draw annotations
    for obj in annot:
            # Draw box
            bbox = obj["bbox"]
            img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), 
                               (int(bbox[2]), int(bbox[3])), colorAnnot, 3)
    
    # Draw detections
    for i in range(det.shape[0]):
        # Draw box
        bbox = det[i,:]
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), 
                           (int(bbox[2]), int(bbox[3])), colorDet, 3)
    
    return img


def randomFrame(videoPath):
    """
    This functions reads a video and returns a random frame and the number.
    Parameters
    ----------
    videoPath : str
        video path.
    Returns
    -------
    image : numpy array
        random frame.
    randomFrameNumber : int
        random frame number.
    """
    vidcap = cv2.VideoCapture(videoPath)
    # get total number of frames
    totalFrames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    randomFrameNumber=random.randint(0, totalFrames)
    # set frame position
    vidcap.set(cv2.CAP_PROP_POS_FRAMES,randomFrameNumber)
    success, image = vidcap.read()
    
    
    return image, randomFrameNumber


def readDetectionsXML(path):
  #Generates detection dictionary where the frame number is the key and the values are the info of the corresponding detections
  
    with open(path,"r") as xml_obj:
        #coverting the xml data to Python dictionary
        gt = xmltodict.parse(xml_obj.read())
        #closing the file
        xml_obj.close()
    

    detections = {}
    for track in gt['annotations']['track']:
        #print(track)
        if track['@label'] == 'car':
            for deteccion in track['box']:
                if deteccion['@frame'] not in detections:
                    detections[deteccion['@frame']] = []
                vh = DetectedVehicle(int(deteccion['@frame']), int(track['@id']), 
                                                        float(deteccion['@xtl']), float(deteccion['@ytl']), 0, 0, 
                                                        1.0, float(deteccion['@xbr']), float(deteccion['@ybr']))
                detections[deteccion['@frame']].append(vh)

    return detections

def getBBmask(mask):
    counts, hier = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    detectedElems = []
    for cont in counts: #First box is always the background
        x,y,w,h = cv2.boundingRect(cont)
        
        if 29 < w < 593 and 12 < h < 442: #Condition based on GT minimums* and maximums
            if 0.4 < w/h < 2.5: #Condition to avoid too elongated boxes
                b = DetectedVehicle(0, -1, float(x), float(y), float(w), float(h), float(-1))
                detectedElems.append(b)

    return detectedElems

def drawBBs(image, predictions, color):
    for b in predictions:
        tl = (int(b.xtl), int(b.ytl))
        br = (int(b.xbr), int(b.ybr))
        image = cv2.rectangle(image, tl, br, color, 2)

    return image
