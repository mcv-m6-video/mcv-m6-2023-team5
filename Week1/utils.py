import xmltodict
import cv2
import numpy as np
import itertools
from collections import OrderedDict
import matplotlib.pyplot as plt
import copy
import xml.etree.ElementTree as ET
import numpy as np
import cv2

# Read the XML file of week 1 GT annotations
def readXMLtoAnnotation(annotationFile):
    # Read XML
    file = ET.parse(annotationFile)
    root = file.getroot()
    
    annotations = {}
    image_ids = []
    # Find objects
    for child in root:
        if child.tag == "track":
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
def drawBoxes(img, det, annot, colorDet, colorAnnot, className):
    img = img.copy()
    
    # Draw annotations
    for obj in annot:
        if obj["name"] == className:
            # Draw box
            bbox = obj["bbox"]
            img = cv2.rectange(img, (int(bbox[0]), int(bbox[1])), 
                               (int(bbox[2]), int(bbox[3])), colorAnnot, 3)
    
    # Draw detections
    for i in det.shape[0]:
        # Draw box
        bbox = det[i,:]
        img = cv2.rectange(img, (int(bbox[0]), int(bbox[1])), 
                           (int(bbox[2]), int(bbox[3])), colorDet, 3)
    
    return img


def group(boxes):
    # Ordenamos los boxes por su atributo frame
    boxes_sorted = sorted(boxes, key=lambda box: box.frame)
    # Agrupamos los boxes por su atributo frame
    grouped = itertools.groupby(boxes_sorted, key=lambda box: box.frame)
    # Creamos un OrderedDict con las listas de boxes agrupados
    return OrderedDict((frame, list(boxes)) for frame, boxes in grouped)

def reader(nombre_archivo):
    # Inicializamos la lista de vectores
    vector_list = []
    list_of_objects = []

    # Abrimos el archivo en modo lectura
    with open(nombre_archivo, "r") as archivo:
        # Read the content by line
        for linea in archivo:
            # split by coma
            elements = linea.strip().split(",")
            try:
               # convert each element to an integer
               elements = [float(e) for e in elements]
               # Add a vector to the list
               vector_list.append(elements)
               list_of_objects.append(BBox(elements))
            except ValueError:
                print(f"Error converting line '{linea.strip()}' to integers, double check data format or reader function")
            
    # Return the list of vectors
    return vector_list, list_of_objects

def drawBB(frame,current_frame,list_of_grouped_boxes,color,Thresh):
    if current_frame in list_of_grouped_boxes:
        frame_boxes= list_of_grouped_boxes[current_frame]
        for i in range(len(frame_boxes)):
            if frame_boxes[i].conf > Thresh:
                   x, y, w, h = frame_boxes[i].left, frame_boxes[i].top, frame_boxes[i].width, frame_boxes[i].height
                   cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)  # Replace (x, y, w, h) with your bounding box coordinates
    return frame

class BBox:
    def __init__(self, data):
        self.frame = int(data[0])
        self.id = int(data[1])
        self.left = int(data[2])
        self.top = int(data[3])
        self.width = int(data[4])
        self.height = int(data[5])
        self.conf = float(data[6])
        self.x_center = float(data[7])
        self.y_center = float(data[8])
        self.z_center = float(data[9])
    
    @property
    def area(self):
        return self.width * self.height
    
    @property
    def bottom(self):
        return self.top + self.height
    
    @property
    def right(self):
        return self.left + self.width
    
    @property
    def top_left(self):
        return (self.left, self.top)
    
    @property
    def top_right(self):
        return (self.right, self.top)
    
    @property
    def bottom_left(self):
        return (self.left, self.bottom)
    
    @property
    def bottom_right(self):
        return (self.right, self.bottom)
    
    @property
    def x_min(self):
        return self.left
    
    @property
    def x_max(self):
        return self.right
    
    @property
    def y_min(self):
        return self.top
    
    @property
    def y_max(self):
        return self.bottom
    @property
    def x(self):
        return self.x_center
    @property
    def y(self):
        return self.y_center
    
    
    #for the plot gif
def update(frame):
    x = range(10)
    y = [frame * i for i in x]
    plt.plot(x, y)
    plt.title(f'Frame {frame}')







