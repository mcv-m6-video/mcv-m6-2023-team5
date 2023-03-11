import xmltodict
import cv2
import numpy as np
import itertools
from collections import OrderedDict

def reader(nombre_archivo):
    # Inicializamos la lista de vectores
    vector_list = []

    # Abrimos el archivo en modo lectura
    with open(nombre_archivo, "r") as archivo:
        # Read the content by line
        for linea in archivo:
            # split by coma
            elements = linea.strip().split(", ")
            # casting to int or floar 
            elements = [int(e) if e.isdigit() else float(e) if e.replace(".", "", 1).isdigit() else e for e in elements]
            # Add a vector to the list
            vector_list.append(elements)

    # Return the list of vectors
    return vector_list


def group(boxes):
    # Ordenamos los boxes por su atributo frame
    boxes_sorted = sorted(boxes, key=lambda box: box.frame)
    # Agrupamos los boxes por su atributo frame
    grouped = itertools.groupby(boxes_sorted, key=lambda box: box.frame)
    # Creamos un OrderedDict con las listas de boxes agrupados
    return OrderedDict((frame, list(boxes)) for frame, boxes in grouped)

def intersection_bboxes(bboxA, bboxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(bboxA.xtl, bboxB.xtl)
    yA = max(bboxA.ytl, bboxB.ytl)
    xB = min(bboxA.xbr, bboxB.xbr)
    yB = min(bboxA.ybr, bboxB.ybr)
    # return the area of intersection rectangle
    return max(0, xB - xA) * max(0, yB - yA)

def intersection_over_union(bboxA, bboxB):
    interArea = intersection_bboxes(bboxA, bboxB)
    iou = interArea / float(bboxA.area + bboxB.area - interArea)
    return iou


def intersection_over_areas(bboxA, bboxB):
    interArea = intersection_bboxes(bboxA, bboxB)
    return interArea / bboxA.area, interArea / bboxB.area
def read_detections(path, grouped=False, confidenceThr=0.5):
    """
    Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    """

    with open(path) as f:
        lines = f.readlines()

    detections = []
    for line in lines:
        det = line.split(sep=',')
        if float(det[6]) >= confidenceThr:
            detections.append(BoundingBox(
                id=int(det[1]),
                label='car',
                frame=int(det[0]) - 1,
                xtl=float(det[2]),
                ytl=float(det[3]),
                xbr=float(det[2]) + float(det[4]),
                ybr=float(det[3]) + float(det[5]),
                confidence=float(det[6])
            ))

    if grouped:
        return group_by_frame(detections)

    return detections