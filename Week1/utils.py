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



