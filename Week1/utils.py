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

