from ctypes import sizeof
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import os
from os import listdir
import tensorflow as tf
import pathlib as Path
import imghdr
from collections import defaultdict
import json

#Aqui importamos las funciones que hemos creado nosotros
import funciones

#Esto es para evitar problemas de memoria
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def main():
    aux_img = []
    aux_txt = []
    # get the path/directory
    initial_dir = "/Users/pablosreyero/Documents/Universidad/TFG/tfg-psr/data/Castings"
    x = []
    
    for folders in os.listdir(initial_dir):
        x.append(folders)

    #Ahora tenemos que ordenar la lista de las carpetas que hemos extraido
    sorted_folders = funciones.list_sorting(x)

    
    current_dir = str(initial_dir + "/" + sorted_folders[1])
    os.chdir(current_dir) #Aqui estamos cambiando el directorio en cada iteracion para analizar cada uno de los directorios
    
    lista_de_imagenes = [ima for ima in os.listdir(current_dir) if ima.endswith(".png")]
    for images in os.listdir(current_dir):
        if (images.endswith(".png")):
            aux_img.append(images)
    
    
    #Aqui chequeamos que nos encontramos en el directorio correcto
    print("El directorio en el que te encuentras es el siguiente :", os.getcwd() + "\n")
    print("\n")

    buff = []
    for images in os.listdir(current_dir):
        buff.append(images) #esto me sirve para leer lo que hay en el directorio

    if ("ground_truth.txt" in buff):
        for images in os.listdir(current_dir):
            if (images.endswith(".png")):
                aux_img.append(images)
            if (images == "ground_truth.txt"):
                aux_txt.append(images)
                current_directory = str(os.getcwd())

                #A la hora de leer las imagenes del directorio C001, me sacaba la imagen 8 repetida, por ello SE ELIMINAN LOS DUPLICADOS EN LA SIGUIENTE LINEA.
                aux_img = list(dict.fromkeys(aux_img))

                image_data = funciones.read_ground_truth(current_directory,images,sorted(aux_img))
                funciones.boundingBox(current_directory,image_data)
    else:
        print("El directorio: " + str(current_dir) + " NO contiene imágenes con defectos")
        print("\n")
main()