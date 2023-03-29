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
import cv2
from collections import defaultdict

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
    
    #AHORA iteramos en cada una de los directorios: "C0001,C0002,...C0067"
    #for directorio in sorted_folders[1:]: #el [1:] lo ponemos para no coger el primer archivo
    
    current_dir = str(initial_dir + "/" + sorted_folders[1])
    os.chdir(current_dir) #Aqui estamos cambiando el directorio en cada iteracion para analizar cada uno de los directorios
    
    lista_de_imagenes = [ima for ima in os.listdir(current_dir) if ima.endswith(".png")]
    for images in os.listdir(current_dir):
        if (images.endswith(".png")):
            aux_img.append(images)
    
    
    #Aqui chequeamos que nos econtramos en el directorio correcto
    print("El directorio en el que te encuentras es el siguiente :", os.getcwd() + "\n")
    for images in os.listdir(current_dir):
        if (images.endswith(".png")):
            aux_img.append(images)
        if (images == "ground_truth.txt"):
            aux_txt.append(images)
            #df = str(os.getcwd() + "/" +str(aux_txt))
            #df = pd.read_csv(os.getcwd() + "/" + str(images), sep="  ",engine="python")
            current_directory = str(os.getcwd())
            image_data = funciones.read_ground_truth(current_directory,images,sorted(aux_img))
            funciones.boundingBox(current_directory,image_data)
    #print("Estos son los datos de las imagenes",img)
    #img.show()


    """
        for images in os.listdir(current_dir):
            if (images.endswith(".png")):
                aux_img.append(images)
                sorted_images = funciones.list_sorting(aux_img)
            if (images == "ground_truth.txt"):
                aux_txt.append(images)
                #df = str(os.getcwd() + "/" +str(aux_txt))
                df = pd.read_csv(os.getcwd() + "/" + str(images), sep="  ",engine="python")
                #print(df)
                #Ahora llamamos a la funcion que nos va a extraer las coordenadas de cada una de las imagenes que tenemos que analizar
                image_data = funciones.read_ground_truth(df)

    """

    """
    #En cada iteracion nos metemos en un directorio
        for images in os.listdir(current_dir):
            if (images.endswith(".png")):
                aux_img.append(images)
            if (images.endswith(".txt")):
                aux_txt.append(images)
        sorted_images = funciones.list_sorting(aux_img)
        print("\n")
        print(sorted_images)

    #Aqui vamos a probar a imprimir el groundtruth
    #df = pd.read_csv("ground_truth.txt", sep="  ")
    #print(df)
    #print(aux_txt)
    """

main()

