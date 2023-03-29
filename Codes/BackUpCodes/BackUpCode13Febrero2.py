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

#Aqui importamos las funciones que hemos creado nosotros
import funciones

def main():

    imagenes = []
    text = []
    aux_img = []
    aux_txt = []
    # get the path/directory
    folder_dir = "/Users/pablosreyero/Documents/Universidad/TFG/Castings/C0001"
    for images in os.listdir(folder_dir):
 
    # check if the image ends with png
        if (images.endswith(".png")):
            #print(images)
            imagenes.append(images)
        if (images.endswith(".txt")):
            #print(images)
            text.append(images)
    #print(imagenes,"\n")
    #print(text)

    initial_dir = "/Users/pablosreyero/Documents/Universidad/TFG/tfg-psr/data/Castings"
    #print("Esto es el directorio actual   " + initial_dir)
    x = []
    for folders in os.listdir(initial_dir):
        x.append(folders)
    #print(x)

    #Ahora tenemos que ordear la lista de las carpetas que hemos extraido
    sorted_folders = funciones.list_sorting(x)
    
    #AHORA iteramos en cada una de los directorios: "C0001,C0002,...C0067"
    #for directorio in sorted_folders[1:]: #el [1:] lo ponemos para no coger el primer archivo
    
        #print(initial_dir + "/" + directorio)
    current_dir = str(initial_dir + "/" + sorted_folders[2])
    print(current_dir)
    os.chdir(current_dir) #Aqui estamos cambiando el directorio en cada iteracion para analizar cada uno de los directorios

        #Aqui chequeamos que nos econtramos en el directorio correcto
    print("El directorio en el que te encuentras es el siguiente :", os.getcwd() + "\n")
    for images in os.listdir(current_dir):
        if (images.endswith(".png")):
            aux_img.append(images)
            sorted_images = funciones.list_sorting(aux_img)

        if (images == "ground_truth.txt"):
            aux_txt.append(images)
            #df = str(os.getcwd() + "/" +str(aux_txt))
            #df = pd.read_csv(os.getcwd() + "/" + str(images), sep="  ",engine="python")
            current_directory = str(os.getcwd())
            #Ahora llamamos a la funcion que nos va a extraer las coordenadas de cada una de las imagenes que tenemos que analizar
            image_data = funciones.read_ground_truth(current_directory,images,sorted_images)
    print("Estos son los datos de lmis imagenes",image_data)

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

