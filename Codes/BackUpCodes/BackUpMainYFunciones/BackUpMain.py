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
    aux_txt = []
    # get the path/directory
    initial_dir = "/Users/pablosreyero/Documents/Universidad/TFG/tfg-psr/data/Castings"
    x = []
    
    for folders in os.listdir(initial_dir):
        x.append(folders)

    #Now we sort the extracted folders
    sorted_folders = funciones.list_sorting(x)

    #Then, we iterate trough all items inside sorted_folders in order to keep folders that only host images
    for j,item in enumerate(sorted_folders):
        if item[0] != "C":
            sorted_folders.pop(j) 
    print(sorted_folders)

    merged_dictionary = {}
    for iter, chose in enumerate(sorted_folders):
        aux_img = []
        current_dir = str(initial_dir + "/" + sorted_folders[iter])
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
                    print("\n")
                    print("Este es el aux_img:" , sorted(aux_img))

                    image_data = funciones.read_ground_truth(current_directory,images,sorted(aux_img))
                    final_dic = funciones.boundingBox(current_directory,image_data)
                    merged_dictionary = merged_dictionary | final_dic #We merge the dictionary each iteration

        else:
            print("El directorio: " + str(current_dir) + " NO contiene imágenes con defectos")
            print("\n")
            
    print("\n")
    print("Este es el diccionario final: ", merged_dictionary)
    print("\n")
    #Now that we have our dictionary with all items we search for train and test images only
    funciones.reading_train_test(merged_dictionary)
    print("EL PROGRAMA HA TERMINADO")

main()