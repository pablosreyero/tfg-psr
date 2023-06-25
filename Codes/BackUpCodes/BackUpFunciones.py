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
from copy import deepcopy
import json

def list_sorting(item):
    
    n = len(item)
    for i in range(n):

    # Traverse the list from 0 to n-i-1
    # (The last element will already be in place after first pass, so no need to re-check)
        for j in range(0, n-i-1):

             # Swap if current element is greater than next
            if item[j] > item[j+1]:
                item[j], item[j+1] = item[j+1],item[j]
            
    return item

def read_ground_truth(current_directory,ground_truth_name_file,sorted_images):

    image_ID= [x.split('   ')[1] for x in open('ground_truth.txt').readlines()]
    x1 = [x.split('   ')[2] for x in open('ground_truth.txt').readlines()]
    x2 = [x.split('   ')[3] for x in open('ground_truth.txt').readlines()]
    y1 = [x.split('   ')[4] for x in open('ground_truth.txt').readlines()]
    y2 = [x.split('   ')[5] for x in open('ground_truth.txt').readlines()]

    image_data = {
        "Titulos" : sorted_images,
        "ID": image_ID,
        "x1": x1,
        "x2": x2,
        "y1": y1,
        "y2": y2
    }
    return image_data

def boundingBox(current_directory,image_data):

    image_title = image_data["Titulos"]
    ID = image_data["ID"]
    x1 = image_data["x1"]
    x2 = image_data["x2"]
    y1 = image_data["y1"]
    y2 = image_data["y2"]
    y2 = [s.rstrip() for s in y2]
    print(y2)

    #CONVERTING FROM FLOAT TO INTEGER
    x1 = [int(float(i)) for i in x1]
    x2 = [int(float(i)) for i in x2]
    y1 = [int(float(i)) for i in y1]
    y2 = [int(float(i)) for i in y2]

    print(x1,x2,y1,y2)
    print("\n")
    print("Este es el directorio en el que tengo que trabajar; \t", current_directory)
    print("\n")
    print("Ahora probamos la implementación que queríamos poner bien")
    
    titlesn = []
    final_dic = {}
    dict_per_image = {}
    for title in image_title:
        for iter,index in enumerate(ID): #tengo que iterar dentro del diccionario para poder coger tambien las coordenadas al mismo tiempo
            if(int(title[6:10]) == int(float(index))):
                image_path = os.path.join(current_directory,title)
                data_image1 = Image.open(image_path)
                if title in titlesn:
                    final_dic[image_path]['boxes'].append([x1[iter],y1[iter],x2[iter],y2[iter]])
                else:
                    titlesn.append(image_path) #dejar la ruta desde castings
                    final_dic[image_path] = {'w': data_image1.width,'h': data_image1.height,'boxes': [[x1[iter],y1[iter],x2[iter],y2[iter]]]} #Añadir un diccionario en title 
                    #Aqui en vez de poner el titulo de cada imagen estamos poniendo la ruta de cada imagen
    print("\n")
    print(final_dic)
    print("\n")
    
    #Ahora pintamos 
    iterador = 0
    for keys, stuff in final_dic.items(): #Para el append de los directorios puedo utilizar la funcion join: os.path.join
        #print(keys,stuff)
        image_path = os.path.join(current_directory,keys)
        #print("\n")
        img = read_image(image_path)
        box = []
        for j in stuff['boxes']:
            for k in range(len(final_dic[keys])):
                if j not in box:
                    box.append(j)
        #print(box)
        
        box = torch.tensor(box, dtype=torch.int)
        img = draw_bounding_boxes(img, box, width=1, colors="red", fill=True)
                            
        # transform this image to PIL image
        img = torchvision.transforms.ToPILImage()(img)
    
        # display output

        #img.show()

        #img.save('/Users/pablosreyero/Documents/Universidad/TFG/tfg-psr/Processed IMages', 'PNG')
    return final_dic
        #Lo que tengo que hacer es que el codigo lea el archivo .txt  del enlace que me mandó Maria José y segun vaya leyendo las imagenes que ya me dicen, el codigo tiene que saber de que imagen se trata y por ende hacer un dssplay de la información de dicha imagen

def reading_train_test (final_dic):

    route = "/Users/pablosreyero/Documents/Universidad/TFG/tfg-psr/Ferguson/metadata/gdxray"
    os.chdir(route)
    for fichiers in os.listdir(route):
        print(fichiers)
        if fichiers == ('castings_test.txt'):
            image_title_test = [os.path.basename(x) for x in open('castings_test.txt').readlines()] #Aqui estamos recorriendo el archivo
            image_title_test = [s.rstrip() for s in image_title_test] #Aqui le estamos quitando el simbolo de salto de linea \n
            print('\n')
            print("This are all TEST images")
            print('\n')
            print(image_title_test)
            print('\n')
        if fichiers == ('castings_train.txt'):
            image_title_train = [os.path.basename(x) for x in open('castings_train.txt').readlines()] #Aqui estamos recorriendo el archivo
            image_title_train = [s.rstrip() for s in image_title_train] #Aqui le estamos quitando el simbolo de salto de linea \n
            print('\n')
            print("This are all TRAIN images")
            print('\n')
            print(image_title_train)
            print('\n')
    
    #Now that we have the titles of both train and test images that will be implemented later on, we proceed by giving the user information about these images
    print("Now information of each TEST image will be printed")
    print('\n')
    for i in image_title_test:
        if i in final_dic:
            print("Information about" + str(i) + " --> " + str(final_dic[i]))
    
    print("\n")
    print("Now information of each TEST image will be printed")
    print("\n")
    for j in image_title_train:
        if j in final_dic:
            print("Information about" + str(i) + " --> " + str(final_dic[i]))

    
            
    



    
    