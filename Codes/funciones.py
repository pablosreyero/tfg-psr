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

    #print(ID[0],x1[0],x2[0],y1[0],y2[0]) #Me he quedado aqui, ya tengo los arrays que queria, ahora me falta comprobar cuantos errores en cada imagen, para dibujar correctamente los boundingboxes
    #print(image_title[1][0])

   #Esto ya nos devuelve la lista de los titulos de las imagenes ordenada
    #print("Esta es mi lista de imagenes: \t", sorted_list) 
    #Ahora tenemos que intentar printear todas las imagenes en un bucle
    
    print("Este es el directorio en el que tengo que trabajar; \t", current_directory)

    box_1 = [399, 112, 411, 127]
    box_2 = [186, 82, 221, 113]
    box_3 = [191, 117, 210, 134]
    box = [box_1, box_2, box_3]
    box = torch.tensor(box, dtype=torch.int)

    for iterator in sorted(os.listdir(current_directory)): #Para el append de los directorios puedo utilizar la funcion join: os.path.join
        if iterator.endswith(".png"):
            image_path = os.path.join(current_directory,iterator)

    
    duplicates = defaultdict(list)
    res = [idx for idx, item in enumerate(ID) if item in ID[:idx]]
    #print(res) #res me devuelve los indices de los elementos repetidos en la lista de los ID's, y con esos indices tendré que introducir en la otra lista en esa posicion que indica el indice 
    
    for i, number in enumerate(ID):
        duplicates[number].append(i)
    result = {key: value for key, value in duplicates.items() if len(value) > 1}
    print(result)
    print("\n")

    duplicated_indices = []
    for i in result.items():
        duplicated_indices.append(i[1])  #esto me devuelve los indices de los numeros duplicados (tambien me devuelve le primero)
    print(duplicated_indices)
    print("\n")

    step = 0
    image_title_new = image_title
    for i in duplicated_indices:
        for pos,title in enumerate(image_title):
            if pos == i[0]:
                insert_at = pos + step
                image_title_new.pop(insert_at) #ME HE QUEDADO AQUI, PROBAR A ELIMINAR EL ELEMENTO DE ESA POSICION
                image_title_new[insert_at:insert_at] = [title] * (len(i))
                #image_title = image_title_new
                #step = step + len(i)
    """
    for i in res:
        for pos,title in enumerate(image_title):                
            if (pos == i):
                insert_at = pos
                image_title[insert_at:insert_at] = [title]  # Insert "3" within "b"
                #El siguiente AVISO ES MUY IMPORTANTE
                #Como estamos recorriendo las dos listas a la vez, detecto si un elemento esta repetido en una y si lo esta, duplico en esa misma posición el elemento de la otra lista 
                #Además para poder conocer las posiciones de los elementos duplicados tengo que usar en el bucle --enumerate(lista)--
                #Para poder iterar en las posiciones de la lista y los elementos de la lista tengo que poner --- for i,j in enumerate(lista): ---
    """
    print(image_title_new)
    """
            img = read_image(image_path)
            img = draw_bounding_boxes(img, box, width=5, colors=["orange", "blue" , "green"], fill=True)
                                
            # transform this image to PIL image
            img = torchvision.transforms.ToPILImage()(img)
                
            # display output
            img.show()
            """



    """
        imga = mpimg. imread(i)
        img = plt.imshow(imga)
        plt.show()
        
        img = read_image(sorted_list[i])
        img.show()
        """

    
    """"
    # create boxes
    box_1 = [399, 112, 411, 127]
    box_2 = [186, 82, 221, 113]
    box_3 = [191, 117, 210, 134]
    box = [box_1, box_2, box_3]

    box = torch.tensor(box, dtype=torch.int)
    
    #for i in sorted_list:
    #img = read_image('C0001_0020.png')
    img = read_image(sorted_list[0]) #esto me devuelve lo mismo que la linea anterior
    #print(img)
    # draw bounding box and fill color
    img = draw_bounding_boxes(img, box, width=5, colors=["orange", "blue" , "green"], fill=True)
                                
    # transform this image to PIL image
    img = torchvision.transforms.ToPILImage()(img)
        
    # display output
    #img.show()
    #plt.imshow(img)
    """
