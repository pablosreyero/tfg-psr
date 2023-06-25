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
import sys
import cv2


#Here we import the used functions
import newSize_augment_anchors
  
 
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
    y2 = [s.rstrip() for s in y2] #in order to remove the \n command at the end

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
    print(image_title)
    for title in image_title:
        for iter,index in enumerate(ID): #tengo que iterar dentro del diccionario para poder coger tambien las coordenadas al mismo tiempo
            if(int(title[6:10]) == int(float(index))):
                image_path = os.path.join(current_directory,title)
                data_image1 = Image.open(image_path)
                if title in titlesn:
                    final_dic[image_path]['boxes'].append([x1[iter],y1[iter],x2[iter],y2[iter]]) #CAMBIAR TITLE por IMAGE_PATH URGENTEEEEEEEEEE
                else:
                    titlesn.append(title) #dejar la ruta desde castings
                    final_dic[image_path] = {'w': data_image1.width,'h': data_image1.height,'boxes': [[x1[iter],y1[iter],x2[iter],y2[iter]]]} #Añadir un diccionario en title 
                    #Aqui en vez de poner el titulo de cada imagen estamos poniendo la ruta de cada imagen
    print("\n")
    print(final_dic)
    print("\n")
    
    #Ahora pintamos 
    iterador = 0
    for keys, stuff in final_dic.items(): #Para el append de los directorios puedo utilizar la funcion join: os.path.join
        #print(keys,stuff)

        #image_path = os.path.join(current_directory,keys) #Esta linea es muy importante

        #print("\n")
        img = read_image(keys) #Aqui he cmabiado keys por image_path que esta arriba comentado 
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

    train_list = []
    test_list = []
    classes_count1 = {}
    classes_count2 = {}
    class_mapping = {}
    defects_test = 0
    defects_train = 0

    route = "/Users/pablosreyero/Documents/Universidad/TFG/tfg-psr/Ferguson/metadata/gdxray"
    route_to_add = "/Users/pablosreyero/Documents/Universidad/TFG/tfg-psr/data/"

    os.chdir(route)
    for fichiers in os.listdir(route):
        print(fichiers)
        if fichiers == ('castings_test.txt'):
            image_title_test = [x for x in open('castings_test.txt').readlines()] #Aqui estamos recorriendo el archivo MODIFICADO, ANTES; image_title_test = [os.path.basename(x) for x in open('castings_test.txt').readlines()]
            image_title_test = [s.rstrip() for s in image_title_test] #Aqui le estamos quitando el simbolo de salto de linea \n
            print('\n')
            print("This are all TEST images")
            print('\n')
            print(image_title_test)
            print('\n')
        if fichiers == ('castings_train.txt'):
            image_title_train = [x for x in open('castings_train.txt').readlines()] #Aqui estamos recorriendo el archivo, MODIFICADO: image_title_train = [os.path.basename(x) for x in open('castings_train.txt').readlines()]
            image_title_train = [s.rstrip() for s in image_title_train] #Aqui le estamos quitando el simbolo de salto de linea \n
            print('\n')
            print("This are all TRAIN images")
            print('\n')
            print(image_title_train)
            print('\n')
    
    #Now that we have the titles of both train and test images that will be implemented later on, we proceed by giving the user information about these images
    print("Now information of each TEST image will be printed")
    print('\n')

    name_list1 = []
    for iter in final_dic.keys():
        name_list1.append(iter[59:])
    print(name_list1)

    for i in image_title_test: #Como ahora ya no tengo solo los titulos de las imagenes si no que no tengo también las rutas completas de las imágenes, tengo que cambiar esta parte también
        if i in name_list1:
            i_prime = os.path.join(route_to_add,i)
            test_string = str(i) + " -> " + str(final_dic[i_prime]) #MODIFICADO, ANTES: test_string = str(i) + " -> " + str(final_dic[i])
            defects_test_aux = len(final_dic[i_prime]['boxes']) #i es un string
            defects_test += defects_test_aux
            test_list.append(test_string)
    
    classes_count1['defects'] = defects_test
    
    print("\n")
    print("Now information of each TRAIN image will be printed")
    print("\n")

    name_list2 = []
    for iter in final_dic.keys():
        name_list2.append(iter[59:])
    
    
    for j in image_title_train:
        if j in name_list2: #MODIFICADO, ANTES: if j in final_dict
            j_prime = os.path.join(route_to_add,j)
            train_string = [str(j),(final_dic[j_prime])] #MODIFICADO, ANTES: test_string = str(j) + " -> " + str(final_dic[j])
            #print(train_string)
            defects_train_aux = len(final_dic[j_prime]['boxes'])
            defects_train += defects_train_aux
            train_list.append(train_string)

    classes_count2['defects'] = defects_train
    
    return test_list, train_list, classes_count1, classes_count2, class_mapping

def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length//16

    return get_output_length(width), get_output_length(height)  


def main(C):
    aux_txt = []
    # get the path/directory
    initial_dir = "/Users/pablosreyero/Documents/Universidad/TFG/tfg-psr/data/Castings"
    x = []
    
    for folders in os.listdir(initial_dir):
        x.append(folders)

    #Now we sort the extracted folders
    sorted_folders = list_sorting(x)

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

                    image_data = read_ground_truth(current_directory,images,sorted(aux_img))
                    final_dic = boundingBox(current_directory,image_data)
                    merged_dictionary = merged_dictionary | final_dic #We merge the dictionary each iteration

        else:
            print("El directorio: " + str(current_dir) + " NO contiene imágenes con defectos")
            print("\n")
            
    print("\n")
    print("Este es el diccionario final: ", merged_dictionary)
    print("\n")
    #Now that we have our dictionary with all items we search for train and test images only
    test_list, train_list, classes_count1, classes_count2, class_mapping = reading_train_test(merged_dictionary)
    print(test_list)
    print("\n")
    print(train_list)
    print("\n")
    print("Número de defectos en castings_test.txt", classes_count1)
    print("Número de defectos en castings_train.txt", classes_count2)
    print("EL PROGRAMA HA TERMINADO")

    #Now that we have all of our data extracted from .txts and images, we proceed by augmenting existing data since we are assuming an overfitting
    all_img_data = train_list
    print("Estos son los datos que nos interesan")
    print("\n")
    print(all_img_data)
    print(all_img_data[0][1]['w'])
    img22 = cv2.imread(img_data_aug['filepath'])

    #Now we create all anchors
    newSize_augment_anchors.get_anchor_gt(all_img_data, C, get_img_output_length, mode='train')



            
    



    
    