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

#Aqui importamos las funciones que hemos creado nosotros
import funciones

#Esto es para evitar problemas de memoria
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#Esta es nuestra funcion principal
def main():
    
    initial_dir = "/Users/pablosreyero/Documents/Universidad/TFG/tfg-psr/data/Castings/C0001"
    x = []
    image_extension = ['png','jpg','jpeg','bmp']

    box_1 = [399, 112, 411, 127]
    box_2 = [186, 82, 221, 113]
    box_3 = [191, 117, 210, 134]
    box = [box_1, box_2, box_3]
    box = torch.tensor(box, dtype=torch.int)

    for iterator in os.listdir(initial_dir): #Para el append de los directorios puedo utilizar la funcion join: os.path.join
        if iterator.endswith(".png"):
            image_path = os.path.join(initial_dir,iterator)
            print(image_path)
            img = read_image(image_path)
            img = draw_bounding_boxes(img, box, width=5, colors=["orange", "blue" , "green"], fill=True)
                                
            # transform this image to PIL image
            img = torchvision.transforms.ToPILImage()(img)
                
            # display output
            img.show()

main()