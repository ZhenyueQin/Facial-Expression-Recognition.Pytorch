import csv
import os
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt

def read_fer_data(file='data/fer2013.csv'):
    # Creat the list to store the data and label information
    Training_x = []
    Training_y = []
    PublicTest_x = []
    PublicTest_y = []
    PrivateTest_x = []
    PrivateTest_y = []

    datapath = os.path.join('data','data.h5')
    if not os.path.exists(os.path.dirname(datapath)):
        os.makedirs(os.path.dirname(datapath))

    upbound = 200000
    current_count = 0
    with open(file,'r') as csvin:
        data=csv.reader(csvin)
        for row in data:
            if row[-1] == 'Training':
                temp_list = []
                for pixel in row[1].split( ):
                    temp_list.append(int(pixel))
                I = np.asarray(temp_list)
                Training_y.append(int(row[0]))
                Training_x.append(I.tolist())

            if row[-1] == "PublicTest" :
                temp_list = []
                for pixel in row[1].split( ):
                    temp_list.append(int(pixel))
                I = np.asarray(temp_list)
                PublicTest_y.append(int(row[0]))
                PublicTest_x.append(I.tolist())

            if row[-1] == 'PrivateTest':
                temp_list = []
                for pixel in row[1].split( ):
                    temp_list.append(int(pixel))
                I = np.asarray(temp_list)

                PrivateTest_y.append(int(row[0]))
                PrivateTest_x.append(I.tolist())
            if current_count > upbound:
                break
            current_count += 1

    # print(np.shape(Training_x))
    # print(np.shape(PublicTest_x))
    # print(np.shape(PrivateTest_x))

    return Training_x, Training_y, PublicTest_x, PublicTest_y, PrivateTest_x, PrivateTest_y
