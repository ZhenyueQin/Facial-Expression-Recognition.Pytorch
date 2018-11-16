import os
import cv2
import numpy as np
from qin_test_field import read_fer_data
import h5py

datapath = os.path.join('data','fer2013_combined.h5')

_, Training_y, _, PublicTest_y, _, PrivateTest_y = read_fer_data()


overlap_or_saliency = ''

def get_a_x_list(target_path):
    rtn_list = []
    file_names = [f for f in os.listdir(target_path) if f.endswith(('.jpg', '.jpeg', '.png')) and f.startswith(overlap_or_saliency)]
    numbers = sorted([int(f.replace(overlap_or_saliency, '').replace('.png', '')) for f in file_names])

    for a_number in numbers:
        a_name = target_path + overlap_or_saliency + str(a_number) + '.png'
        a_img_2d = cv2.imread(a_name, 0)
        a_img_list = np.reshape(a_img_2d, [48 * 96,]).tolist()
        rtn_list.append(a_img_list)
    return rtn_list

Training_x = get_a_x_list('FER2013_combined_overlay/training_x/')
PublicTest_x = get_a_x_list('FER2013_combined_overlay/public_x/')
PrivateTest_x = get_a_x_list('FER2013_combined_overlay/private_x/')

datafile = h5py.File(datapath, 'w')
datafile.create_dataset("Training_pixel", dtype = 'uint8', data=Training_x)
datafile.create_dataset("Training_label", dtype = 'int64', data=Training_y)
datafile.create_dataset("PublicTest_pixel", dtype = 'uint8', data=PublicTest_x)
datafile.create_dataset("PublicTest_label", dtype = 'int64', data=PublicTest_y)
datafile.create_dataset("PrivateTest_pixel", dtype = 'uint8', data=PrivateTest_x)
datafile.create_dataset("PrivateTest_label", dtype = 'int64', data=PrivateTest_y)
datafile.close()

print("Save data finish!!!")
