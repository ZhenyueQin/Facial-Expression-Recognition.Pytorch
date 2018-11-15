import cv2
import os


def square_overlay_images(path_1, path_2, save_path):
    s_img = cv2.imread(path_2, 0)
    l_img = cv2.imread(path_1, 0)
    s_height, s_width = s_img.shape
    l_height, l_width = l_img.shape

    ss_img = cv2.resize(s_img, (int(l_height/4), int(l_width/4)))
    print('l img: ', ss_img.shape)
    l_img[0:0+ss_img.shape[0], 0:0+ss_img.shape[1]] = ss_img

    cv2.imwrite(save_path, l_img)


dir_name_1 = 'ck_prediction'
dir_name_2 = 'CK+48'
dir_name_3 = 'CK+squared_top_left'


for root, dirs, files in os.walk("ck_prediction"):
    for file in files:
        if file.endswith(".png"):
            if 'saliency_' in file:
                a_file_name_1 = os.path.join(root, file)
                print('file name 1: ', a_file_name_1)
                a_file_name_2 = a_file_name_1.replace(dir_name_1, dir_name_2).replace('saliency_', '')
                print('file name 2: ', a_file_name_2)

                datapath = a_file_name_1.replace('saliency_', '').replace(dir_name_1, dir_name_3)
                if not os.path.exists(os.path.dirname(datapath)):
                    os.makedirs(os.path.dirname(datapath))

                square_overlay_images(a_file_name_2, a_file_name_1, datapath)





