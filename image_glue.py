import os
import cv2
import numpy as np


def combine_images(imgs, save_path):
    from PIL import Image
    import matplotlib.pyplot as plt

    images = list(map(Image.open, imgs))
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]

    new_im = np.array(new_im)

    # new_im = new_im.resize([196, 48], Image.ANTIALIAS)
    new_im = cv2.cvtColor(new_im, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(save_path, new_im)
    # new_im.save(save_path)


dir_name_1 = 'FER2013_prediction'
dir_name_2 = 'FER2013'
dir_name_3 = 'FER2013_combined_overlay'

saliency_or_overlap = 'overlap_'

for root, dirs, files in os.walk(dir_name_1):
    for file in files:
        if file.endswith(".png"):
            if saliency_or_overlap in file:
                a_file_name_1 = os.path.join(root, file)
                print('file name 1: ', a_file_name_1)
                a_file_name_2 = a_file_name_1.replace(dir_name_1, dir_name_2).replace(saliency_or_overlap, '')

                datapath = a_file_name_1.replace(saliency_or_overlap, '').replace(dir_name_1, dir_name_3)
                if not os.path.exists(os.path.dirname(datapath)):
                    os.makedirs(os.path.dirname(datapath))

                combine_images([a_file_name_1, a_file_name_2], datapath)
