import h5py
import matplotlib.pyplot as plt
import numpy as np

filename = 'data/fer2013_data_top_left.h5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % list(f.keys()))
# a_group_key = list(f.keys())[0]
#
# # Get the data
# data = list(f[a_group_key])
#
# print('data: ', data)

public_test_pixel = f['PrivateTest_pixel']

# print(public_test_pixel[0])

print('shape: ', np.array(public_test_pixel[59]).shape)

plt.imshow(np.reshape(np.array(public_test_pixel[59]), [48, 48]), cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
