from qin_test_field import read_fer_data
import cv2
import numpy as np

training_x_output_folder = 'predictions/training_x/'
public_x_output_folder = 'predictions/public_x/'
private_x_output_folder = 'predictions/private_x/'

Training_x, Training_y, PublicTest_x, PublicTest_y, PrivateTest_x, PrivateTest_y = read_fer_data()

for i in range(len(Training_x)):
    a_training_x = np.reshape(Training_x[i], [48, 48])
    cv2.imwrite(training_x_output_folder + str(i) + '.png', a_training_x.astype(int))

for i in range(len(PublicTest_x)):
    a_training_x = np.reshape(PublicTest_x[i], [48, 48])
    cv2.imwrite(public_x_output_folder + str(i) + '.png', a_training_x.astype(int))

for i in range(len(PrivateTest_x)):
    a_training_x = np.reshape(PrivateTest_x[i], [48, 48])
    cv2.imwrite(private_x_output_folder + str(i) + '.png', a_training_x.astype(int))

