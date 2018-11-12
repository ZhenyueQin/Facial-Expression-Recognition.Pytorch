from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data


class CK(data.Dataset):
    """`CK+ Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``

        there are 135,177,75,207,84,249,54 images in data
        we choose 123,159,66,186,75,225,48 images for training
        we choose 12,18,9,21,9,24,6 images for testing
        the split are in order according to the fold number
    """

    def __init__(self, split='Training', fold = 1, transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        self.fold = fold # the k-fold cross validation
        self.data_face = h5py.File('./data/CK_data_face.h5', 'r', driver='core')
        self.data_saliency = h5py.File('./data/CK_data_saliency.h5', 'r', driver='core')

        if not (len(self.data_face['data_label']) == len(self.data_saliency['data_label'])):
            raise AssertionError()
        number = len(self.data_face['data_label']) #981
        sum_number = [0,135,312,387,594,678,927,981] # the sum of class number
        test_number = [12,18,9,21,9,24,6] # the number of each class

        test_index = []
        train_index = []

        for j in range(len(test_number)):
            for k in range(test_number[j]):
                if self.fold != 10: #the last fold start from the last element
                    test_index.append(sum_number[j]+(self.fold-1)*test_number[j]+k)
                else:
                    test_index.append(sum_number[j+1]-1-k)

        for i in range(number):
            if i not in test_index:
                train_index.append(i)

        print(len(train_index),len(test_index))

        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data_face = []
            self.train_data_saliency = []
            self.train_labels = []
            for ind in range(len(train_index)):
                self.train_data_face.append(self.data_face['data_pixel'][train_index[ind]])
                self.train_data_saliency.append(self.data_saliency['data_pixel'][train_index[ind]])
                self.train_labels.append(self.data_face['data_label'][train_index[ind]])

        elif self.split == 'Testing':
            self.test_data_face = []
            self.test_data_saliency = []
            self.test_labels = []
            for ind in range(len(test_index)):
                self.test_data_face.append(self.data_face['data_pixel'][test_index[ind]])
                self.test_data_saliency.append(self.data_saliency['data_pixel'][test_index[ind]])
                self.test_labels.append(self.data_face['data_label'][test_index[ind]])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img_data, img_saliency, target = self.train_data_face[index], self.train_data_saliency[index], self.train_labels[index]
        elif self.split == 'Testing':
            img_data, img_saliency, target = self.test_data_face[index], self.test_data_saliency, self.test_labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if np.newaxis is not None:
            print('np.newaxis: ', np.newaxis)
        img_data, img_saliency = img_data[:, :, np.newaxis], img_saliency[:, :, np.newaxis]
        img_data, img_saliency = np.concatenate((img_data, img_data, img_data), axis=2), \
                                 np.concatenate((img_saliency, img_saliency, img_saliency), axis=2)
        img_data, img_saliency = Image.fromarray(img_data), Image.fromarray(img_saliency)
        if self.transform is not None:
            img_data = self.transform(img_data)
            img_saliency = self.transform(img_saliency)
        return img_data, img_saliency, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data_face)
        elif self.split == 'Testing':
            return len(self.test_data_face)

