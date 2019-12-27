from mask_functions import *
from collections import defaultdict

import tensorflow as tf
import numpy as np
import pandas as pd
import pydicom as dicom
import glob
import os


class SIIM:

    # Private Functions

    def __init__(self, data_path):
        self.data_path = data_path
        self.df_rle = pd.read_csv(data_path + 'train-rle.csv')
        self.__labels = {}
        self.__masks = {}
        self.__pixel_arrays = []

    def __fetch_image_label(self, dcm_image_file_path, img_count):
        image_id = self.df_rle.loc[img_count, 'ImageId']
        enc_pixel = self.df_rle.loc[img_count, ' EncodedPixels']

        current_file_path = glob.glob(dcm_image_file_path + image_id + '.dcm')[0]
        if (os.path.exists(current_file_path)):
            self.__pixel_arrays.append(dicom.dcmread(current_file_path).pixel_array)

            if (enc_pixel.strip() != "-1"):
                image_rle_mask = rle2mask(enc_pixel, 1024, 1024)
                self.__masks[image_id] = image_rle_mask
                self.__labels[image_id] = 1
            else:
                self.__labels[image_id] = 0

    def __get_label_mask_data(self, extension_string):
        dcm_image_file_path = self.data_path + extension_string + '/*/*/'
        for img_count in range(len(self.df_rle)):
            print("Count: " + str(img_count))
            self.__fetch_image_label(dcm_image_file_path, img_count)

            # TODO: Remove this comment and next condition block in the actual experiment in Jupyter Notebook
            if (img_count > 9):
                break



    def __fetch_train_data(self):
        train_data_extension_string = 'dicom-images-train'

        self.__get_label_mask_data(train_data_extension_string)



    def __fetch_test_data(self):
        test_data_extension_string = 'dicom-images-test'



    def __save_labels(self):
        pass

    # Public Functions


    def start(self, train_data=True, test_data=False):
        """
        Default of this function fetches train data only. (Set test_data = True to fetch test data as well)
        This function starts the data fetch and setting the labels as well as the mask
        for each image.
        :return: None
        """
        if (train_data):
            self.__fetch_train_data()
        if (test_data):
            self.__fetch_test_data()



class SIIMDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDS, labels, batch_size=64, dim=(1024, 1024), n_chanels=1, shuffle=True):
        'Initialization'
        self.list_IDs = list_IDS
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_chanels = n_chanels
        self.shuffle = shuffle
        self.on_epoch_end()


    def __data_generation(self, list_IDs_temp):
        'Generate data containing batch_size samples' # X: (n_samples, *dim, n_chanels)
        X = np.empty((self.batch_size, *self.dim, self.n_chanels))
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            # Store a sample
            X[i,] = np.expand_dims(dicom.read_file(ID).pixel_array, axis=2)

            image_ID = ID.split('/')[-1][:-4]
            rle = self.labels[image_ID]

            if (rle is None):
                y[i,] = np.zeros((1024, 1024, 1))
            else:
                if (len(rle) == 1):
                    y[i,] = np.expand_dims(rle2mask(rle[0], self.dim[0], self.dim[1]).T,
                                           axis=2)
                else:
                    y[i,] = np.zeros((1024, 1024, 1))
                    for x in rle:
                        y[i,] = y[i,] + np.expand_dims(rle2mask(x, 1024, 1024).T, axis=2)

        return X, y

    # keras.utils

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y



    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

        if (self.shuffle):
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))




if __name__ == '__main__':
    # Data Paths
    siim_train_data = glob.glob('Data/SIIM/dicom-images-train/*/*/*.dcm')
    siim_test_data = glob.glob('Data/SIIM/dicom-images-test/*/*/*.dcm')
    rle_df = pd.read_csv('Data/SIIM/train-rle.csv')

    temp_rle_dict = defaultdict(list)

    for image_ID, rle in zip(rle_df['ImageId'], rle_df[' EncodedPixels']):
        temp_rle_dict[image_ID].append(rle)

    rle_df = temp_rle_dict

    # Fetch Labels
    labels = {key: val for key, val in rle_df.items() if val[0] != ' -1'}


    params = {'labels': labels,
              'batch_size': 32}

    train_generator = SIIMDataGenerator(siim_train_data[0:9000], **params)
    val_generator   = SIIMDataGenerator(siim_train_data[9000:], **params)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(epsilon=0.1)
    metrics = ['acc']

