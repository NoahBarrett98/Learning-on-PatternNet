import matplotlib.pyplot as plt
import time
from IPython import display as ipythondisplay
import numpy as np
import tensorflow as tf
import os
import random
import shutil
class LossHistory:
    """
    Mit Smoothing function for history
    """
    def __init__(self, smoothing_factor=0.0):
        self.alpha = smoothing_factor
        self.loss = []
    def append(self, value):
        self.loss.append( self.alpha*self.loss[-1] + (1-self.alpha)*value if len(self.loss)>0 else value )
    def get(self):
        return self.loss



class PeriodicPlotter:
    """
    Mit plotting function
    """
    def __init__(self, sec, xlabel='', ylabel='', scale=None):

        self.xlabel = xlabel
        self.ylabel = ylabel
        self.sec = sec
        self.scale = scale

        self.tic = time.time()

    def plot(self, data):
        if time.time() - self.tic > self.sec:
            plt.cla()

        if self.scale is None:
            plt.plot(data)
        elif self.scale == 'semilogx':
            plt.semilogx(data)
        elif self.scale == 'semilogy':
            plt.semilogy(data)
        elif self.scale == 'loglog':
            plt.loglog(data)
        else:
            raise ValueError("unrecognized parameter scale {}".format(self.scale))

        plt.xlabel(self.xlabel); plt.ylabel(self.ylabel)
        ipythondisplay.clear_output(wait=True)
        ipythondisplay.display(plt.gcf())


        self.tic = time.time()

def bool_tensor_to_idx(b_tensor, batch_size):
    """
    conversion for bool array to idx
    :param b_tensor:
    :return:
    """
    return tf.constant(np.where(b_tensor.numpy())[1].astype('float32').reshape((batch_size, 1)))

class data_manipulator:
    def __init__(self, base_dir):
        self._base_dir = base_dir

    def train_test_split(self, validation=False, data_split=0.80):
        classes = os.listdir(self._base_dir)
        train_dir = self._base_dir + "/train_data"
        test_dir = self._base_dir + "/test_data"
        os.mkdir(train_dir)
        os.mkdir(test_dir)
        if validation:
            os.mkdir(self._base_dir + "/validation_data")
        for c in classes:
            train_class_dir = train_dir+"/"+c
            test_class_dir = test_dir + "/" + c
            os.mkdir(train_class_dir)
            os.mkdir(test_class_dir)
            if not validation:
                c_path = self._base_dir + "/" + c
                c_dir = os.listdir(c_path)
                #shuffle images so we take random ones each time for split
                random.shuffle(c_dir)
                print(c_dir)
                train = c_dir[0:int(data_split*len(c_dir))]
                test = c_dir[int(data_split*len(c_dir)):]
                #move all train data to train folder
                for train_data in train:
                    shutil.copy(c_path+ "/" + train_data, train_class_dir)
                # move all test data to test folder
                for test_data in test:
                    shutil.copy(c_path+ "/" + test_data, test_class_dir)
            """TODO: add validation"""
        print(classes)

def split_data(basedir, data_split=0.80):
    """
    quicker for calls in py console
    """
    manip = data_manipulator(basedir)
    manip.train_test_split(data_split=data_split)

def show_batch(image_batch, label_batch):
    """
    visualization of batch
    :param image_batch: from ds_train typically
    :param label_batch: from ds_train typically
    :return: None`
    """
    plt.figure(figsize=(10,10))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.axis('off')
    plt.show()