import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib
import shutil
import random
import functools


AUTOTUNE = tf.data.experimental.AUTOTUNE

class loader(object):
    def __init__(self, base_dir, img_width=256, img_height=256):
        self._base_dir = pathlib.Path(base_dir)
        # toggling custom or all classes selection
        self._class_names = np.array([c.name for c in self._base_dir.glob('*')])
        #self._labels_as_bool = labels_as_bool
        self._image_count = len(list(self._base_dir.glob('*/*.jpg')))
        #batch_size as class feature is clunky
        #self._batch_size = batch_size
        #setting image width and height set here
        self._img_width = img_width
        self._img_height = img_height
        self._ds_size = 0

        # load 2 idx can be used to convert output to label
        self._current_class_load_idx_2_list = {}
        # current data set loaded in loader
        self._current_ds = {}



    def load_data(self, selected_classes=False, label_as_idx=True):
        """
        load data into memory
        :param dirs: if all, use every class in dir, else use selected files
        :return: None (void, loads data into loader object)
        """
        #self._ds = tf.data.Dataset.list_files(str(self._base_dir / '*/*'))
        if isinstance(selected_classes, list):
            to_load = selected_classes
        else:
            to_load = self._class_names
        """
        idx name to conversion
        """
        if label_as_idx:
            self._current_class_load_idx_2_list = {c: float(i) for i, c in enumerate(to_load)}
        else:
            # convert to bool
            self._current_class_load_idx_2_list = {c: c == self._class_names for c in to_load}
        """
        potentially block out to function
        """
        for c in to_load:
            # class files indexed using dict
            # dict: key is class label as index, val is shuffleDataset of list dir
            self._current_ds[self._current_class_load_idx_2_list[c]] = \
                tf.data.Dataset.list_files(str((self._base_dir/c/'*')))
            #print(str((self._base_dir / c / '*/*')))
        """
        convert loaded dirs to image tensors
        """
        self._process()


    def _decode_img(self, img):
        """
        take tf string and convert to img tensor
        :param img:
        :return:
        """
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [self._img_width, self._img_height])

    def _map_img(self, class_idx, img_file_path):
        """
        map image function
        :param class_idx:
        :return:
        """
        # iterate one class at a time
        label = tf.constant(np.array(class_idx).astype('float32').reshape((1)))
        # load file
        img = tf.io.read_file(img_file_path)
        # string tensor is converted to tf.image
        img = self._decode_img(img)
        return img, label



    def _process(self):
        """
        process the files into image, label tensors
        :return:
        """

        # intialize with first class
        keys = iter(self._current_ds.keys())
        init_idx = next(keys)
        this_map_func = functools.partial(self._map_img, class_idx=init_idx)
        _concat_ds = self._current_ds[init_idx].map(lambda x: this_map_func(img_file_path=x),
                                                    num_parallel_calls=AUTOTUNE)
        # iter through rest of keys
        for cur_class_idx in keys:
            this_map_func = functools.partial(self._map_img, class_idx=cur_class_idx)
            #convert from dir, and class, to img and class tensors
            _concat_ds = _concat_ds.concatenate(self._current_ds[cur_class_idx].map(lambda x: this_map_func(img_file_path=x),
                                                                        num_parallel_calls=AUTOTUNE))
        # assign concatonated ds to current_ds
        self._current_ds = _concat_ds
        # get size of ds to set to
        self._ds_size = tf.data.experimental.cardinality(self._current_ds).numpy()

    def get_ds_size(self):
        """
        currently train is entire dataset
        :return: size of training dataset
        """
        "TODO:"
        return self._ds_size

    def reset_load(self):
        """
        resets loaded data set and converter
        :return: None
        """
        ### reset ###
        self._current_class_load_idx_2_list = {}
        self._current_ds = {}

    def get_dataset(self):
        return self._current_ds





class training_data_loader(loader):

    def __init__(self, base_dir, img_width=256, img_height=256):
        super().__init__(base_dir, img_width=256, img_height=256)

    def prepare_for_training(self, batch_size=25, cache=True, shuffle_buffer_size=1000):
        """
        prepares dataset for training makes use of caching to speed up transfer
        cache is used so we only need to load the ds once and after that it will
        refer to cached data rather than reloading multiple instances into mem
        :param cache: bool toggle
        :param shuffle_buffer_size:
        :param batch_size: batch_size for prepared dataset
        :return: returns prepared ds
        """

        if cache:
            if isinstance(cache, str):
                # if we pass a string to cache
                self._current_ds = self._current_ds.cache(cache)
            else:
                # filename not provided, stored in memory
                self._current_ds = self._current_ds.cache()
        # fills a buffer with buffer_size els
        # randomly samples els from this buffer
        self._current_ds = self._current_ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever (indefinately) -> this duplicates the data N times
        self._current_ds = self._current_ds.repeat()

        # Seperates ds into batches of batch size, remainder is not dropped
        # may be a batch which has less els than others
        # N % D els in last batch
        self._current_ds = self._current_ds.batch(batch_size)

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        self._current_ds = iter(self._current_ds.prefetch(buffer_size=AUTOTUNE))

    def get_train_batch(self):
        return next(self._current_ds)

class testing_data_loader(loader):
    def __init__(self, base_dir, img_width=256, img_height=256):
        super().__init__(base_dir, img_width=256, img_height=256)

    def prepare_for_testing(self, cache=True, shuffle_buffer_size=1000):
        """
        prepares dataset for training makes use of caching to speed up transfer
        cache is used so we only need to load the ds once and after that it will
        refer to cached data rather than reloading multiple instances into mem
        :param cache: bool toggle
        :param shuffle_buffer_size:
        :param batch_size: batch_size for prepared dataset
        :return: returns prepared ds
        """
        if cache:
            if isinstance(cache, str):
                # if we pass a string to cache
                self._current_ds = self._current_ds.cache(cache)
            else:
                # filename not provided, stored in memory
                self._current_ds = self._current_ds.cache()
        # fills a buffer with buffer_size els
        # randomly samples els from this buffer
        self._current_ds = self._current_ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever (indefinately) -> this duplicates the data N times
        self._current_ds = self._current_ds.repeat()

        # Seperates ds into batches of batch size, remainder is not dropped
        # may be a batch which has less els than others
        # N % D els in last batch


        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        self._current_ds = self._current_ds.prefetch(buffer_size=AUTOTUNE)


    def get_test_batch(self, batch_size):
        self._current_ds = iter(self._current_ds.batch(batch_size))
        return next(self._current_ds)



#l = training_data_loader(base_dir="C:/Users/Noah Barrett/Desktop/School/Research 2020/data/PatternNet/PatternNet/images")
#print(l._class_names)
#print(l._current_ds)
#l.load_data()
#l.prepare_for_training()
#print(next(l._current_ds))
#test = l._current_ds.take(1)
