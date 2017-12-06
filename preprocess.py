from __future__ import print_function

from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
from PIL import Image, ImageFilter
import platform
import sys
import random
from visualize_and_counting import rgb2gray
import scipy

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """
    load sebuah batch dari dataset cifar-10
    input:
        alamat file
    return:
        cifar_10 data dan label
    contoh penggunaan:
        data, labels = load_CIFAR_batch( 'data_batch_1' )
    """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        data = datadict['data']
        labels = datadict['labels']
        data = data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        labels = np.array(labels)
        return data, labels


def load_CIFAR10(ROOT):
    """
    load seluruh batch dari cifar dataset di dalam sebuah directory
    inputs:
        alamat folder
    returns:
        cifar_10 batch data dan labels untuk dataset latih dan uji
    conoth penggunaan:
        x_train, y_train, x_test, y_test = load_CIFAR10( 'dataset/cifar-10-batches-py' )
    """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        x_temp, y_temp = load_CIFAR_batch(f)
        xs.append(x_temp)
        ys.append(y_temp)
    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    del x_temp, y_temp
    x_test, y_test = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return x_train, y_train, x_test, y_test


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load CIFAR-10 dataset dari disk dan lakukan preprocessing untuk mempersiapkannya
    untuk klasifikasi. Proses yang dilakukan sama dengan yang dilakukan pada tugas 1,
    namun dipadatkan pada sebuah fungsi
    """
    # Load data CIFAR-10
    cifar10_dir = 'datasets/cifar-10-batches-py'
    x_train, y_train, x_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample data
    mask = range(num_training, num_training + num_validation)
    x_val = x_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    x_train = x_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    x_test = x_test[mask]
    y_test = y_test[mask]

    # Normalisasi data : kurangi dengan rata-rata citra
    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_val -= mean_image
    x_test -= mean_image

    # Transpose data
    x_train = x_train.copy()
    x_val = x_val.copy()
    x_test = x_test.copy()

    # Simpan data ke dalam sebuah dictionary
    return {
        'x_train': x_train, 'y_train': y_train,
        'x_val': x_val, 'y_val': y_val,
        'x_test': x_test, 'y_test': y_test,
    }


def load_models(models_dir):
    """
    Load model yang disimpan di dalam disk. Proses akan membaca (unpickle)
    seluruh file yang ada di dalam sebuah direktori; Semua file yang menyebabkan
    error saat pembacaan (seperti file README.txt) akan dilewati

    inputs:
        - models_dir: String berisi alamat direktori yang menampung file model
        - Setiap file model adalah dictionary dengan field 'model'

    returns:
        dictionary yang memetakan model file names terhadap models.
    """
    models = {}
    for model_file in os.listdir(models_dir):
        with open(os.path.join(models_dir, model_file), 'rb') as f:
            try:
                models[model_file] = pickle.load(f)['model']
            except pickle.UnpicklingError:
                continue
    return models


def load_object_data(num_training=1000):
    data = {}

    x_train, y_train, label_list = load_train(num_training)
    x_val, y_val = load_val()

    data['x_train'] = np.array(x_train)
    data['y_train'] = np.array(y_train)
    data['x_val'] = np.array(x_val)
    data['y_val'] = np.array(y_val)

    return data, label_list


def load_train(num_training=1000):
    dir = 'datasets/train_bike_car_person'

    if num_training > 6000:
        sys.exit("Number Data training must be less than or equal to 6000")

    x_train_all = []
    y_train_all = []
    label_list = []
    label = 0

    for subdir, dirs, files in os.walk(dir):
        if len(dirs) == 0:
            for file in files:
                ori_image = Image.open(os.path.join(subdir, file)).resize((96, 96))

                image = np.array(ori_image)
                image1 = rgb2gray(image)
                image2 = np.array(ori_image.crop(box=(0, 0, 48, 96)).resize((96, 96)))
                image3 = np.array(ori_image.crop(box=(48, 0, 96, 96)).resize((96, 96)))
                image4 = np.array(ori_image.filter(ImageFilter.GaussianBlur(radius=1)))

                x_train_all.extend([image, image1, image2, image3, image4])
                y_train_all.extend([label, label, label, label, label])

            label += 1

        else:
            label_list = dirs

    indices_train = random.sample(range(len(x_train_all)), k=num_training)

    x_train = [x_train_all[i] for i in indices_train]
    y_train = [y_train_all[i] for i in indices_train]

    return x_train, y_train, label_list


def load_val():
    dir = 'datasets/val_bike_car_person'

    x_val_all = []
    y_val_all = []
    label = 0

    for subdir, dirs, files in os.walk(dir):
        if len(dirs) == 0:
            for file in files:
                ori_image = Image.open(os.path.join(subdir, file)).resize((96, 96))
                image = np.array(ori_image)

                x_val_all.append(image)
                y_val_all.append(label)

            label += 1

    indices = random.sample(range(len(x_val_all)), k=len(x_val_all))

    x_val = [x_val_all[i] for i in indices]
    y_val = [y_val_all[i] for i in indices]

    return x_val, y_val


def load_test():
    dir = 'datasets/test_bike_car_person'
    dir_class = 'datasets/target test bike car person.txt'

    x_test = []
    y_test = []

    for subdir, dirs, files in os.walk(dir):
        if len(dirs) == 0:
            for file in files:
                ori_image = Image.open(os.path.join(subdir, file)).resize((96, 96))
                image = np.array(ori_image)

                x_test.append(image)

    fp = open(dir_class)
    labels = [word.strip() for line in fp.readlines() for word in line.split(',') if word.strip()]
    y_test = labels

    return np.array(x_test), np.array(y_test, dtype=int)


def load_label():
    dir = 'datasets/train_bike_car_person'

    for subdir, dirs, files in os.walk(dir):
        if len(dirs) != 0:
            label = dirs

            return label