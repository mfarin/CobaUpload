import os
import scipy.misc as scimisc
import matplotlib.pyplot as plt
from preprocess import load_object_data
import sys
import numpy as np
from PIL import Image, ImageFilter


def rgb2gray(image):
    gray = image[:, :, 0] * 0.3 + image[:, :, 1] * 0.2 + image[:, :, 2] * 0.5
    image[:, :, 0] = gray
    image[:, :, 1] = gray
    image[:, :, 2] = gray

    return image


def main():
    # dir = 'datasets/train_bike_car_person'
    #
    # for subdir, dirs, files in os.walk(dir):
    #     print(dirs)
    #     num = 0
    #     for file in files:
    #         # print(os.path.join(subdir, file))
    #         num += 1
    #     print(num)
    #
    dir = 'datasets/train_bike_car_person/bike/bike_015.bmp'
    dir2 = 'datasets/train_bike_car_person/bike/bike_016.bmp'

    ori_image = Image.open(dir).resize((96, 96))

    image = rgb2gray(np.array(ori_image))

    # image3 = ori_image.filter(ImageFilter.GaussianBlur(radius=1))
    image3 = ori_image.crop(box=(0, 0, 48, 96)).resize((96, 96))
    image3 = np.array(image3)
    print(image3.shape)

    image2 = scimisc.imread(dir2, mode='RGB')
    image2 = scimisc.imresize(image2, (96, 96, 3), interp='cubic')

    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.xlabel('Image \n from \n Dataset')
    plt.yticks([])
    plt.xticks([])

    plt.subplot(2, 2, 2)
    plt.imshow(image3)
    plt.xlabel('Image \n from \n Dataset')
    plt.yticks([])
    plt.xticks([])

    plt.subplot(2, 2, 3)
    plt.imshow(image2)
    plt.xlabel('ANOTHER Image \n from \n Dataset')
    plt.yticks([])
    plt.xticks([])

    plt.tight_layout()
    plt.show()

    data_label = [[0, 1, 2, 0], [3, 0, 2, 1], [9, 3, 4, 1]]
    label_list = ['a', 'b', 'c', 'd']

    class_res = {}

    for i in range(len(label_list)):
        class_res[label_list[i]] = data_label[0][i]

    numbers = {'first': 8, 'second': 20, 'third': 2, 'fourth': 10}
    numbers = [(k, numbers[k]) for k in sorted(numbers, key=numbers.get, reverse=True)]

    caption = ''
    for k, v in numbers:
        caption += k + ': %.2f%% \n' % v

    print(caption)

    data, label_list = load_object_data(num_training=2000)

    print(data['x_train'].shape)
    print(data['y_train'].shape)
    print(data['x_val'].shape)
    print(data['y_val'].shape)
    print(label_list)

main()
