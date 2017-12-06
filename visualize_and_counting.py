import numpy as np
import operator
import matplotlib.pyplot as plt


def count_accuracy(classes, y_test):
    right = 0

    for i in range(len(classes)):

        if np.argmax(classes[i]) == np.argmax(y_test[i]):
            right += 1

    return right/len(classes)


def visualize(data, data_label, true_label, label_list):
    for i in range(len(data)):
        plt.subplot(4, 6, i+1)
        plt.imshow(data[i])
        plt.xticks([])
        plt.yticks([])

        class_res = {}

        for j in range(len(label_list)):
            class_res[label_list[j]] = data_label[i, j]

        class_res = [(k, class_res[k]) for k in sorted(class_res, key=class_res.get, reverse=True)]
        if len(class_res) > 3:
            class_res = class_res[:3]

        caption = ''

        for k, v in class_res:
            caption += k + ': %.2f%% \n' % (v * 100)

        true_idx = np.argmax(true_label[i])
        caption += 'True Class : ' + label_list[true_idx]

        plt.xlabel(caption)

    plt.tight_layout()
    plt.show()


def rgb2gray(image):
    gray = image[:, :, 0] * 0.3 + image[:, :, 1] * 0.2 + image[:, :, 2] * 0.5
    image[:, :, 0] = gray
    image[:, :, 1] = gray
    image[:, :, 2] = gray

    return image
