from preprocess import load_test, load_label
import keras
from keras.models import load_model
from visualize_and_counting import count_accuracy


def main():
    x_test, y_test = load_test()
    label_list = load_label()

    y_test = keras.utils.to_categorical(y_test, num_classes=4)

    cnn_model = load_model('cnn_model.h5')

    print('===============================================================================================')
    print('Testing')

    classes = cnn_model.predict(x_test, verbose=1)
    print(classes)

    acc = count_accuracy(classes, y_test)
    print('\nTesting Accuracy = %.3f%%' % (acc*100))


main()
