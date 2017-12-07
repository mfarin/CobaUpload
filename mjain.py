import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from preprocess import load_object_data
import keras
from visualize_and_counting import visualize, count_accuracy


def main():
    data, label_list = load_object_data(num_training=6000)
    for k, v in data.items():
        print('%s: ' % k, v.shape)

    x_train = data['x_train']
    y_train = keras.utils.to_categorical(data['y_train'], num_classes=4)
    x_val = data['x_val']
    y_val = keras.utils.to_categorical(data['y_val'], num_classes=4)

    print('===============================================================================================')

    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(96, 96, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Activation('tanh'))
    model.add(Dense(4, activation='softmax'))

    print('===============================================================================================')
    print('Training')
    print('Using 6-e4 learning rate')

    lr = 6e-4

    adam = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])

    model.fit(x_train, y_train, validation_data=(x_val, y_val), verbose=1, batch_size=50, epochs=50)

    classes = model.predict(x_val, batch_size=20)
    acc = count_accuracy(classes, y_val)

    print('Validation Accuracy = %.3f%%' % (acc*100))

    visualize(x_val[:24], classes[:24], y_val[:24], label_list)

    model.save('cnn_model.h5')

main()
