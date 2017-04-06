from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from feature import process2
from lib import imgio as ImgIO
from keras.models import load_model
from lib import dataset
from segment.color import *
import h5py
import numpy as np



def cnn_keras_mnist(name, batch_size=32, num_classes=10, epochs=12, img_rows=28, img_cols=28):
    # batch_size = 32
    # num_classes = 9
    # epochs = 12
    # img_rows = 28
    # img_cols = 28

    name = 'gdgs'
    data_file = h5py.File(name+'.h5', 'r')

    X_train = data_file['X_train'][:]
    X_test = data_file['X_test'][:]
    Y_train = data_file['y_train'][:]
    Y_test = data_file['y_test'][:]
    data_file.close()


    label_map = {}
    for label, index in zip(np.unique(Y_train),range(num_classes)):
        label_map[label] = index
    key = {}.fromkeys(label_map.keys(), 0)
    for label in Y_train:
        key[label] += 1
    for label, value in key.items():
        print("Label: %s, num: %d" % (label, value))


    # X_train = np.array(map(lambda img: process2.inverse(process2.resize_transform(img, output_shape=(img_rows, img_cols))), X_train))
    # X_test = np.array(map(lambda img: process2.inverse(process2.resize_transform(img, output_shape=(img_rows, img_cols))), X_test))
    X_train = np.array([process2.inverse(process2.resize_transform(img, output_shape=(img_rows, img_cols))) for img in X_train])
    X_test =  np.array([process2.inverse(process2.resize_transform(img, output_shape=(img_rows, img_cols))) for img in X_test])
    # print(X_train.max())
    y_train = np.array([label_map[v] for v in Y_train])
    y_test = np.array([label_map[v] for v in Y_test])
    # y_train = np.array(y_train)
    # y_test = np.array(y_test)
    # print(np.unique(y_test))
    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = X_train.astype('float32')
    x_test = X_test.astype('float32')
    # x_train /= 255
    # x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # print(model.predict_classes(x_test[:10]))
    # print(Y_test[:10])
    # for img in x_test[:10]:
    #     i = img.reshape(img_rows, img_cols)
    #     ImgIO.show_image(i)
    #     im = img.reshape(1, img_rows, img_cols, 1)
    #     print(model.predict_classes(im))
    # model.save('cnn_test.h5')
    # print(model.predict_classes(x=x_test, batch_size=1))
    images, labels = dataset.load_captcha_pkl('result/processing/gdgs.pkl')#dataset.pkl')
    for image, label in zip(images, labels):
        imgs = ColorFillingSeparator(image).get_characters()
        ImgIO.show_images_list(imgs)
        predict_image = np.array([process2.inverse(process2.resize_transform(img, output_shape=(img_rows, img_cols))) for img in imgs])
        print(predict_image.max())
        predict_image.astype('float32')
        if imgs[0].max() > 1:
            predict_image /= 255.

        if K.image_data_format() == 'channels_first':
            predict_image = predict_image.reshape(predict_image.shape[0], 1, img_rows, img_cols)
        else:
            predict_image = predict_image.reshape(predict_image.shape[0], img_rows, img_cols, 1)
        predict_result = model.predict_classes(predict_image, batch_size=1)
        print(predict_result)
        result = [label_map.keys()[label_map.values().index(i)] for i in predict_result]
        print("Label: %s \t Predict: %s" % (label, ''.join(result)))


def cnn_leNet5(name, batch_size=16, num_classes=10, epochs=12, img_rows=28, img_cols=28):
    #name = 'ndataset'
    data_file = h5py.File(name+'.h5', 'r')

    X_train = data_file['X_train'][:]
    X_test = data_file['X_test'][:]
    Y_train = data_file['y_train'][:]
    Y_test = data_file['y_test'][:]
    data_file.close()

    label_map = {}
    for label, index in zip(np.unique(Y_train),range(num_classes)):
        label_map[label] = index

    X_train = np.array([process2.inverse(process2.resize_transform(img, output_shape=(img_rows, img_cols))) for img in X_train])
    X_test =  np.array([process2.inverse(process2.resize_transform(img, output_shape=(img_rows, img_cols))) for img in X_test])
    y_train = np.array([label_map[v] for v in Y_train])
    y_test = np.array([label_map[v] for v in Y_test])

    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = X_train.astype('float32')
    x_test = X_test.astype('float32')
    # x_train /= 255
    # x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print(input_shape)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    model = Sequential()
    model.add(Conv2D(25, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print("Baseline Error: %.2f%%" % (100-score[1]*100))

    print(model.predict_classes(x_test[:10]))
    print(''.join(Y_test[:10]))
    images, labels = dataset.load_captcha_pkl('result/processing/gdgs.pkl')#dataset.pkl')
    for image, label in zip(images[:10], labels[:10]):
        imgs = ColorFillingSeparator(image).get_characters()
        ImgIO.show_images_list(imgs)
        predict_image = np.array([process2.inverse(process2.resize_transform(img, output_shape=(img_rows, img_cols))) for img in imgs])
        print(predict_image.max())
        predict_image.astype('float32')
        if imgs[0].max() > 1:
            predict_image /= 255.

        if K.image_data_format() == 'channels_first':
            predict_image = predict_image.reshape(predict_image.shape[0], 1, img_rows, img_cols)
        else:
            predict_image = predict_image.reshape(predict_image.shape[0], img_rows, img_cols, 1)
        predict_result = model.predict_classes(predict_image, batch_size=1)
        print(predict_result)
        result = [label_map.keys()[label_map.values().index(i)] for i in predict_result]
        print("Label: %s \t Predict: %s" % (label, ''.join(result)))


# cnn_keras_mnist('gdgs')
cnn_leNet5('gdgs')