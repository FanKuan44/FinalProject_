import pickle as pk

import keras
import kerop
import numpy as np
import tensorflow as tf
from keras_preprocessing.image.image_data_generator import ImageDataGenerator
from keras.optimizers import SGD


def remove_values_from_list_x(x, val):
    return [value for value in x if value != val]


def insert_to_list_x(x):
    added = ['|', '|', '|', '|']
    indices = [4, 7, 10, 14]

    acc = 0
    for i in range(len(added)):
        x.insert(indices[i] + acc, added[i])
        acc += 1
    return x


def convert_X_to_hashX(x):
    if not isinstance(x, list):
        x = x.tolist()
    x = insert_to_list_x(x)
    x = remove_values_from_list_x(x, 'I')
    hashX = ''.join(x)
    return hashX


def create_and_evaluate_model(x):
    """
    :param x: hashX
    :return:
    """
    keras.backend.clear_session()
    model = keras.Sequential()
    TF_CONFIG_ = tf.compat.v1.ConfigProto()
    TF_CONFIG_.gpu_options.per_process_gpu_memory_fraction = 0.9
    TF_CONFIG_.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=TF_CONFIG_)
    keras.backend.set_session(sess)

    model.add(keras.layers.InputLayer(input_shape=(32, 32, 3)))
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.BatchNormalization())
    filters = 32
    for layer in x:
        if layer == '1':
            model.add(keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu'))
        if layer == '2':
            model.add(keras.layers.Conv2D(filters, (5, 5), padding='same', activation='relu'))
        if layer == '|':
            model.add(keras.layers.MaxPooling2D((2, 2)))
            model.add(keras.layers.Dropout(0.3))
            filters *= 2
        else:
            model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(filters, activation='relu'))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.summary()

    # early_stopping = keras.callbacks.EarlyStopping(patience=8)
    # optimizer = keras.optimizers.Adam(lr=1e-3, amsgrad=True)

    # model.compile(optimizer=optimizer,
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    #
    # # Fit model in order to make predictions
    # history = model.fit_generator(generator=data.flow(X_train, y_train, batch_size=128),
    #                               epochs=1,
    #                               validation_data=(X_val, y_val),
    #                               callbacks=[early_stopping])
    #
    _, layer_flops, _, _ = kerop.profile(model)
    #
    # val_acc = history.history['val_accuracy']
    FLOPs = sum(layer_flops) / 1e6  # --> or test_acc
    print(x, FLOPs)
    # return FLOPs, val_acc


def _sampling(n_samples):
    pop_X, pop_hashX = [], []

    allowed_choices = ['I', '1', '2']

    while len(pop_X) < n_samples:
        new_X = np.random.choice(allowed_choices, 13)
        new_hashX = convert_X_to_hashX(new_X)
        if new_hashX not in pop_hashX:
            pop_X.append(new_X)
            pop_hashX.append(new_hashX)
    return pop_X, pop_hashX


if __name__ == '__main__':
    np.random.seed(0)

    # X_train, y_train = pk.load(open('SVHN_dataset/training_data.p', 'rb'))
    # print('Load training data - done')
    # X_val, y_val = pk.load(open('SVHN_dataset/validating_data.p', 'rb'))
    # print('Load validation data - done')
    # X_test, y_test = pk.load(open('SVHN_dataset/testing_data.p', 'rb'))
    # print('Load testing data - done')
    popX, pop_hashX = _sampling(2)
    for i in range(len(popX)):
        create_and_evaluate_model(pop_hashX[i])
        print(popX[i])

    # keras.backend.clear_session()
    #
    # model = keras.Sequential([
    #     keras.layers.Conv2D(32, (3, 3), padding='same',
    #                         activation='relu',
    #                         input_shape=(32, 32, 3)),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Conv2D(32, (3, 3), padding='same',
    #                         activation='relu'),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Conv2D(32, (3, 3), padding='same',
    #                         activation='relu'),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Conv2D(32, (3, 3), padding='same',
    #                         activation='relu'),
    #     keras.layers.MaxPooling2D((2, 2)),
    #     keras.layers.Dropout(0.3),
    #
    #     keras.layers.Conv2D(64, (3, 3), padding='same',
    #                         activation='relu'),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Conv2D(64, (3, 3), padding='same',
    #                         activation='relu'),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Conv2D(64, (3, 3), padding='same',
    #                         activation='relu'),
    #     keras.layers.MaxPooling2D((2, 2)),
    #     keras.layers.Dropout(0.3),
    #
    #     keras.layers.Conv2D(128, (3, 3), padding='same',
    #                         activation='relu'),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Conv2D(128, (3, 3), padding='same',
    #                         activation='relu'),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Conv2D(128, (3, 3), padding='same',
    #                         activation='relu'),
    #     keras.layers.MaxPooling2D((2, 2)),
    #     keras.layers.Dropout(0.3),
    #
    #     keras.layers.Conv2D(256, (3, 3), padding='same',
    #                         activation='relu'),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Conv2D(256, (3, 3), padding='same',
    #                         activation='relu'),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Conv2D(256, (3, 3), padding='same',
    #                         activation='relu'),
    #     keras.layers.MaxPooling2D((2, 2)),
    #     keras.layers.Dropout(0.3),
    #
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(512, activation='relu'),
    #     keras.layers.Dropout(0.4),
    #     keras.layers.Dense(10, activation='softmax')
    # ])
    #
    # early_stopping = keras.callbacks.EarlyStopping(patience=8)
    # optimizer = keras.optimizers.Adam()
    # data = ImageDataGenerator(rotation_range=8,
    #                           zoom_range=[0.95, 1.05],
    #                           height_shift_range=0.10,
    #                           shear_range=0.15)
    #
    # model.compile(optimizer=optimizer,
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    #
    # _ = model.fit_generator(data.flow(X_train, y_train, batch_size=128), epochs=30, validation_data=(X_val, y_val),
    #                               callbacks=[early_stopping])
    #
    # _, test_acc = model.evaluate(x=X_test, y=y_test, verbose=1)
    # print(test_acc)
