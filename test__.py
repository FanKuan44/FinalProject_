import random

import keras
import kerop
import numpy as np


def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]


def create_and_evaluate_model(x):
    x = remove_values_from_list(x.tolist(), 'I')

    keras.backend.clear_session()
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))

    if len(x) != 0:
        if x[0] == '1':
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        elif x[0] == '2':
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu'))

        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Dropout(0.25))

        filters = 64

        for i in range(1, len(x)):
            if x[i] == '1':
                model.add(keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu'))
            else:
                model.add(keras.layers.Conv2D(filters, (5, 5), padding='same', activation='relu'))
            if i % 2 == 0:
                model.add(keras.layers.MaxPooling2D((2, 2)))
                model.add(keras.layers.Dropout(0.25))
                filters *= 2
            else:
                model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(10, activation='softmax'))

    # model.summary()
    _, layer_flops, _, _ = kerop.profile(model)
    return sum(layer_flops) / 1e6

    # data = ImageDataGenerator(rotation_range=8,
    #                           zoom_range=[0.95, 1.05],
    #                           height_shift_range=0.10,
    #                           shear_range=0.15)
    # early_stopping = keras.callbacks.EarlyStopping(patience=8)
    # optimizer = keras.optimizers.Adam(lr=1e-3, amsgrad=True)
    #
    # model.compile(optimizer=optimizer,
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    #
    # # Fit model in order to make predictions
    # history = model.fit_generator(generator=data.flow(X_val, y_val, batch_size=128),
    #                               steps_per_epoch=10, epochs=1,
    #                               validation_data=(X_val, y_val),
    #                               callbacks=[early_stopping])
    #
    # # Evaluate train and validation accuracies and losses
    # val_acc = history.history['val_accuracy']
    #
    # val_loss = history.history['val_loss']
    #
    # print(val_acc, val_loss)


def initialize(n_samples):
    allowed_choices = ['I', '1', '2']
    pop_X = np.random.choice(allowed_choices, (n_samples, 9))
    return pop_X


if __name__ == '__main__':
    np.random.seed(20)
    random.seed(20)

    # X_train, y_train = pk.load(open('training_data.p', 'rb'))
    # print('Load training data - done')

    # X_val, y_val = pk.load(open('SVHN_dataset/validating_data.p', 'rb'))
    # print('Load validation data - done')
    #
    # X_test, y_test = pk.load(open('SVHN_dataset/testing_data.p', 'rb'))
    # print('Load testing data - done')

    # Define actual model

    # allowed = ['I', '1', '2']
    # POP_X = np.random.choice(allowed, size=(3**9 * 10, 9), replace=True)
    #
    # POP_X = np.unique(POP_X, axis=0)
    #
    # MIN_FLOPs = np.inf
    # model_minFLOPs = None
    #
    # MAX_FLOPs = -np.inf
    # model_maxFLOPs = None
    #
    # for X in POP_X:
    #     flops = create_and_evaluate_model(X)
    #     if flops < MIN_FLOPs:
    #         MIN_FLOPs = flops
    #         model_minFLOPs = X
    #     if flops > MAX_FLOPs:
    #         MAX_FLOPs = flops
    #         model_maxFLOPs = X
    # print(model_minFLOPs, MIN_FLOPs)
    # print(model_maxFLOPs, MAX_FLOPs)

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(32, 32, 3)))

    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.summary()


    ''' Sua lai trong cifar10, cifar100 '''
    # benchmark = pk.load(open('bosman_benchmark/cifar10/cifar10.p', 'rb'))
    #
    # X1 = ['2','I','I','2','1','2','I','1','1','1','2','2','2','2']
    # X2 = ['I','2','2','I','1','I','2','1','1','1','2','2','2','2']
    #
    # hashX1 = ''.join(X1)
    # hashX2 = ''.join(X2)
    #
    # print(benchmark[hashX1]['val_acc'], benchmark[hashX2]['val_acc'])
    # added = ["|", "|", "|"]
    # pos = [4, 8, 12]
    # hashX1 = X1
    # hashX2 = X2
    # assert (len(added) == len(pos))
    # acc = 0
    # for i in range(len(added)):
    #     hashX1.insert(pos[i] + acc, added[i])
    #     hashX2.insert(pos[i] + acc, added[i])
    #     acc += 1
    # hashX1 = remove_values_from_list(hashX1, 'I')
    # hashX1 = ''.join(hashX1)
    # hashX2 = remove_values_from_list(hashX2, 'I')
    # hashX2 = ''.join(hashX2)
    # print(hashX1 == hashX2)
