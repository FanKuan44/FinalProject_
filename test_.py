import itertools

stuff = [1, 2]
count = 0
for L in range(0, len(stuff)+1):
    for subset in itertools.combinations(stuff, L):
        print(subset)
        count += 1
print(count)

# X_train, y_train = pk.load(open('training_data.p', 'rb'))
# print('Load training data - done')
# X_val, y_val = pk.load(open('validation_data.p', 'rb'))
# print('Load validation data - done')
#
# # Data augmentation
# datagen = ImageDataGenerator(rotation_range=8,
#                              zoom_range=[0.95, 1.05],
#                              height_shift_range=0.10,
#                              shear_range=0.15)
#
# # Define actual model
# keras.backend.clear_session()
#
# model = keras.Sequential([
#     keras.layers.Conv2D(32, (3, 3), padding='same',
#                         activation='relu',
#                         input_shape=(32, 32, 3)),
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
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Dropout(0.3),
#
#     keras.layers.Conv2D(128, (3, 3), padding='same',
#                         activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.Conv2D(128, (3, 3), padding='same',
#                         activation='relu'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Dropout(0.3),
#
#     keras.layers.Flatten(),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dropout(0.4),
#     keras.layers.Dense(10, activation='softmax')
# ])
#
# early_stopping = keras.callbacks.EarlyStopping(patience=8)
# optimizer = keras.optimizers.Adam(lr=1e-3, amsgrad=True)
# model_checkpoint = keras.callbacks.ModelCheckpoint(
#     'best_cnn.h5',
#     save_best_only=True)
# model.compile(optimizer=optimizer,
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# # Fit model in order to make predictions
# print(len(datagen.flow(X_train, y_train, batch_size=128)))
# history = model.fit_generator(generator=datagen.flow(X_train, y_train, batch_size=128),
#                               steps_per_epoch=200, epochs=1,
#                               validation_data=(X_val, y_val),
#                               callbacks=[early_stopping, model_checkpoint])
#
# # Evaluate train and validation accuracies and losses
# train_acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
#
# train_loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# print(val_acc, val_loss)