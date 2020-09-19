import splitfolders
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
from keras.models import Sequential
from keras.optimizers import SGD
import os


import tensorflowjs as tfjs

path = './images'
labels = os.listdir(path)

splitfolders.ratio('./images/', output="./data", seed=1337, ratio=(.7, .3))  # default values

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('./data/train', target_size=(64, 64), batch_size=32,
                                                         class_mode='categorical', shuffle=False)
test_set = test_datagen.flow_from_directory('./data/val', target_size=(64, 64), batch_size=32,
                                                    class_mode='categorical', shuffle=False)

model = Sequential()
model.add(Conv2D(32, (3, 3), strides=(2, 2), input_shape=(64, 64, 3)))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))
model.add(Activation('relu'))  # 2nd hidden layer
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))
model.add(Activation('relu'))  # 3rd hidden layer
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))
model.add(Activation('relu'))  # Flatten
model.add(Flatten())
model.add(Dropout(rate=0.5))  # Add fully connected layer.
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))  # Output layer
model.add(Dense(len(labels)))
model.add(Activation('softmax'))
model.summary()

steps=100
val_steps=200
epochs=40
learning_rate=0.01
momentum=0.9

sgd = SGD(lr=learning_rate, momentum=momentum, decay=learning_rate / epochs, nesterov=False)
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=['accuracy', 'mse'])

model.fit(training_set, steps_per_epoch=steps, epochs=epochs, validation_data=test_set,
                    validation_steps=val_steps)

tfjs.converters.save_keras_model(model, './js')
tfjs.converters.save_keras_model(model, './js')
model.save('model.h5')
model.save_weights('model_weight.h5')

score = model.evaluate(training_set, verbose=0)
print("Training Accuracy: ", score[1] * 100)
print("Training mse: ", score[2])
score = model.evaluate(test_set, verbose=0)
print("Testing Accuracy: ", score[1] * 100)
print("Testing mse: ", score[2])
