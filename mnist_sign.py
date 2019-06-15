import tensorflow as tf
import numpy as np
import csv
from keras import backend as K
import keras.preprocessing.image import ImageDataGenerator
from google.colab import files
from keras.models import Model, Sequential, model_from_json
from keras.layers.convolutional import Conv2D, Conv3D
from keras.layers import AveragePooling2D, MaxPooling2D, Activation, Add, ActivityRegularization, Dense, Dropout, Flatten
import matplotlib.pyplot as plt

uploaded = files.upload()

def get_data(filename):
    with open(filename) as training_file:
        csv_reader = csv.reader(training_file, dilimeter = ',')
        first_line = True
        temp_images = []
        temp_labels = []
        for row in csv_reader:
            if first_line:
                print("Ignorig the first line")
                first_line = False
        else:
            temp_labels.append(row[0])
            image_data = row[1:785]
            image_data_as_array = np.array_split(image_data, 28)
            temp_images.append(image_data_as_array)
        images = np.array(temp_images).astype('float')
        labels = np.array(temp_labels).astype('float')

    return images, labels

training_images, training_labels = get_data('sign_mnist_train.csv')
testing_images, testing_labels = get_data('sign_mnist_test.csv')

print(training_images.shape)
print(testing_images.shape)
print(training_labels.shape)
print(testing_labels.shape)

training_images = np.expand_dims(training_images, axis = 3)
testing_images = np.expand_dims(testing_images, axis = 3)

train_datagen = ImageDataGenerator(rescale=  1./255., rotation_range = 40, width_shift_range = 0.2, heightshift_range = 0.2, shear_range = 0.2, zoom_range = 0.2,
                                   horizontal_flip = True, fill_mode = 'nearest')
validation_datagen = ImageDataGenerator(rescale = 1./255.)

print(training_images.shape)
print(testing_images.shape)

model = Sequential()

model.add(Conv2D(64, (3, 3), strides = (1, 1), padding = 'valid', activation = 'relu', input_shape = (28, 28, 1)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), strides = (2, 2), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation = tf.nn.relu))
model.add(Dense(26, activation = tf.nn.softmax))

model.compile(optimizer = tf.train.AdamOptimizer(), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

history = model.fit_generator(train_datagen.flow_from_directory(training_images, training_labels, batch_size = 32), steps_per_epoch = len(training_images)/32,
                              epochs = 16, validation_data = validation_datagen.flow_from_directory(testing_images, testing_labels, batch_size = 32),
                              validation_steps = len(testing_images)/32)

model.evaluate(testing_images, testing_labels)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label = 'Training_accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label = 'Validation Accuracy')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.legend()
plt.figure()

plt.show()















