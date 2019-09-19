import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np

## Settings in training phase.
batch_size = 32    # Change batch size based on the capability of your resource.
num_classes = 10   # MNIST
epochs = 12

## Loads MNIST datasets.
MNIST_shape224 = np.load('04.MNIST_shape224.npz')
x_train, y_train = MNIST_shape224['x_train'], MNIST_shape224['y_train']
x_test, y_test = MNIST_shape224['x_test'], MNIST_shape224['y_test']

## Size of input images.
img_x, img_y = 224, 224
input_shape = (img_x, img_y, 1)

## Set ground truth to be one-hot value.
y_train = np_utils.to_categorical(y_train, num_classes=num_classes)
y_test = np_utils.to_categorical(y_test, num_classes=num_classes)

## Normalize the input images' value.
 # If your memory size of GPU is capable to carry the resource requirement of float32,
 # you can consider to uncomment it.
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

## Initialize Model.
model = Sequential()
## VGG-16.
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu', input_shape=input_shape))
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
## We changed the fully connected layers in VGG-16.
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(), metrics=['accuracy'])

## Show model summary.
print(model.summary())

## Strat to train model.
train_history = model.fit(x_train, y_train, epochs=epochs,
                          batch_size=batch_size, verbose=1, validation_data=(x_test, y_test))

## Show results.
scores = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

## Save the configurations and weights of the model we just trained.
with open('04.MNIST_cnn_model_VGG16.config', 'w') as text_file:
    text_file.write(model.to_json())

model.save_weights('04.MNIST_cnn_model_VGG16.weights')