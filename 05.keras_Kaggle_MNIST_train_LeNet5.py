import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np

## Settings in training phase.
batch_size = 256
num_classes = 10
epochs = 100

## Loads MNIST datasets(Kaggle).
MNIST_dataset = np.load('05.Kaggle_digit-recognizer.npz')
mnist_x, mnist_y = MNIST_dataset['x_train'], MNIST_dataset['y_train']

## Split the MNIST dataset into training set and developmnet set.
x_train, x_dev, y_train, y_dev = train_test_split(mnist_x, mnist_y, test_size=0.05, random_state=1)
print('Size of training set:  {0}'.format(x_train.shape[0]))
print('Size of dev. set:      {0}'.format(x_dev.shape[0]))

print(y_dev.shape)

## Size of input images.
img_x, img_y = 28, 28
input_shape = (28, 28, 1)

## Set ground truth.
y_train = np_utils.to_categorical(y_train, num_classes=num_classes)
y_dev = np_utils.to_categorical(y_dev, num_classes=num_classes)

## Normalize the input images' value.
x_train = x_train.astype('float32')
x_train /= 255

## Initialize model.
model = Sequential()
## LeNet-5
 # Note 1: The shape of input images were changed to be (28x28), instead of (32x32).
 # Note 2: We replaced the activation functions with ReLU.
 # Note 3: 2 dropout layers were added into the last two fully connected layers.
model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(units=120, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.SGD(),
                metrics=['accuracy'])

## Show model summary.
print(model.summary())

## Start training model.
train_history = model.fit(x_train, y_train, epochs=epochs,
                            batch_size=batch_size, verbose=2, validation_data=(x_dev, y_dev))

## Save the configurations and weights of the model we just trained.
with open('05.MNIST_cnn_model_Kaggle_LeNet5.config', 'w') as text_file:
    text_file.write(model.to_json())

model.save_weights('05.MNIST_cnn_model_Kaggle_LeNet5.weights')