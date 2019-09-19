from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense
from keras.utils import np_utils  # For transformation of one-hot-encoding.

## Settings in training phase.
batch_size = 1024   # Change batch size based on the capability of your resource.
num_classes = 10   # MNIST
epochs = 12

## Load MNIST dataset.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

## Normalize the input images.
x_train = x_train.reshape(60000, 28*28).astype('float32')
x_test = x_test.reshape(10000, 28*28).astype('float32')
x_train = x_train/255
x_test = x_test/255

## Initialize model.
model = Sequential()
## Build model.
model.add(Dense(units=512, input_dim=784, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=128, input_dim=512, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

## Show model summary
print(model.summary())

## Transfer ground truth to one-hot format. (e.g. 7=>0000001000)
y_train = np_utils.to_categorical(y_train, num_classes=num_classes)
y_test = np_utils.to_categorical(y_test, num_classes=num_classes)

## Start training this model.
train_history = model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=(x_test, y_test))

## Show results.
scores = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

## Save configurations and weights of this model.
with open('02.MNIST_model_linear.config', 'w') as text_file:
    text_file.write(model.to_json())

model.save_weights('02.MNIST_model_linear.weights')