import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from keras.utils import np_utils
from keras import backend as K

## Settings in training phase.
batch_size = 100   # Change batch size based on the capability of your resource.
num_classes = 10   # MNIST
epochs = 12

## Size of input images.
img_x, img_y = 28, 28

## Loads MNIST datasets.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_test_orig = y_test

## Show one of the images in training set.
# tmp = x_train[0].astype('float32')
# tmp = np.expand_dims(tmp, axis=3)
# print(tmp)
# keras.preprocessing.image.array_to_img( (tmp/255) ).show()

## channels_first: It means the color channels are located at the 2nd dimension of the whole dataset.
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_x, img_y)
    x_test = x_test.reshape(x_test.shape[0], 1, img_x, img_y)
    imput_shape = (1, img_x, img_y)
# channels_last: It means the color channels are located at the 4th of the whole dataset.
#                Besides, the images' height and width are located at the 2nd and 3rd dimension.
else:
    x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
    x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
    input_shape = (img_x, img_y, 1)

## Normalize the images' value.
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

## Set ground truth to be one-hot value.
y_train = np_utils.to_categorical(y_train, num_classes=num_classes)
y_test = np_utils.to_categorical(y_test, num_classes=num_classes)

## Initialize a model.
model = Sequential()
## LeNet-5.
 # Note: The shape of input images were changed to be (28x28), instead of (32x32).
model.add(Conv2D(6, kernel_size=(5, 5), activation='sigmoid', input_shape=input_shape))
model.add(AveragePooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(16, kernel_size=(5, 5), activation='sigmoid'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=2))

model.add(Flatten())
model.add(Dense(units=120, activation='sigmoid'))
model.add(Dense(units=84, activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

## Show model summary.
print(model.summary())

## Start training model.
train_history = model.fit(x_train, y_train, epochs=epochs,
                            batch_size=batch_size, verbose=1, validation_data=(x_test, y_test))

## Show results.
scores = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

## Save the configurations and weights of the model we just trained.
with open('03.MNIST_cnn_model_LeNet5.config', 'w') as text_file:
    text_file.write(model.to_json())

model.save_weights('03.MNIST_cnn_model_LeNet5.weights')