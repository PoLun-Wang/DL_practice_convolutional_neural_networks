import numpy as np
from keras.layers import Input, Add, Dense, Activation, Dropout, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.utils import np_utils
from keras.initializers import glorot_normal
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split

def identity_block(X, filters, stage, block, kernel_size=3, strides=1):
    """
    Implementation of the identity block of ResNet.

    :param X: Input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    :param f: Integer, specifying the shape of the middle CONV's window for the main path
    :param filters: Python list of integers, defining the number of filters in the CONV layers of the main path
    :param stage: Integer, used to name the layers, depending on their position in the network
    :param block: String/character, used to name the layers, depending on their position in the network
    :return: Output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res_' + str(stage) + block
    bn_name_base =   'bn_' + str(stage) + block

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=1, strides=strides, padding='valid', name=conv_name_base + 'a',
               kernel_initializer=glorot_normal(), kernel_regularizer=l2())(X)
    X = BatchNormalization(axis=3, name=bn_name_base + 'a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=kernel_size, strides=(1, 1), padding='same', name=conv_name_base + 'b',
               kernel_initializer=glorot_normal(), kernel_regularizer=l2())(X)
    X = BatchNormalization(axis=3, name=bn_name_base + 'b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=1, strides=(1, 1), padding='valid', name=conv_name_base + 'c',
               kernel_initializer=glorot_normal(), kernel_regularizer=l2())(X)
    X = BatchNormalization(axis=3, name=bn_name_base + 'c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def conv_block(X, filters, stage, block, kernel_size=3, strides=1):
    """
    Implementation of the convolutional block of ResNet.

    :param X: Input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    :param f: Integer, specifying the shape of the middle CONV's window for the main path
    :param filters: Python list of integers, defining the number of filters in the CONV layers of the main path
    :param stage: Integer, used to name the layers, depending on their position in the network
    :param block: String/character, used to name the layers, depending on their position in the network
    :return: Output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res_' + str(stage) + block
    bn_name_base =   'bn_' + str(stage) + block

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = Conv2D(filters=F3, kernel_size=1, strides=strides, padding='valid', name=conv_name_base + 's',
               kernel_initializer=glorot_normal(), kernel_regularizer=l2())(X)

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=1, strides=strides, padding='valid', name=conv_name_base + 'a',
               kernel_initializer=glorot_normal(), kernel_regularizer=l2())(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=kernel_size, strides=(1, 1), padding='same', name=conv_name_base + 'b',
               kernel_initializer=glorot_normal(), kernel_regularizer=l2())(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=1, strides=(1, 1), padding='valid', name=conv_name_base + 'c',
               kernel_initializer=glorot_normal(), kernel_regularizer=l2())(X)
    X = BatchNormalization(axis=3)(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def CombinedNet(input_shape=(32, 32, 3), classes=6):
    """
    We combined the ResNet and traditional CNNs together.

    :param input_shape: Shape of the images of the dataset
    :param classes: Integer, number of classes
    :return: a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Conv. group 1
    X = Conv2D(32, kernel_size=3, strides=1, name='conv0_1', padding='same',
               kernel_initializer=glorot_normal(), kernel_regularizer=l2())(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(32, kernel_size=3, strides=1, name='conv0_2', padding='same',
               kernel_initializer=glorot_normal(), kernel_regularizer=l2())(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=2)(X)

    # Stage 1
    X = conv_block(X, kernel_size=3, strides=1, filters=[32, 32, 128], stage=1, block='1')

    # Conv. group 2
    X = Conv2D(64, kernel_size=3, strides=1, name='conv2_1', padding='same',
               kernel_initializer=glorot_normal(), kernel_regularizer=l2())(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(64, kernel_size=3, strides=1, name='conv2_2', padding='same',
               kernel_initializer=glorot_normal(), kernel_regularizer=l2())(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=2)(X)

    # Stage 2
    X = conv_block(X, kernel_size=3, strides=1, filters=[64, 64, 256], stage=2, block='1')

    # Conv. group 3
    X = Conv2D(128, kernel_size=3, strides=1, name='conv3_1', padding='same',
               kernel_initializer=glorot_normal(), kernel_regularizer=l2())(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(128, kernel_size=3, strides=1, name='conv3_2', padding='same',
               kernel_initializer=glorot_normal(), kernel_regularizer=l2())(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=2)(X)

    # output layer
    X = Flatten()(X)
    X = Dense(64, activation='relu', name='fc1', kernel_initializer=glorot_normal(), kernel_regularizer=l2())(X)
    X = Dropout(0.2)(X)
    X = Dense(64, activation='relu', name='fc2', kernel_initializer=glorot_normal(), kernel_regularizer=l2())(X)
    X = Dropout(0.2)(X)
    X = Dense(classes, activation='softmax', name='fc_final', kernel_initializer=glorot_normal(), kernel_regularizer=l2())(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model


## Settings in training phase.
batch_size = 256
num_classes = 10
epochs = 200

## Load training dataset.
CIFAR10_dataset = np.load('06.Kaggle_CIFAR-10_train.npz')
CIFAR10_x, CIFAR10_y = CIFAR10_dataset["x_train"], CIFAR10_dataset["y_train"]

## Split CIFAR-10 dataset into training dataset and dev. dataset.
x_train, x_dev, y_train, y_dev = train_test_split(CIFAR10_x, CIFAR10_y, test_size=0.1, random_state=0)
print('Size of training set:  {0}'.format(x_train.shape[0]))
print('Size of dev. set:      {0}'.format(x_dev.shape[0]))

## Size of input images.
img_x, img_y = 32, 32
input_shape = (img_x, img_y, 3)

## Normalize the input images' value.
x_train = x_train.astype('float32')/255.0
x_dev = x_dev.astype('float32')/255.0

## Set ground truth. (Transfer y_train and y_dev into one-hot vectors.)
y_train_orig = y_train
y_dev_orig = y_dev
y_train = np_utils.to_categorical(y_train, num_classes)
y_dev = np_utils.to_categorical(y_dev, num_classes)

## Image data generator.
img_gen = ImageDataGenerator( featurewise_center=False,
                              samplewise_center=False,
                              featurewise_std_normalization=False,
                              samplewise_std_normalization=False,
                              zca_whitening=False,
                              rotation_range=8,
                              zoom_range=0.10,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              horizontal_flip=False,
                              vertical_flip=False)

## Create ResNet
model = CombinedNet(input_shape = input_shape, classes = num_classes)

## Show model summary.
print(model.summary())

## Learning rate decay.
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.8), patience=4, verbose=2, cooldown=0, min_lr=0.5e-6)

## Strat training.
model.compile(optimizer=Adam(lr=1e-3),
              loss='categorical_crossentropy', metrics=['accuracy'])

img_gen.fit(x_train)
model.fit_generator(img_gen.flow(x_train, y_train, batch_size=batch_size),
                    epochs = epochs,
                    steps_per_epoch=x_train.shape[0]//batch_size,
                    verbose=2,
                    validation_data=(x_dev, y_dev),
                    callbacks=[lr_reduction])

## Save configuration and model weights.
with open('06.CIFAR-10_CombinedNet.config', 'w') as text_file:
    text_file.write(model.to_json())

model.save_weights('06.CIFAR-10_CombinedNet.weights')
