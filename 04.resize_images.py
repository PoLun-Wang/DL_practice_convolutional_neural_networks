import keras
from keras.datasets import mnist
import numpy as np
from PIL import Image

def resize(img_array, target_h, target_w):
    """
    Resize the image array, and return the resized image array.
    This function only supports gray images.
    :param img_array: Image array. (image_num, image_height, image_width)
    :param target_h: Target image height.
    :param target_w: Target image weight.
    :return: (image_num, target_h, target_w)
    """
    num_img = img_array.shape[0]
    img_resized = np.zeros((num_img, target_h, target_w, 1), dtype=np.float16)

    for i in range(num_img):
        tmp = img_array[i].astype('float16')
        tmp = np.expand_dims(tmp, axis=2)
        img = keras.preprocessing.image.array_to_img((tmp / 255))
        img = img.resize((target_h, target_w), resample=Image.NEAREST)
        tmp_img = keras.preprocessing.image.img_to_array(img)  # (224, 224, 1) => (img_h, img_w, 1-channel)

        img_resized[i] = tmp_img

        if ((i+1)%100 == 0):
            print('\t\t{0}% completed.'.format(np.int(i/num_img*100)))

    return img_resized

## Loads MNIST datasets.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_test_orig = y_test

# x_train.shape  (60000,28,28,1)
print('Start to resize...')
print('  Resizing x_train:')
resized_train = resize(x_train, 224, 224)
print('  Resizing x_test:')
resized_test = resize(x_test, 224, 224)

## Save resized images.
np.savez_compressed('04.MNIST_shape224.npz', x_train=resized_train, y_train=y_train, x_test=resized_test, y_test=y_test)
print('Saving completed.')