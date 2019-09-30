import numpy as np
import csv
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array, array_to_img

training_images_path = '../CIFAR-10_train/'
testing_images_path = '../CIFAR-10_test/'

## Initialize
dataset_count = 50000

# Save the final train data.
x_train = np.zeros((50000, 32, 32, 3), dtype=np.int)
y_train = np.zeros(50000, dtype=np.int)

x_test = None  # Avoid wasting resources, so we set is as None.

## Process Images of training dataset.
print("Start processing the \"Images\" of training dataset...")
for i in range(dataset_count):
    try:
        img = load_img(training_images_path + str(i+1) + '.png', 'r')
        img_array = img_to_array(img)
        x_train[i] = img_array

    except FileNotFoundError:
        print('{0}.png was not found'.format(i))
        pass
print("The \"Imgages\" of training dataset are completed processing.")


## Process labels.
print("Start processing the \"Labels\" of training dataset...")

# Load the labels from the training dataset csv.
with open('06.Kaggle_CIFAR-10_trainLabels.csv', 'r', newline='') as text_file:
    rows = csv.reader(text_file)

    is_first_row = True
    i = 0
    for row in rows:
        if not is_first_row:
            class_num = 0

            if row[1] == 'airplane':
                class_num = 0
            elif row[1] == 'automobile':
                class_num = 1
            elif row[1] == 'bird':
                class_num = 2
            elif row[1] == 'cat':
                class_num = 3
            elif row[1] == 'deer':
                class_num = 4
            elif row[1] == 'dog':
                class_num = 5
            elif row[1] == 'frog':
                class_num = 6
            elif row[1] == 'horse':
                class_num = 7
            elif row[1] == 'ship':
                class_num = 8
            elif row[1] == 'truck':
                class_num = 9

            y_train[i] = class_num
            i += 1
        else:
            is_first_row = False

print("The \"Labels\" of training dataset are completed processing.")

print("Start saving file: 06.Kaggle_CIFAR-10_train.npz ...")
np.savez_compressed(file="06.Kaggle_CIFAR-10_train.npz", x_train=x_train, y_train=y_train)
print("File: 06.Kaggle_CIFAR-10_train.npz saved completed.")

## Release resources.
 # P.s. The testing dataset contains 300,000 images.
x_train = None
y_train = None

## Process Images of testing dataset.
print("Start processing the \"Images\" of testing dataset...")
dataset_count = 300000
x_test = np.zeros((300000, 32, 32, 3), dtype=np.int)
for i in range(dataset_count):
    try:
        img = load_img(testing_images_path + str(i + 1) + '.png', 'r')
        img_array = img_to_array(img)
        x_test[i] = img_array
    except FileNotFoundError:
        print('{0}.png was not found'.format(i))
        pass
print("The \"Images\" of testing dataset are completed processing.")

print("Start saving file: 06.Kaggle_CIFAR-10_test.npz ...")
np.savez_compressed(file="06.Kaggle_CIFAR-10_test.npz", x_test=x_test)
print("File: 06.Kaggle_CIFAR-10_test.npz saved completed.")
