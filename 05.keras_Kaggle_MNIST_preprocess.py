import csv
import numpy as np

def transfer_to_array(file_name, with_label=True):
    print('Start reading {0}.'.format(file_name))
    # Check row count.
    with open(file_name) as file:
        cnt = sum(1 for line in file)

    # Initialize variables.
    labels = np.zeros((cnt-1))
    data = np.zeros((cnt-1, 28, 28, 1))
    if with_label:
        label_bit = 1
    else:
        label_bit = 0

    # Transfer to array. (cnt, 28, 28, 1)
    with open(file_name, newline='') as csv_file:
        rows = csv.reader(csv_file)
        is_first_row = True
        i = 0
        for row in rows:
            if not is_first_row:
                if with_label:
                    labels[i] = np.int(row[0])
                tmp = np.array(row[label_bit:784+label_bit], dtype=np.int).reshape(28, 28)
                tmp = np.expand_dims(tmp, axis=3)
                data[i] = tmp
                i += 1
            else:
                is_first_row = False

    print('Reading completed.')

    # Show image.
    # img = image.array_to_img(data[3])
    # img.show()

    return data, labels


x_train, y_train = transfer_to_array('05.Kaggle_MNIST_train.csv', with_label=True)
x_test, y_test = transfer_to_array('05.Kaggle_MNIST_test.csv', with_label=False)  # y_test will not be used.
print('Saving...')
np.savez_compressed('05.Kaggle_digit-recognizer.npz', x_train=x_train, y_train=y_train, x_test=x_test)
print('File: 05.Kaggle_digit-recognizer.npz Saved completed.')