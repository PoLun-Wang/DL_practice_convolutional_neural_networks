from keras.models import model_from_json
from keras.optimizers import RMSprop
import numpy as np
import csv

## Loads model weights.
with open('06.CIFAR-10_CombinedNet.config', 'r') as text_file:
    json_config = text_file.read()
    model = model_from_json(json_config)
    model.load_weights('06.CIFAR-10_CombinedNet.weights')
print(model.summary())

## Read image
MNIST_dataset = np.load('06.Kaggle_CIFAR-10_test.npz')
x_test = MNIST_dataset['x_test']
x_test = x_test.astype('float32')/255.0

## Predict the x_test.
predictions = model.predict(x_test)
print('Prediction completed.')
predictions = np.argmax(predictions, axis=1)

## Save as CSV for submission.
with open('06.Kaggle_submission.csv', 'w', newline='') as csv_file:
    print('Saving file...')
    csv_writer = csv.writer(csv_file, delimiter=',')

    # Define column name.
    csv_writer.writerow(['id', 'label'])
    for i in range(len(predictions)):
        label = ''
        if predictions[i] == 0:
            label = 'airplane'
        elif predictions[i] == 1:
            label = 'automobile'
        elif predictions[i] == 2:
            label = 'bird'
        elif predictions[i] == 3:
            label = 'cat'
        elif predictions[i] == 4:
            label = 'deer'
        elif predictions[i] == 5:
            label = 'dog'
        elif predictions[i] == 6:
            label = 'frog'
        elif predictions[i] == 7:
            label = 'horse'
        elif predictions[i] == 8:
            label = 'ship'
        elif predictions[i] == 9:
            label = 'truck'

        csv_writer.writerow([i+1, label])

    print('File: 06.Kaggle_submission.csv Saved completed.')
