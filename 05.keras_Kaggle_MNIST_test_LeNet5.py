from keras.models import Sequential
from keras.models import model_from_json
import keras
import numpy as np
import csv

model = None

## Loads model weights.
with open('05.MNIST_cnn_model_Kaggle_LeNet5.config', 'r') as text_file:
    json_config = text_file.read()
    model = Sequential()
    model = model_from_json(json_config)
    model.load_weights('05.MNIST_cnn_model_Kaggle_LeNet5.weights')
print(model.summary())
# Compile model.
model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

## Read image
MNIST_dataset = np.load('05.Kaggle_digit-recognizer.npz')
x_test = MNIST_dataset['x_test']

## Predict the x_test.
predictions = model.predict_classes(x_test)
print('Prediction completed.')

## Save as CSV for submission.
with open('05.Kaggle_prediction_for_submission.csv', 'w', newline='') as csv_file:
    print('Saving file...')
    csv_writer = csv.writer(csv_file, delimiter=',')

    # Define column name.
    csv_writer.writerow(['ImageId', 'Label'])
    for i in range(len(predictions)):
        csv_writer.writerow([i+1, predictions[i]])

    print('File: 05.Kaggle_prediction_for_submission.csv Saved completed.')