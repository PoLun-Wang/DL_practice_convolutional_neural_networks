from keras.models import Sequential
from keras.models import model_from_json
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
import keras
import numpy as np
import json

model = None

## Loads model weights.
with open('02.MNIST_model_linear.config', 'r') as text_file:
    json_config = text_file.read()
    model = Sequential()
    model = model_from_json(json_config)
    model.load_weights('02.MNIST_model_linear.weights')
print(model.summary())
# Compile model.
model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

## Read images.
num_images = 30
test_images = np.array([])
for i in range(num_images):
    img = load_img('02-04.images/numbers_' + str(i) +'.jpg', color_mode='grayscale', target_size=(28, 28))
    # img = img.resize((28, 28)) # Resize image.
    img_arr = img_to_array(img).reshape(1, 28*28)  # Reshape into size of (1, 784).
    if test_images.shape[0] == 0:
        test_images = img_arr
    else:
        test_images = np.append(test_images, img_arr, axis=0)

## Read ground truth.
with open('02-04.numbers_groundtruth.json', 'r') as text_file:
    text_json = text_file.read()
    ans = json.loads(text_json)
    test_groundtruth = np.array(ans['ans'])

## Evaluate the images we loaded.
y_test = np_utils.to_categorical(test_groundtruth, num_classes=10)
score = model.evaluate(test_images, y_test, verbose=1)
print('Loss:\t', score[0])
print('Accuracy:\t', score[1])

## Show the predictions and the corresponding ground truth.
predictions = model.predict_classes(test_images)
print('Prediction:    {0}'.format(predictions))
print('Ground truth:  {0}'.format(test_groundtruth))

## Show the images if you want to.
# for i in range(num_images):
#     keras.preprocessing.image.array_to_img(test_images[i]).show()