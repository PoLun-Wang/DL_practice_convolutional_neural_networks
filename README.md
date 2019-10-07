# Deep Learning practice - CNNs
 - We want to use an easy way to introduce the popular technologies of computer vision for people who are interested in this field.
 - These programs are basically modified from the exams in "Convolutional Neural Networks" on Coursera. If you want a comprehensive explanation or any tutorial videos, I recommended you enroll in the course on Coursera.
 - If you need more information you could go to my website or leave message to me.
     - [BrilliantCode.net](https://www.brilliantcode.net/)

## Programs
### Basic concepts
#### Step by step build components of CNNs.
  - Descriptions:
    - We used Numpy to build each component of CNNs including convolutional layer, pooling layer, padding, strides and their backpropagations.
  - Related files:
    - 01.The components of CNNs.ipynb
  - Related tutorial:
      - [CNN #1 Kernel, Stride, Padding](https://www.brilliantcode.net/1584/convolutional-neural-networks-1-convolution-layer-stride-padding-kernel/)
      - [CNN #2 池化層(Pooling layer)](https://www.brilliantcode.net/1586/convolutional-neural-networks-2-pooling-layer/)
      - [CNN #3 計算參數量](https://www.brilliantcode.net/1646/convolutional-neural-networks-3-calculate-number-of-parameters/)
      - [CNN #4 卷積核的Back propagation](https://www.brilliantcode.net/1670/convolutional-neural-networks-4-backpropagation-in-kernels-of-cnns/)
      - [CNN #5 特徵圖&偏差值的導數](https://www.brilliantcode.net/1748/convolutional-neural-networks-5-backpropagation-in-feature-maps-biases-of-cnns/)
      - [CNN #6 Pooling in Backward pass](https://www.brilliantcode.net/1781/convolutional-neural-networks-6-backpropagation-in-pooling-layers-of-cnns/)

### Keras
#### Classify MNIST dataset using linear regression model
   - Descriptions:
     - We used MNIST dataset to train a linear model. (**02.keras_MNIST_linear.py**)
     - And we saved the settings and weights of this model as "**02.MNIST_cnn_model_linear.config**" and "**02.MNIST_cnn_model_linear.weights**", respectively.
     - In addition, you could run **02.MNIST_using_linear.py** to test this model through the images that I paint by mouse. (These images painted by mouse were put in folder "**02-04.images**".)
   - Related files:
     - 02.keras_MNIST_linear.py
     - 02.MNIST_using_linear.py
     - 02.MNIST_model_linear.config
     - 02.MNIST_model_linear.weights
     - 02-04.numbers_groundtruth.json
     - 02-04.images/*

#### Classify MNIST dataset using LeNet-5 model
  - Descriptions:
    - We used MNIST dataset to train a LeNet-5 model. (**03.keras_MNIST_LeNet5.py**)
    - And we saved the settings and weights of this model as "**03.MNIST_cnn_model_LeNet5.config**" and "**03.MNIST_cnn_model_LeNet5.weights**", respectively.
    - In addition, you could run **03.MNIST_cnn_using_LeNet5.py** to test this model through the images that I paint by mouse. (These images painted by mouse were put in folder "**02-04.images**".)
  - Related files:
    - 03.keras_MNIST_LeNet5.py
    - 03.MNIST_cnn_using_LeNet5.py
    - 03.MNIST_cnn_model_LeNet5.config
    - 03.MNIST_cnn_model_LeNet5.weights
    - 02-04.numbers_groundtruth.json
    - 02-04.images/*

#### Classify MNIST dataset using VGG-16 model
  - Descriptions:
    - We used MNIST dataset to train a VGG-16 model *which was modified the numbers of the filter in the final dence layer*. (**04.keras_MNIST_VGG16.py**)
    - And we saved the settings and weights of this model as "**04.MNIST_cnn_model_VGG16.config**" and "**04.MNIST_cnn_model_VGG16.weights**", respectively.
    - In addition, you could run **04.MNIST_cnn_using_VGG16.py** to test this model through the images that I paint by mouse. (These images painted by mouse were put in folder "**02-04.images**".)
    - Due to the shape of the input layer of VGG-16 is 224x224, so we designed a program for resizing images, and the reized images were saved as **04.MNIST_shape224.npz**. (Run **04.resize_images.py** for resizing images.)
  - Related files:
    - 04.keras_MNIST_VGG16.py
    - 04.MNIST_cnn_using_VGG16.py
    - 04.resize_images.py
    - 04.MNIST_shape224.npz
    - 04.MNIST_cnn_model_VGG16.config
    - 04.MNIST_cnn_model_VGG16.weights
    - 02-04.numbers_groundtruth.json
    - 02-04.images/*

#### Classify MNIST dataset using LeNet-5 model on the Kaggle competition
  - Descriptions:
    - We built a CNNs model based on LeNet-5, and trained it through the MNIST dataset from Kaggle.
    - First of all, we wrote a program (**05.keras_Kaggle_MNIST_preprocess.py**) to preprocess the MNIST dataset from Kaggle and saved as a npz file called "**05.Kaggle_digit-recognizer.npz**".
    - After then, use the program "**05.keras_Kaggle_MNIST_train_LeNet5.py**" to train this model. And this program saved the trained model configuration and weights as "**05.MNIST_cnn_model_Kaggle_LeNet5.config**" and "**05.MNIST_cnn_model_Kaggle_LeNet5.weights**", respectively.
    - Finally, we ran the program (**05.keras_Kaggle_MNIST_test_LeNet5.py**) to predict the test dataset for submission.
    - **In this model**, we replaced the original activation function (Sigmoid) of LeNet-5 with **ReLU**, and we added **2 dropout layers** into the fully connected layers. And the training **epoch** increased to 100. Everything else in LeNet-5 had no changes.
    - And the prediction accuracy of this model is 98.528%.
    - It seems not very high accuracy. As you can see the architecture of this model is quite simple, so I thought the results seem to be acceptable.
  - Related files:
    - 05.keras_Kaggle_MNIST_preprocess.py
    - 05.keras_Kaggle_MNIST_test_LeNet5.py
    - 05.keras_Kaggle_MNIST_train_LeNet5.py
    - 05.Kaggle_MNIST_test.csv
    - 05.Kaggle_MNIST_train.csv
    - 05.Kaggle_digit-recognizer.npz
    - 05.Kaggle_prediction_for_submission.csv
    - 05.MNIST_cnn_model_Kaggle_LeNet5.config
    - 05.MNIST_cnn_model_Kaggle_LeNet5.weights
  - Related links:
    - [Digit recognizer - Kaggle](https://www.kaggle.com/c/digit-recognizer)

#### Classify CIFAR-10 dataset using the model conbined with ResNet and traditional CNNs on the Kaggle competition
  - Descriptions:
    - We built this model mainly is using the traditional CNNs to extract features from images, and used the **convolutional block of ResNet** to connect two traditional convolutional layers which have different number of kernels.
    - The dataset provided by this competition was not really good to use, because it presented as 50,000 PNG for training dataset and 300,000 PNG for testing dataset. As the preprocessing phase, we spent a lot of time to uncompress these images and we met a disaster that PyCharm tried to pre-load the folder full of a bunch of images. Fortunately, we had already preprocessed these images and saved them as npz files.(**06.Kaggle_CIFAR-10_train.npz**, **06.Kaggle_CIFAR-10_test.npz**) If you want to try this competition, just load these npz file by Numpy.
    - The testing prediction accuracy is 86.63%.
  - Related files:
    - 06.CIFAR-10_CombinedNet.config
    - 06.CIFAR-10_CombinedNet.weights
    - 06.Kaggle_CIFAR-10_test.npz
    - 06.Kaggle_CIFAR-10_train.npz
    - 06.Kaggle_CIFAR-10_trainLabels.csv
    - 06.Kaggle_submission.csv
    - 06.keras_Kaggle_CIFAR10_test.py
    - 06.keras_Kaggle_CIFAR10_train.py
    - 06.preprocess_Kaggle_CIFAR-10.py
  - Related links:
    - [CIFAR-10 - Object Recognition in Images - Kaggle](https://www.kaggle.com/c/cifar-10/)

#### Classify Kannada-MNIST dataset using CNNs
  - Descriptions:
    - We built this model basically mimics the architecture of the VGG16, but the most different design is this model only has 10 CONV layers.
    - We didn't provide the datasets in this repository. So, if you want to run this model, you have to download the datasets from Kaggle.
    - The testing prediction accuracy is 97.88%.
  - Related files:
    - 07.Keras_Kaggle_Kannada-MNIST.ipynb
  - Related links:
    - [A simple model with 10-layer CNNs](https://www.kaggle.com/polunwang/a-simple-model-with-10-layer-cnns)
    - [MNIST like datatset for Kannada handwritten digits - Kaggle](https://www.kaggle.com/c/Kannada-MNIST)

## How to use?
 - Just fork to your own GitHub or download this repository directly. And run the programs you interested.

## References
 - "Convolutional Neural Networks" on Coursera.
 - [Keras Documentation](https://keras.io/)
