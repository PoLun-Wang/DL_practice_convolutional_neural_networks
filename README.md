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
    - Due to the shape of the input layer of VGG-16 is 244x244, so we designed a program for resizing images, and the reized images were saved as **04.MNIST_shape224.npz**. (Run **04.resize_images.py** for resizing images.)
  - Related files:
    - 04.keras_MNIST_VGG16.py
    - 04.MNIST_cnn_using_VGG16.py
    - 04.resize_images.py
    - 04.MNIST_shape224.npz
    - 04.MNIST_cnn_model_VGG16.config
    - 04.MNIST_cnn_model_VGG16.weights
    - 02-04.numbers_groundtruth.json
    - 02-04.images/*

## How to use?
 - Just fork to your own GitHub or download this repository directly. And run the programs you interested.

## References
 - "Convolutional Neural Networks" on Coursera.
 - [Keras Documentation](https://keras.io/)
