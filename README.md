# TF2_CNN_modules
some useful CNN blocks and modules implemented in tensorflow2 as a layer class,\
which means you can use it like any other tf.keras.layers layer in **Sequential or Functional** api!\
checkout demo that uses fewer than 30 lines of code to build a resnet.

currently implemented:

SE module\
Squeeze-and-Excitation Networks\
https://arxiv.org/abs/1709.01507
    

ResNet-d version of Skip Connection\
Bag of Tricks for Image Classification with Convolutional Neural Networks\
https://arxiv.org/abs/1812.01187


ECA module\
ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks\
https://arxiv.org/abs/1910.03151


Ghost Convolution\
GhostNet: More Features from Cheap Operations\
https://arxiv.org/abs/1911.11907


ConvBatchNorm\
a combination of conv2d + batchnorm + activation for easier use


MBConv Block (or inverted residual)\
MobileNetV2: Inverted Residuals and Linear Bottlenecks\
https://arxiv.org/abs/1801.04381


Bottleneck and classic Residual Block\
Identity Mappings in Deep Residual Networks\
https://arxiv.org/abs/1603.05027
