ConvNet for Digit classification
================================

Author: Chau Bui <chaukbui@gmail.com>

### Introduction ###
In this repository, I implement the first application of deep learning: a 2-layer convolutional neural network that learns to classify images of digit.

The model trains and tests on the standard machine learning benchmark [MNIST](http://yann.lecun.com/exdb/mnist/), which consists of 60K training examples and 10K test examples. At best, the model achieves **99.67%** accuracy.

I don't make it support GPU training. However, on a MacbookPro `2.8 GHz Intel Core i7`, it takes only `1.6 minutes` to train through `60K`, easing the need for using GPUs.

### Installation ###
The implementation uses [Torch](http://torch.ch/), an open source deep learning framework by Facebook. I highly recommend that you follow their setup guide, which works for both Mac OS and Linux:
```shell
# in a terminal, run the commands WITHOUT sudo
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
```

### Usage ###
To train the model and see how the test result improves accordingly
```shell
th convnet.lua
```

### Comments and Bug Reports ###
I'm a newbie and constantly look forward to improving. If you find a spot to improve the codebase, please contact me via my email chaukbui@gmail.com.
