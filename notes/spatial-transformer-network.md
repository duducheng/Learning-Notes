# [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025)
* [The power of Spatial Transformer Networks](http://torch.ch/blog/2015/09/07/spatial_transformers.html)
* [TensorFlow official implementation](https://github.com/tensorflow/models/tree/master/transformer)

## Summary
This is one of the papers I like most. DeepMind is great!

Spatial Transformer Network (STN) is a magic layer (or network) that can be put normally in the Deep Networks, trained by backprop, and extended the ability of original network by attention mechanism. With STN, the model can focus on some particular region of the input image, adding other spatial transformation, to make the task for the original networks.

The following image shows the magic: after some epochs of training, the output of STN will focus on some better region.

![Magic show](https://raw.githubusercontent.com/moodstocks/gtsrb.torch/master/resources/epoch_evolution.gif)

## Structure
![STN Sturc](https://raw.githubusercontent.com/moodstocks/gtsrb.torch/master/resources/spatial-transformer-structure.png)

There are 3 parts in STNs:
* Localization network: input the original images, and output the parameters of the transformation (e.g. affine)
* Grid generator: make the transformation
* Sampler: sampling (filtering) the grid given by the Grid generator

## Note
* All of them are differentiable, make it possible to be trained by standard backprop.

* You can even use multiple STNs, to let you network focus many parts on the original images; this is really useful for Fine-grained classification.

* In my test, for my classification task, a STN make ResNet-50 as powerful as ResNet-152 without STN.
