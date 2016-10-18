# [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)
This model could inspire a lot on Sequence / Time Series modeling.
* Conv ops could be faster when training, compared to RNNs.
* Skip connections could be useful for deep nets, which makes the training faster.
* Activation could be multiplied, for tanh and sigmoid, to make a "gated" mechanism.
* Atrous Conv means "A trous", or conv with holes. In the paper, it's called "dilated conv". It makes the model have larger receptive fields, with less computing cost. Theano has a great explanation [page](http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic). The following fig shows some intuition, but for 2D deconv.

![Atrous Conv](http://deeplearning.net/software/theano_versions/dev/_images/padding_strides_transposed.gif)
