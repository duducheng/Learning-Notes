# Notes on Neural Style and Deep Dream
* A Neural Algorithm of Artistic Style ([arXiv](https://arxiv.org/abs/1508.06576))
* Keras Deep Dream ([code](https://github.com/fchollet/keras/blob/master/examples/deep_dream.py))
* Keras Neural Style ([code](https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py))
* CS231n Lecture 9

## How to study
Watch the CS231n lecture or read the slide first, great course. The original paper is not that clear for Machine Learning people, it focus more on explaining why it works, but not that easy to understand. Yet the effort is magic.

## Principle
Deep Dream and Neural Style are basically doing the same thing, trying to find some images to minimize the defined cost.

For Deep Dream, the cost is the some particular layer's (e.g. block5_conv4) output, or activation, other than the regularized cost. It means, we let the backwards delta of this layer equal its activation. This operation will strengthen the neuron's activation; in Deep Dream's output, we can always find many cats and dogs. It's just because the VGG net is trained on the ImageNet dataset, where there are lots of dog's and cat's pictures.

For Neural Style, the cost is the defined content loss and style loss, and use some weights to combine them. The content loss is the L2 loss of 2 terms: some particular layer's output of the content image (the image that you want to transfer, e.g. your portraits) and the optimization target. Note the target is as the same size of the content images, and could be randomly initialized. Then the Style loss is also L2 loss of 2 terms: some "statistics" of layers' output of the optimization target and the style image (the image that you want to copy the style, e.g. the Starry Night). Here, the statistics could be something that doesn't care the spatial information, like Gram Matrix, which is just the outer production of the flattened conv output; the details could be found in the Lecture or the paper.

For Deep Dream and Neural Style, they don't need to iterate on many images, which means less memory needed, so the use of L-BFGS is a natural choice.
