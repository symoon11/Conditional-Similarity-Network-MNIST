## Conditional-Similarity-Network-MNIST
This is a toy example of Conditional Similarity Networks on MNIST dataset. It is based on a paper named "Conditional Similarity Networks" written by A. Veit, S. Belongie and T. Karaletsos.

## Overview
In this paper, they proposed a network named "Conditional Similarity Network" to measure the similarity between images having various attributes. The network consists of two partial networks. One is Convolutional Network to extract features from an image, and the other is a set of mask, each of which is corresponding to one attribute(color, shape, category or something), and works as an element-wise gating function selecting relevant features of the attributes from a feature vector. The most important thing is that the set of mask is also trainable. That is, the masks learn by themselves what features are actually need to distinguish images with respect to the corresponding attributes.
