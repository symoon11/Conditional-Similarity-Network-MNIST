# Conditional-Similarity-Network-MNIST
This is a toy example of Conditional Similarity Networks on MNIST dataset. It is based on a paper named "Conditional Similarity Networks" written by A. Veit, S. Belongie and T. Karaletsos.

## Overview
In this paper, they proposed a network named "Conditional Similarity Network" to measure the similarity between images having various attributes. The network consists of two partial networks. One is Convolutional Network to extract features from an image, and the other is a set of mask, each of which is corresponding to one attribute(color, shape, category or something), and works as an element-wise gating function selecting relevant features of the attributes from a feature vector. The most important thing is that the set of mask is also trainable. That is, the masks learn by themselves what features are actually need to distinguish images with respect to the corresponding attributes. If we apply a mask on the feature vector, then we can measure conditional similarities of images.

## The characteristics of the network
It is based on deep metric learning. Deep metric learning is to train a network that maps similar input data to similar feature vectors. It means that deep metric network embeds input data in high dimensional space into low dimensional space, conserving the metric between the data. The most usual way to implement deep metric leaning is Triplet network. First, we pick a input x, calle an anchor. Then We choose a positive sample x+, which is similar to x(for example, x+ is in the same category with x), and a negative sample x-, which is not similar to x. Now, we construct a 3 parallel networks, each of which has the same weights with the others, and feed a pair of inputs (x, x+, x-) into the networks. Then, we measure the distance between the anchor output and the positive output(d+) and the distance between the anchor output and the negative output(d-). We want d+ to be small and d- to be large. So we use a hinge loss = max(0, (d+)-(d-)+margin) as our objective loss function.

According to the paper, Conditional Similarity Network works better to learn multiple similarities, which shares some features between, than standard Triplet network.

## Question
The network has better performance on learning multiple similarities that correlate each other. In this case, some masks for the similarities are activated on the same indices. Then what if the similarities is unrelated? For example, there is a digit image which has a font color. It has two attributes, digit and color. However there is no relation between two. It means that, in feature vector, some dimensions are representing color attribute, and some dimensions are representing digit attribute, but there is no index that represent both color and digit. I wanted to experimentally show that if I trained the network with unrelated features, the mask of each feature does not share indecies with the other masks.

## Experiment setting
1. Data Set
I made a new data set from MNIST dataset. First, I picked rgb values from [0~200](to avoid a letter to be white) randomly per iamge, and add color on a fixel whose greyscale is nonzero. Finally I got a image whose backgraound is black and digit is colored.
Because my computer is super slow, I just only use 5000 images from MNIST dataset. I assigned 50% of the images to traning set, 30% to validation set, 10% to test set.

2. Network structure
I used Lenet as the encoder of the network, 2 convolutional layers followed by 2 dense layers. Output dimension of the encoder is 20. I used two masks for the attributes, color and digit. So the total dimension of the masks are [20, 2]. I used deep metric learning mentioned above.

3. Training
I used AdamOptimizer as the optimizer of the network. The learning rate was 10^-3. I set the batch size 100. For each minibatch, I picked 100 triplets randomly from training dataset.
