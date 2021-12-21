## (Un)Supervised Learning

This repo contains code for NYU AI Fall21 Assignment 4

Using the provided dataset, we are implementing a multi-layer perceptron for classification via supervised learning, as well as the unsupervised k-means and AGNES clustering algorithms.

## Data

The attached csv file contains all the data. The run file handles importing it and converting it to numpy arrays. A description of the dataset is in the run file.

## Algorithms

### Multi-Layer Perceptron (MLP)

The MLP class calls a fully connected layer ("FCLayer") and a Sigmoid layer.The forward pass is for prediction and the backward pass is for doing gradient descent. The backward-pass function takes the previous gradient as input, updates the layer weights (for FCLayer) and returns the gradients for the next layer.

### K-Means:

This used Euclidean distance between the features and a maximum number of iterations, t. We use k-means to split data in k clusters provided to the k_means class.

### AGNES:

This uses the Single-Link method (distance between cluster a and b=distance between closest members of clusters a and b) and the dissimilarity matrix.