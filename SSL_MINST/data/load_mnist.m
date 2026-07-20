function [images, realLabels]=load_mnist
% addpath('./MNIST');

images1 = loadMNISTImages('train-images.idx3-ubyte');
[~, realLabels1] = labelgroups('train-labels.idx1-ubyte');

images2 = loadMNISTImages('t10k-images.idx3-ubyte');
[~, realLabels2] = labelgroups('t10k-labels.idx1-ubyte');

images=[images1, images2];
realLabels=[realLabels1;realLabels2];