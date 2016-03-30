#!/bin/sh
set -eu

mkdir -p mnist
cd mnist
curl -O "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
curl -O "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
curl -O "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
curl -O "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
gunzip -c train-images-idx3-ubyte.gz > train-images-idx3-ubyte
gunzip -c train-labels-idx1-ubyte.gz > train-labels-idx1-ubyte
gunzip -c t10k-images-idx3-ubyte.gz > t10k-images-idx3-ubyte
gunzip -c t10k-labels-idx1-ubyte.gz > t10k-labels-idx1-ubyte
