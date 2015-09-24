#!/bin/sh
set -e
set -u

mkdir -p examples/mnist
cd examples/mnist
curl -O "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
curl -O "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
curl -O "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
curl -O "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
gunzip -c train-images-idx3-ubyte.gz > train-images-idx3-ubyte
gunzip -c train-labels-idx1-ubyte.gz > train-labels-idx1-ubyte
gunzip -c t10k-images-idx3-ubyte.gz > t10k-images-idx3-ubyte
gunzip -c t10k-labels-idx1-ubyte.gz > t10k-labels-idx1-ubyte
cd ../..

mkdir -p examples/cifar10
cd examples/cifar10
curl -O "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
tar -xzkf cifar-10-binary.tar.gz
rm -f train.bin
touch train.bin
cat cifar-10-batches-bin/data_batch_1.bin >> train.bin
cat cifar-10-batches-bin/data_batch_2.bin >> train.bin
cat cifar-10-batches-bin/data_batch_3.bin >> train.bin
cat cifar-10-batches-bin/data_batch_4.bin >> train.bin
cat cifar-10-batches-bin/data_batch_5.bin >> train.bin
cp cifar-10-batches-bin/test_batch.bin test.bin
