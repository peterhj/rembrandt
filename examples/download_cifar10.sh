#!/bin/sh
set -eu

mkdir -p cifar10
cd cifar10
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
