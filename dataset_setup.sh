#!/bin/bash

if [ ! -d "data" ]; then
  mkdir data
fi
cd data
if [ ! -d "cifar-10" ]; then
  mkdir cifar-10
fi
if [ ! -d "cifar-100" ]; then
  mkdir cifar-100
fi
cd cifar-10
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar zxvf cifar-10-binary.tar.gz
cp -r cifar-10-batches-bin/* ./
rm -r cifar-10-batches-bin/
rm cifar-10-binary.tar.gz
cd ../cifar-100
wget https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
tar zxvf cifar-100-binary.tar.gz
cp -r cifar-100-binary/* ./
rm -r cifar-100-binary/
rm cifar-100-binary.tar.gz


