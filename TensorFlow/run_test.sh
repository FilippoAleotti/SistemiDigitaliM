#!/bin/bash

#test_folder="/home/filippo/Scrivania/MNIST/test"
test_folder="/media/faleotti/SSD/MNIST/test"
model="trained_models/model-18000"
network='lenet'

# run the training!
python test.py --testing_folder $test_folder \
                --model $model \
                --network $network