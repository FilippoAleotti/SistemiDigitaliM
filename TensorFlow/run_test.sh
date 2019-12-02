#!/bin/bash

test_folder="/media/faleotti/SSD/MNIST/test"
model="trained_models/model-18000"
network='lenet'

python test.py --testing_folder $test_folder \
                --model $model \
                --network $network