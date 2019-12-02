#!/bin/bash

#mnist="/home/filippo/Scrivania/MNIST"
mnist="/media/faleotti/SSD/MNIST"
step=18000
batch=128
val_step=2000
network='lenet'
train_mode=${1:-"1"}

if [ $train_mode == "1"  ]
then
    echo "placeholder train validation mode"
    python train.py --training_folder "$mnist/train" \
                    --validation_folder "$mnist/validation" \
                    --step $step \
                    --batch $batch \
                    --validation_step $val_step \
                    --network $network
elif [ $train_mode == "2"  ]
then
    echo "placeholder train mode"
    python train.py --training_folder "$mnist/full_train" \
                    --step $step \
                    --batch $batch \
                    --network $network \
                    --validation_step -1 \
                    --validation_folder "$mnist/validation"

elif [ $train_mode == "3"  ]
then
    echo "tf-data mode"
    python train_dataloader.py --training_folder "$mnist/full_train" \
                    --step $step \
                    --batch $batch \
                    --network $network \
                    --validation_step -1 \
                    --validation_folder "$mnist/validation"
else
    echo "train_mode is not valid"
fi