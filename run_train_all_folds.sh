#!/bin/bash

model_type="cnnt"
data_dir="/export/catch2/data"
num_classes=2
epochs=50
expname="fbank_best_auc_weighted"

# export CUDA_VISIBLE_DEVICES=1

for fold in {0..9}
do
    echo "Starting Fold $fold in background..."
    
    # The '&' symbol at the end puts the task in the background
    # python train.py --fold "$fold" &
    CUDA_VISIBLE_DEVICES=1 python main.py --data-dir "/export/catch2/data" --model-type ${model_type} --num-classes $num_classes --fold $fold --epochs 50 --batch-size 128 --lr 0.0001 --early-stop-patience 20 --exp-name ${expname} 
done

wait
echo "All folds finished!"
