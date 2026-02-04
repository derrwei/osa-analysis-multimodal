#!/bin/bash

model_type="cnnt"
expname="fbank_best_auc_weighted_${model_type}"

for fold in {0..6}
do
    echo "Starting Fold $fold in background..."
    
    # get all saved best models in checkpoints folder and get the highest auc one
    # model_paths=$(ls logs/${expname}/fold${fold}/checkpoints/best-epoch=*.ckpt | sort -V -t '=' -k2)
    # model_paths=$(ls logs/${expname}/fold${fold}/checkpoints/last.ckpt)
    # get the last one (best_epoch=xx-val_auc=xx.ckpt)
    model_paths=$(ls logs/${expname}/fold${fold}/checkpoints/best-epoch=*.ckpt | sort -V -t '=' -k4 | tail -n 1)
    # model_paths=$(ls logs/${expname}/fold${fold}/checkpoints/best-epoch=*.ckpt | sort -V -t '=' -k2 | tail -n 1)
    echo $model_paths
    best_model_path=$(echo "$model_paths" | head -n 1)
    echo "Using model: $best_model_path"
    CUDA_VISIBLE_DEVICES=0 python inference.py --model_type $model_type --exp_name $expname --model_path "$best_model_path" --fold $fold
done

wait
echo "All folds finished!"
