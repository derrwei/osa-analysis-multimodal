#! /bin/bash

FOLD=$1
NUM_CLASSES=2
EXP_NAME="audio_only_${NUM_CLASSES}cls"
echo "Experiment Name: ${EXP_NAME}"
OUTPUT_DIR="output_logs"
MODEL="cnn"

# export CUDA_VISIBLE_DEVICES=1

if [ ! -d "${OUTPUT_DIR}" ]; then
    mkdir -p "${OUTPUT_DIR}"
fi

echo "Starting training for fold ${FOLD}..."
nohup python main.py --data-dir "/export/catch2/data" --model-type ${MODEL} --num-classes ${NUM_CLASSES} --fold ${FOLD} --epochs 50 --batch-size 128 --lr 0.005 --num-workers 4 --exp-name "${EXP_NAME}" > ${OUTPUT_DIR}/train_fold_${FOLD}_${EXP_NAME}_${MODEL}.log 2>&1 &
PID=$!
echo "Training for fold ${FOLD} started with PID: ${PID}"
echo "To check progress: tail -f ${OUTPUT_DIR}/train_fold_${FOLD}_${EXP_NAME}_${MODEL}.log"
echo "To check if still running: ps aux | grep main.py"
echo "To kill if needed: kill ${PID}"