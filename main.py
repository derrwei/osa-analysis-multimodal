#!/usr/bin/env python3
"""Training script for audio-only OSA classification models."""

import argparse
import os
import sys
from pathlib import Path
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning_modules.osa_classification import AudioOnlyLightningModule

def main():
    parser = argparse.ArgumentParser(description='Train audio-only OSA classification model')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='cnnt', choices=['cnnt', 'cnn', 'hubert'], help='Type of model to use')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--feature-type', type=str, default='fbank', choices=['fbank', 'mfcc', 'hubert'], help='Type of input features')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--norm', type=str, default='batch_frequency', choices=['batch', 'batch_frequency', 'layer'], help='Normalization type')
    
    # Data arguments
    parser.add_argument('--fold', type=int, default=0, help='Fold index (0-9)')
    parser.add_argument('--data-dir', type=str, required=True, help='Feature directory')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=6, help='Number of data loader workers')
    parser.add_argument('--devices', type=int, default=1, help='Number of devices/GPUs')
    parser.add_argument('--precision', type=str, default='32', choices=['16', '32', '16-mixed'], help='Training precision')
    
    # Class weighting
    parser.add_argument('--class-weight-method', type=str, choices=['balanced', 'none'], default='balanced', help='Class weighting method')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    
    # Logging and saving
    parser.add_argument('--log-dir', type=str, default='logs/audio_only', help='Logging directory')
    parser.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory (default: log_dir/checkpoints)')
    parser.add_argument('--save-top-k', type=int, default=3, help='Save top k checkpoints')
    parser.add_argument('--monitor-metric', type=str, default='val_auc', help='Metric to monitor for checkpointing')
    parser.add_argument('--early-stop-patience', type=int, default=20, help='Early stopping patience')
    
    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    parser.add_argument('--quiet', dest='verbose', action='store_false', help='Quiet mode')
    parser.add_argument('--exp-name', type=str, default='audio_only', help='Experiment name for logging')
    
    args = parser.parse_args()
    
    # Set seed
    pl.seed_everything(args.seed)
    
    # Create experiment name

    # Setup logging
    logger = TensorBoardLogger(
        save_dir='logs',
        name=args.exp_name+'_'+args.model_type,
        version='fold{}'.format(args.fold)
    )

    
    # Create model
    model = AudioOnlyLightningModule(
        fold_idx=args.fold,
        feature_type=args.feature_type,
        model_type=args.model_type,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
        data_dir=args.data_dir,
        verbose=True
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename='best-{epoch:02d}-{val_auc:.4f}',
            monitor='val_auc',
            mode='max',
            save_top_k=args.save_top_k,
            save_last=True,
            verbose=args.verbose
        ),
        EarlyStopping(
            # monitor='val_loss',
            monitor='val_auc',
            patience=args.early_stop_patience,
            mode='max',
            verbose=args.verbose
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Trainer
    # Use ddp_find_unused_parameters=False for better performance
    strategy = "auto"
    if args.devices > 1:
        strategy = "ddp_find_unused_parameters_false"
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=args.devices,
        precision=args.precision,
        strategy=strategy,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=args.verbose,
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        # num_sanity_val_steps=0,
        deterministic=True,
        # Additional settings for better multi-GPU performance
        sync_batchnorm=args.devices > 1,
        use_distributed_sampler=True
    )

    # Train
    trainer.fit(model)
    # Test best model on test set
    trainer.test(model)


if __name__ == '__main__':
    # script example (fbank,hubert): 
    # python main.py --data-dir "/export/catch2/data" --model-type cnn --num-classes 2 --fold 0 --epochs 100 --batch-size 128 --lr 0.0001 --num-workers 4 --exp-name fbank
    # python main.py --data-dir "/export/catch2/data" --model-type hubert --feature-type hubert --num-classes 2 --fold 0 --epochs 100 --batch-size 32 --lr 0.0001 --num-workers 4 --exp-name hubert
    main()
