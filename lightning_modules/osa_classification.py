"""Lightning module for audio-only OSA classification models.

This module handles audio-only (mel-spectrogram) input for OSA classification,
without requiring effort signals. It's simpler than the multimodal approach
and serves as a strong baseline.
"""
from __future__ import annotations
from typing import Optional, Dict, Any
from pathlib import Path
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from sklearn.metrics import confusion_matrix, roc_auc_score
import random, math

import sys
sys.path.append('../models')
sys.path.append('../')
from dataset import OsaDataset, BalancedBatchSampler, BalancedBatchSampler_multiclass
from models.cnns import CNN_OSA, CNN_OSA_Large
from models.cnnt import CnnT
from models.unsupervised import HubertClassificationHead
from utilities import calculate_scalar_torch, scale_torch, Facalloss
from torchaudio.transforms import SpecAugment, TimeMasking, FrequencyMasking, TimeStretch


class BucketedRandomSampler(torch.utils.data.Sampler):
    """Shuffle indices but yield them in bucketed order to maximize locality.
       group_size: number of samples likely sharing the same shard/file."""
    def __init__(self, data_source, group_size=256, generator=None):
        self.data_source = data_source
        self.group_size = group_size
        self.generator = generator

    def __iter__(self):
        n = len(self.data_source)
        idx = list(range(n))
        g = self.generator or torch.Generator()
        random.shuffle(idx)  # or torch.randperm if you prefer
        # group into buckets, then shuffle buckets only
        buckets = [idx[i:i+self.group_size] for i in range(0, n, self.group_size)]
        random.shuffle(buckets)
        for b in buckets:
            # keep local order (or lightly shuffle inside bucket)
            yield from b

    def __len__(self):
        return len(self.data_source)


class AudioOnlyLightningModule(LightningModule):
    """Lightning module for audio-only OSA classification."""
    def __init__(
        self,
        fold_idx: int = 0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        num_classes: int = 2,
        batch_size: int = 32,
        num_workers: int = 4,
        threshold: float = 0.5,
        feature_type: str = 'fbank',
        data_dir: str = "/export/raid1/users/xiaolei/tools/osa-analysis/data_xiaolei/osa-brahms",
        model_type: str = 'cnnt',
        verbose: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        # self.model = AudioOnlyOSA(
        #     dropout=dropout_rate, 
        #     norm=norm, 
        # )
        if model_type == 'cnn':
            # self.model = CNN_OSA_Large(
            #     # dropout=0.3,
            #     # norm='batch',
            #     # num_classes=num_classes,
            #     # data_batch_norm=False
            # )
            self.model = CNN_OSA(
                dropout=0.4,
                norm='batch',
                num_classes=num_classes,
                data_batch_norm=True
            )
        elif model_type == 'cnnt':
            self.model = CnnT(
                classes_num=num_classes,
                batchnormal=True,
                dropout=True
            )
        elif model_type == 'hubert':
            # simple mean pooling + linear layer
            self.model = HubertClassificationHead(
                embedding_dim=768,
                num_classes=num_classes
            )

        print("model type:", model_type)
        # Loss function
        self.mean, self.std = None, None
        self.num_classes = num_classes
        # self.loss_cls = nn.CrossEntropyLoss(weight=torch.tensor([0.67, 2.0]) if num_classes==2 else None)
        self.loss_cls = nn.CrossEntropyLoss()
        # self.loss_cls = Facalloss(gamma=2.0, reduction='mean')
        
        # Datasets placeholders
        self._train_dataset: Optional[OsaDatasetH5] = None
        self._val_dataset: Optional[OsaDatasetH5] = None
        self._test_dataset: Optional[OsaDatasetH5] = None
        
        # Buffers for classification metrics
        self.all_preds = []
        self.all_labels = []
        
        # Hyperparameters
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.verbose = verbose
        self.fold_idx = fold_idx
        self.feature_type = feature_type
        
        # Data paths
        self.data_dir = data_dir

        # self.augmentor = Spec(n_freq_masks=2, n_time_masks=2, freq_mask_param=40, time_mask_param=8, iid_masks=True)
        # self.time_masking = TimeMasking(time_mask_param=16, iid_masks=True)
        # self.freq_masking = FrequencyMasking(freq_mask_param=50, iid_masks=True)
        # self.stretch = TimeStretch(hop_length=320, n_freq=64)
        self.aug = nn.Sequential(
            # TimeStretch(hop_length=320, n_freq=64, fixed_rate=0.8),
            FrequencyMasking(freq_mask_param=8, iid_masks=True),
            FrequencyMasking(freq_mask_param=8, iid_masks=True),
            # FrequencyMasking(freq_mask_param=50, iid_masks=True),
            # FrequencyMasking(freq_mask_param=50, iid_masks=True),
            TimeMasking(time_mask_param=200, iid_masks=True),
            TimeMasking(time_mask_param=200, iid_masks=True),
            # TimeMasking(time_mask_param=100, iid_masks=True),
            
        )

    # ---------------- Dataset -----------------
    def setup(self, stage: Optional[str] = None):
        if self._train_dataset is not None:
            return
            
        data_path = self.data_dir
            
        # For distributed training, only log on rank 0 to avoid spam
        verbose_for_setup = self.verbose and (not hasattr(self.trainer, 'global_rank') or self.trainer.global_rank == 0)
        
        if verbose_for_setup:
            print(f"[AudioOnly] Setting up datasets for fold {self.fold_idx} from {data_path}")
            
        # Use faster dataset loading for distributed training
        # Don't build full index during distributed training to avoid hanging
        # build_index = not (hasattr(self.trainer, 'num_devices') and self.trainer.num_devices > 1)
        # Datasets
        print("Building train dataset...")
        self._train_dataset = OsaDataset(
            root_folder=data_path,
            fold_id=self.fold_idx,
            split='train',
            feature_type=self.feature_type,
            normalise=False,
            num_classes=self.num_classes
        )

        loader = torch.utils.data.DataLoader(
            self._train_dataset, 
            batch_size=1024, 
            shuffle=False,
            num_workers=6,
            pin_memory=True,
        )
        # print("Calculating train set mean and std...")
        # all_means = []
        # all_stds = []
        # for batch in loader:
        #     mels, labels = batch
        #     batch_mean, batch_std = mels.mean(), mels.std()
        #     all_means.append(batch_mean)
        #     all_stds.append(batch_std)
        # overall_mean = torch.stack(all_means).mean()
        # overall_std = torch.stack(all_stds).mean()
        # print(f"Overall train set mean: {overall_mean}, std: {overall_std}")
        # self.mean = overall_mean.item()
        # self.std = overall_std.item()
        # # save mean and std to checkpoint later during training
        # self.save_hyperparameters({'mean': self.mean, 'std': self.std})
        # del loader, all_means, all_stds, overall_mean, overall_std

        loss_weights = self._train_dataset.get_class_weights()
        loss_weights = [loss_weights[i] for i in range(self.num_classes)]
        loss_weights[1]*=0.75 #regulate weight for positive class
        print(f"Class weights for loss: {loss_weights}")
        if loss_weights is not None:
            self.loss_cls = nn.CrossEntropyLoss(weight=torch.tensor(loss_weights).to(self.device))
        if self.num_classes == 2:
            self.train_sampler = BalancedBatchSampler(
                labels=self._train_dataset.fold_labels,
                batch_size=self.batch_size,
                n_pos=int(self.batch_size /2.8),
                replacement=True,
                seed=42
            )
        else:
            self.train_sampler = BalancedBatchSampler_multiclass(
                labels=self._train_dataset.fold_labels,
                # labels=self._train_dataset.labels,
                batch_size=self.batch_size,
                n_classes=self.num_classes,
                # n_pos=self.batch_size // 4,
                replacement=True,
                seed=42
            )
        print("Building val dataset...")

        self._val_dataset = OsaDataset(
            root_folder=data_path,
            fold_id=self.fold_idx,
            split='val',
            feature_type=self.feature_type,
            normalise=False,
            num_classes=self.num_classes
        )
        self._test_dataset = OsaDataset(
            root_folder=data_path,
            fold_id=self.fold_idx,
            split='test',
            feature_type=self.feature_type,
            normalise=False,
            num_classes=self.num_classes
        )

    # --------------- Dataloaders ---------------
    def train_dataloader(self):
        if self._train_dataset is None:
            return None
        
        # For multi-GPU training, reduce num_workers per process
        num_workers = self.num_workers
        if hasattr(self.trainer, 'num_devices') and self.trainer.num_devices > 1:
            num_workers = max(1, self.num_workers // self.trainer.num_devices)
        train_loader = torch.utils.data.DataLoader(
            self._train_dataset, 
            batch_size=self.batch_size,
            # sampler=self.train_sampler,
            # batch_sampler=self.train_sampler,
            num_workers=6,
            shuffle=True,
            prefetch_factor=6,
            # persistent_workers=True,
            pin_memory=True, 
            # drop_last=True,
        )

        return train_loader

    def val_dataloader(self):
        if self._val_dataset is None:
            return None
            
        # For multi-GPU training, reduce num_workers per process
        num_workers = self.num_workers
        if hasattr(self.trainer, 'num_devices') and self.trainer.num_devices > 1:
            num_workers = max(1, self.num_workers // self.trainer.num_devices)
            
        return torch.utils.data.DataLoader(
            self._val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=3,
            pin_memory=True,
            # persistent_workers=num_workers > 0
        )

    def test_dataloader(self):
        if self._test_dataset is None:
            return None
            
        # For multi-GPU training, reduce num_workers per process
        num_workers = self.num_workers
        if hasattr(self.trainer, 'num_devices') and self.trainer.num_devices > 1:
            num_workers = max(1, self.num_workers // self.trainer.num_devices)
        
        return torch.utils.data.DataLoader(
            self._test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=3,
            pin_memory=True,
            # prefetch_factor=2,
            # persistent_workers=True
        )

    # --------------- Forward -------------------
    def forward(self, x: torch.Tensor):
        return self.model(x)

    def _shared_step(self, batch, stage: str) -> Dict[str, Any]:
        mels, labels = batch  # shapes: (B,1500,64), (B,960), (B,)        
        # mels = (mels - mels.mean(dim=(1,2), keepdim=True)) / (mels.std(dim=(1,2), keepdim=True) + torch.finfo().eps)
        # mels = (mels - mels.mean()) / (mels.std() + torch.finfo().eps)
        # mels = (mels - self.mean) / (self.std + torch.finfo().eps)
        # mels = mels - mels.mean(dim=1, keepdim=True)
        # mels = (mels - mels.mean(dim=1, keepdim=True)) / (mels.std(dim=1, keepdim=True) + torch.finfo().eps)
        if stage == 'train':
            # mels = self.freq_masking(mels)
            mels = mels.transpose(1,2)  # (B,64,1500)
            mels = self.aug(mels) # to apply specaug after normalisation, zero masking is used refering to mean value of normalised mel
            mels = mels.transpose(1,2)  # (B,1500,64)
        
        if self.global_step < 10 and stage == 'train' and self.verbose:
            unique, counts = torch.unique(labels, return_counts=True)
            label_dist = dict(zip(unique.tolist(), counts.tolist()))
            print(f"\n==============================")
            print(f"[AudioOnly][Train] Step {self.global_step} Label distribution in batch: {label_dist}")
        
        # Labels
        # labels = labels.float().view(-1, 1)
        # normalise mel
        # means, stds = calculate_scalar_torch(mels)
        # mels = scale_torch(mels, means, stds)
        # del means, stds
        # if self.feature_type == 'fbank':
        # elif self.feature_type in ['hubert', 'wav2vec2']:
        #     # print(mels.shape)
        #     mels = mels.mean(dim=1)  # (B, embedding_dim)
        # mels = (mels - mels.mean()) / (mels.std() + torch.finfo().eps)
        # mels = mels - mels.mean(dim=1, keepdim=True) # remove freq bias

        # Forward pass
        # if mels.dim() == 3:
        #     mels = mels.unsqueeze(1)  # (B, 1, 1500, 64)
        logits = self.model(mels)
        loss_cls = self.loss_cls(logits, labels)
        
        # del mels
        # Predictions
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        # Store for metrics calculation
        if stage in ['val', 'test']:
            self.all_preds.append(preds)
            self.all_labels.append(labels.long())
        
        # Logging
        self.log(f"{stage}_loss", loss_cls, on_step=(stage=="train"), on_epoch=True, prog_bar=True, sync_dist=True)
        
        # pos_rate = probs.ge(self.threshold).float().mean()
        # self.log(f"{stage}_pos_rate", pos_rate, on_epoch=True, sync_dist=True)
        
        return {
            "loss": loss_cls, 
            "probs": probs, 
            "labels": labels, 
            "preds": preds
        }

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, 'test')

    # ------------- Epoch End Metrics -----------
    def on_validation_epoch_start(self):
        self.all_preds, self.all_labels = [], []

    def on_validation_epoch_end(self):
        if len(self.all_preds) == 0:
            return
            
        if self.trainer.is_global_zero:
            preds = torch.cat(self.all_preds, dim=0).view(-1).long()
            labels = torch.cat(self.all_labels, dim=0).view(-1).long()
            # Compute confusion matrix
            cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels=list(range(self.num_classes)))
            print("\n==============================")
            print("Confusion Matrix for all classes:")  
            print(cm)
            print("==============================\n")
            # Calculate metrics
            # convert all labels not 0 to be 1
            labels = (labels != 0).long()
            preds = (preds != 0).long()
            tp = ((preds==1) & (labels==1)).sum()
            fp = ((preds==1) & (labels==0)).sum()
            tn = ((preds==0) & (labels==0)).sum()
            fn = ((preds==0) & (labels==1)).sum()
            cm = torch.tensor([[tn, fp], [fn, tp]])
            
            acc = (preds==labels).float().mean()
            precision = tp.float() / (tp + fp + 1e-8)
            recall = tp.float() / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            auc = roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
            
            # Log metrics
            self.log('val_acc', acc, prog_bar=True)
            self.log('val_precision', precision)
            self.log('val_recall', recall)
            self.log('val_f1', f1, prog_bar=True)
            self.log('val_auc', auc, prog_bar=True)
            
            if self.verbose:
                print("\n==============================")
                print(f'[AudioOnly][Val] Acc {acc:.4f} P {precision:.4f} R {recall:.4f} F1 {f1:.4f}')
                print("confusion matrix:")
                print(cm)
                print("==============================\n")
        self.all_preds, self.all_labels = [], []

    # def on_test_epoch_start(self):
    #     self.all_preds, self.all_labels = [], []

    def on_test_epoch_end(self):
        if len(self.all_preds) == 0:
            return
            
        if self.trainer.is_global_zero:
            preds = torch.cat(self.all_preds, dim=0).view(-1).long()
            labels = torch.cat(self.all_labels, dim=0).view(-1).long()
            cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels=list(range(self.num_classes)))
            print("\n==============================")
            print("Confusion Matrix for all classes:")  
            print(cm)
            print("==============================\n")
            # convert all labels not 0 to be 1
            labels = (labels != 0).long()
            preds = (preds != 0).long()
            # Calculate metrics
            tp = ((preds==1) & (labels==1)).sum()
            fp = ((preds==1) & (labels==0)).sum()
            tn = ((preds==0) & (labels==0)).sum()
            fn = ((preds==0) & (labels==1)).sum()
            
            acc = (preds==labels).float().mean()
            precision = tp.float() / (tp + fp + 1e-8)
            recall = tp.float() / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            auc = roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
            
            # Log metrics
            self.log('test_acc', acc, prog_bar=True)
            self.log('test_precision', precision)
            self.log('test_recall', recall)
            self.log('test_f1', f1, prog_bar=True)
            self.log('test_auc', auc, prog_bar=True)
            if self.verbose:
                print(f'[AudioOnly][Test] Acc {acc:.4f} P {precision:.4f} R {recall:.4f} F1 {f1:.4f}')
                
        self.all_preds, self.all_labels = [], []

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # For multi-GPU training, return optimizer only to avoid scheduler sync issues
        if hasattr(self.trainer, 'num_devices') and self.trainer.num_devices > 1:
            return optimizer
        
        # Optional learning rate scheduler for single GPU
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 
        #     mode='min', 
        #     factor=0.5, 
        #     patience=5
        # )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )    
        return [optimizer], [scheduler]

if __name__ == '__main__':
    # Simple smoke test
    mod = AudioOnlyLightningModule(verbose=False)
    mel = torch.randn(2, 1500, 64)
    effort = torch.randn(2, 960)
    labels = torch.randint(0, 2, (2,))
    out = mod._shared_step((mel, effort, labels), 'train')
    print('AudioOnly loss:', float(out['loss']))
    print('AudioOnly smoke test OK.')
