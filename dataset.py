from torch.utils import data
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import utilities
from os.path import join
import os
import torch
import math
import h5py
import glob
import gc
import random
import time
import tqdm

label_dict = {
    'n': 0,
    'o': 1,
    'c': 2,
    'm': 3,
    'h': 4,
    'b': 5,
}

class ShardedOsaDatasetH5(data.IterableDataset):
    def __init__(self, hdf5_folder, fold, split='train', shuffle=False):
        self.hdf5_folder = hdf5_folder
        self.split = split
        self.fold = fold
        self.shuffle = shuffle
        self.file_list = self._get_hdf_file()
        self.chunk_size = 1024  # Number of samples to read in each chunk
    
    def _get_hdf_file(self):
        # get all hdf5 files for the fold
        if self.split == 'train':
            folds = [f for f in range(10) if f not in [self.fold, (self.fold + 1) % 10]]
            file_list = []
            for f in folds:
                fold_files = glob.glob(join(self.hdf5_folder, f'fold_{f}_shard_*.hdf5'))
                file_list.extend(fold_files)
        elif self.split == 'val':
            val_fold = (self.fold + 1) % 10
            file_list = glob.glob(join(self.hdf5_folder, f'fold_{val_fold}_shard_*.hdf5'))
        elif self.split == 'test':
            test_fold = self.fold
            file_list = glob.glob(join(self.hdf5_folder, f'fold_{test_fold}_shard_*.hdf5'))
        else:
            raise ValueError("split must be 'train', 'val' or 'test'")
        return file_list

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.file_list)
            
        for shard_path in self.file_list:
            print(f"Opening Shard: {shard_path}")
            
            with h5py.File(shard_path, 'r') as hf:
                # Get the dataset handles (Zero IO happens here)
                dset_features = hf['features']
                dset_labels = hf['labels']
                
                total_samples = dset_features.shape[0]
                
                # Create a list of chunk start indices
                indices = list(range(0, total_samples, self.chunk_size))
                if self.shuffle:
                    random.shuffle(indices) # Shuffle the ORDER of chunks
                
                # --- STREAMING LOOP ---
                for start_idx in indices:
                    end_idx = min(start_idx + self.chunk_size, total_samples)
                    
                    # 1. READ SMALL CHUNK (Fast! ~5-10 seconds)
                    # We only read a slice, not the whole file
                    features_chunk = dset_features[start_idx : end_idx]
                    labels_chunk = dset_labels[start_idx : end_idx]
                    
                    # 2. SHUFFLE INSIDE THE CHUNK
                    if self.shuffle:
                        perm = np.random.permutation(len(features_chunk))
                    else:
                        perm = np.arange(len(features_chunk))
                    features_chunk = features_chunk[perm]
                    labels_chunk = labels_chunk[perm]
                
                # 3. YIELD TO GPU
                for i in range(len(features_chunk)):
                    # print(labels_chunk[i])
                    # str label to digit '0' -> 0
                    yield torch.from_numpy(features_chunk[i]), int(labels_chunk[i].decode('utf-8'))
                
                # Cleanup immediately
                del features_chunk, labels_chunk
                gc.collect()
    


class OsaDatasetH5(data.Dataset):
    def __init__(self, h5_file_path, metadata_csv='/export/catch2/data/osa-brahms/features/hubert_base_samples_labels.csv', fold_id=0, split='train'):
        self.h5_file_path = h5_file_path
        
        # Load metadata ONCE. Keep it lightweight.
        self.df = pd.read_csv(metadata_csv)
        self.fold_df = self.get_fold_df(fold_id=fold_id, split=split)
        self.sample_ids = self.fold_df['SampleID'].tolist()
        self.labels = self.fold_df['Label'].tolist()
        # self.sample_ids = [f"sample_{i}" for i in range(10000)]
        # self.labels = [np.random.randint(0, 2) for _ in range(10000)]  # Dummy binary labels
        # self.transform = transform
        
        # IMPORTANT: Do not open the HDF5 file in __init__.
        # If you open it here, it breaks when PyTorch creates workers.
        self.h5_file = None

    def _open_h5(self):
        # This function runs inside each worker process
        # rdcc_nbytes: Sets the chunk cache size (100MB here). 
        # Helps when reading sequentially or re-reading same data.
        self.h5_file = h5py.File(self.h5_file_path, 'r', rdcc_nbytes=1024*1024*100)


    def get_fold_df(self, fold_id, split):
        'Get dataframe from folds (each fold saved in a txt file named osa_nights_fold_{fold_id}.txt)'
        all_folds = list(range(10))
        test_fold = fold_id
        val_fold = (fold_id + 1) % 10
        if split == 'test':
            selected_folds = [test_fold]
        elif split == 'val':
            selected_folds = [val_fold]
        elif split == 'train':
            selected_folds = [f for f in all_folds if f not in [test_fold, val_fold]]
        else:
            raise ValueError("split must be 'train', 'val' or 'test'")
        nights = []
        for fold in selected_folds:
            fold_file = join('/export/catch2/data/osa-brahms/folds', f'osa_nights_fold_{fold}.txt')
            with open(fold_file, 'r') as f:
                nights_in_fold = [line.strip() for line in f.readlines()]
                nights.extend(nights_in_fold)
        # get dataframe rows from nights, SampleID is in the format of patientID/nightID/segmentID
        # get get all SampleIDs that start with any of the nights in the list
        # df_split = self.df[self.df['Sample_Path'].str.startswith(tuple(nights))]
        df_split = self.df[self.df['SampleID'].apply(lambda x: any(x.startswith(night) for night in nights))]
        return df_split
    def __getitem__(self, index):
        if self.h5_file is None:
            self._open_h5()
            
        # sample_id = self.dataset_indices[index]
        sample_id = self.sample_ids[index]
        
        # Fast Read
        # The [:] syntax copies the data from disk to memory instantly
        feature = self.h5_file[sample_id][:]
        
        # Handle Transforms / Tensor Conversion
        # if self.transform:
        #     feature = self.transform(feature)
        # else:
        #     feature = torch.from_numpy(feature).float()
            
        label = self.labels[index]
        return feature, label

    def __len__(self):
        return len(self.labels)

    # Optional: Clean close (though Python usually handles this)
    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()


def find_resume_point(fp, total_samples):
    """
    Scans the file backwards to find the last written sample.
    Assumes unwritten sections are absolute zeros.
    """
    print("Scanning file to find resume point (this takes a moment)...")
    
    # We check every 1000th sample backwards to be fast
    for i in range(total_samples - 1, -1, -1):
        # If a sample is NOT all zeros, we assume we wrote it.
        # (NOTE: If your actual data contains samples that are naturally ALL zeros, 
        # this logic is risky. But for embeddings/features, it's usually safe.)
        if np.any(fp[i] != 0):
            return i + 1
            
    return 0

class OsaDataset(data.Dataset):
    """Fbank dataset for OSA data samples
    Args:
        folder: Root folder of the dataset
        fold_id: Fold ID (0-9), refer to which fold to be used as test set
        split: 'train', 'val' or 'test' -- 8 folds for training, 1 fold for validation, 1 fold for test
        feature_type: Feature type: fbank, mfcc
    """

    def __init__(self, root_folder, fold_id, split, feature_type, normalise=False, num_classes=2):
        self.num_classes = num_classes
        self.normalise = normalise
        self.feature_type = feature_type
        if feature_type in ['hubert', 'wav2vec2']:
            self.features = join(root_folder, "osa-brahms", "features", "{}_base_samples".format(feature_type))
            self.metadata_file = join(root_folder,"osa-brahms", "features", "{}_base_samples_labels.csv".format(feature_type))
        else:
            self.features = join(root_folder, "osa-brahms", "features", "{}_win50ms_hop20ms_samples".format(feature_type))
            self.metadata_file = join(root_folder,"osa-brahms", "features", "fbank_win50ms_hop20ms_samples_labels.csv")
            # self.folds_folder = '/export/catch2/users/xiaolei/projects/BRAHMS/data/folds'
        self.folds_folder = join(root_folder, "osa-brahms", "folds")
        # self.folds_folder = join(root_folder, "folds")
        self.full_meta_df = pd.read_csv(self.metadata_file)
        # create memmap for all features
        # self.sample_ids = self.full_meta_df['Sample_path'].tolist()
        self.sample_ids = self.full_meta_df['SampleID'].tolist()

        self.sample_ids = [join(self.features, f"{sample_id}.npy") for sample_id in self.sample_ids]
        self.real_indices = self._get_df_from_folds(fold_id, split)
        # print(len(self.real_indices))
        if self.feature_type in ['hubert', 'wav2vec2']:
            self.shape = (len(self.full_meta_df), 1500, 768)  # Example shape, adjust as needed
        else:
            self.shape = (len(self.full_meta_df), 1500, 64)  # Example shape, adjust as needed
        # self.fp = None
        # self._generate_memmap() if not Path(f'all_features_{self.feature_type}.dat').exists() else None
        # len_memmap = Path('all_features.dat').stat().st_size / (4 * 1500 * 64)  # float32 = 4 bytes
        # self._generate_memmap()
        # self.fold_df = self._get_df_from_folds(fold_id, split)
        # fp = np.memmap('all_features.dat', dtype='float32', mode='w+', shape=(len(self.fold_df), *self.shape))
        # # load all features into memmap
        # def load_feature_to_memmap(index):
        #     sample_id = self.sample_ids[index]
        #     feature = np.load(sample_id).astype(np.float32)
        #     fp[index] = feature
        # Parallel(n_jobs=8)(delayed(load_feature_to_memmap)(i) for i in range(len(self.fold_df)))
        # fp.flush()

        # self.dataset_indices = self.fold_df['Sample_Path'].tolist() # sample selection based these indices
        # dataset_idx -> SampleID -> feature file, label
        # self.labels = self.fold_df['Label'].tolist() 
        # self.feature_label = list(zip(self.fold_df['SampleID'], self.fold_df['Label']))
        # self.labels = self.fold_df['Event'].tolist()
        self.labels = self.full_meta_df['Event'].tolist()
        self.labels = [self.convert_label_todigit(label) for label in self.labels]

        self.fold_labels = self.full_meta_df.iloc[self.real_indices]['Event'].tolist()
        self.fold_labels = [self.convert_label_todigit(label) for label in self.fold_labels]
        print(f"Label distribution in {split} set: {pd.Series(self.fold_labels).value_counts().to_dict()}")
        # self.feature_label = list(zip(self.fold_df['SampleID'], self.fold_df['Label']))
        print(f"OsaDataset initialized with {len(self.real_indices)} samples for fold {fold_id} split '{split}'")
        # to digit labels
        # load all data to memory if is train split
        # if split == 'train':
        #     # start caching data
        #     self.data_cache = {}
        #     for idx in range(len(self.fold_df)):
        #         feature, label = self._get_feature_label(idx)
        #         self.data_cache[idx] = (feature, label)
        #     print(f"Data cached in memory for training: {len(self.data_cache)} samples loaded.")
        
    def _generate_memmap(self):
        # create memmap for all features
        if os.path.exists(f'all_features_{self.feature_type}.dat'):
            print(f"Memmap file 'all_features_{self.feature_type}.dat' already exists.")
            return
        mode = 'w+'
        fp = np.memmap(f'all_features_{self.feature_type}.dat', dtype='float32', mode=mode, shape=self.shape)
        start_index = 0
        if mode == 'r+':
            start_index = find_resume_point(fp, len(self.full_meta_df))
            print(f"Resuming from index: {start_index}/{len(self.full_meta_df)}")
        if start_index == len(self.full_meta_df):
            print("Memmap file 'all_features.dat' already complete.")
            return
        # load all features into memmap
        def load_feature_to_memmap(index):
            sample_id = self.sample_ids[index]
            feature = np.load(sample_id).astype(np.float32)
            # print(feature.shape)
            fp[index] = feature
            # sanity check the shape 
            # print(fp[index].shape, feature.shape)
            assert fp[index].shape == feature.shape, f"Shape mismatch at index {index}"
        # Parallel(n_jobs=8)(delayed(load_feature_to_memmap)(i) for i in range(len(self.full_meta_df)))
        for i in tqdm.tqdm(range(start_index, len(self.full_meta_df)), initial=start_index, total=len(self.full_meta_df), desc="Generating memmap"):
            load_feature_to_memmap(i)
        fp.flush()
        del fp
        print("Memmap file 'all_features.dat' created.")
    def _get_df_from_folds(self, fold_id, split):
        'Get dataframe from folds (each fold saved in a txt file named osa_nights_fold_{fold_id}.txt)'
        all_folds = list(range(10))
        test_fold = fold_id
        val_fold = (fold_id + 1) % 10
        if split == 'test':
            selected_folds = [test_fold]
        elif split == 'val':
            selected_folds = [val_fold]
        elif split == 'train':
            selected_folds = [f for f in all_folds if f not in [test_fold, val_fold]]
        else:
            raise ValueError("split must be 'train', 'val' or 'test'")
        nights = []
        for fold in selected_folds:
            fold_file = join(self.folds_folder, f'osa_nights_fold_{fold}.txt')
            with open(fold_file, 'r') as f:
                nights_in_fold = [line.strip() for line in f.readlines()]
                nights.extend(nights_in_fold)
        # get dataframe rows from nights, SampleID is in the format of patientID/nightID/segmentID
        # get get all SampleIDs that start with any of the nights in the list
        # df_split = self.full_meta_df[self.full_meta_df['Sample_path'].str.startswith(tuple(nights))]
        df_split = self.full_meta_df[self.full_meta_df['SampleID'].apply(lambda x: any(x.startswith(night) for night in nights))]
        # to return real indices in the full dataframe
        indices = df_split.index.tolist()
        del df_split
        return indices
        # df_split = self.full_meta_df[self.full_meta_df['SampleID'].apply(lambda x: any(x.startswith(night) for night in nights))]
        # return df_split
    
    def get_class_weights(self):
        # class counts 
        from collections import Counter
        # convert labels to digits
        # digit_labels = [self.convert_label_todigit(label) for label in self.labels]
        # label_counts = Counter(digit_labels)
        label_counts = Counter(self.fold_labels)
        label_counts = dict(sorted(label_counts.items()))
        # print(f"Label distribution in {split} set: {label_counts}")
        total_count = sum(label_counts.values())
        class_weights = {}
        # set weight to be total_count / (num_classes * count)
        for cls, count in label_counts.items():
            class_weights[cls] = total_count / (self.num_classes * count)
        return class_weights
    def __len__(self):
        'Denotes the total number of samples'
        # return len(self.fold_df)
        return len(self.real_indices)
    
    def get_norm_stats(self):
        # compute mean and std over the dataset
        fp = np.memmap('all_features.dat', dtype='float32', mode='r', shape=self.shape)
        mean = np.mean(fp[self.real_indices], axis=(0,1,2))
        std = np.std(fp[self.real_indices], axis=(0,1,2))
        return mean, std
        
    def convert_label_todigit(self, label):
        if self.num_classes == 2:
            if label == 'n':
                return 0
            else:
                return 1
        elif self.num_classes == 3:
        # hypopnea as separate class
            if label == 'n':
                return 0
            elif label == 'h':
                return 1
            else:
                return 2 # apnea, mix, central are similar
        elif self.num_classes == 4:
            if label == 'n':
                return 0
            elif label == 'h':
                return 1
            elif label == 'o' or label == 'c' or label == 'm':
                return 2
            elif label == 'b':
                return 3
        else:
            raise ValueError("num_classes must be 2, 3 or 4")
    def _get_feature_label(self, index):
        # index -> SampleID (first column in dataframe) -> feature file path
        # sample_id = self.fold_df.iloc[index]['SampleID']
        # sample_id, label = self.feature_label[index]
        # if hasattr(self, 'data_cache') and index in self.data_cache:
        #     return self.data_cache[index]
        real_index = self.real_indices[index]
        sample_id = self.sample_ids[real_index]
        label = self.labels[real_index]
        # feature_file = join(self.features, f"{sample_id}.npy")
        # feature = np.load(sample_id)
        # feature = fp[index]
        # if self.fp is None:
        # fp = np.memmap(f'all_features_{self.feature_type}.dat', dtype='float32', mode='r', shape=self.shape)
        # feature = np.array(fp[real_index])
        #
        fp = np.memmap(f'all_features_{self.feature_type}.dat', dtype='float32', mode='r', shape=self.shape)
        feature = np.array(fp[real_index]).astype(np.float32)
        # if self.feature_type in ['hubert', 'wav2vec2']:
        #     feature = np.load(feature_file).astype(np.float32)
            # feature = feature.mean(axis=0)  # (embedding_dim,)
        # feature = torch.from_numpy(feature).float()
        # feature = torch.from_numpy(feature_arr.copy()).float()
        # label = self.fold_df.iloc[index]['Label']
        # convert label to digit
        # label = self.convert_label_todigit(label)
        return feature, label

    def __getitem__(self, index):
        feature, label = self._get_feature_label(index)
        # feature to tensor
        # feature = torch.from_numpy(feature).float()

        return feature, label
    

class BalancedBatchSampler(data.Sampler):
    """
    Yields batches of indices with a fixed number of positives/negatives.
    Works with binary labels in {0,1}.
    """
    def __init__(self, labels, batch_size=32, n_pos=8, replacement=True, seed=0):
        assert 0 < n_pos <= batch_size
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.n_pos = n_pos
        self.n_neg = batch_size - n_pos
        rng = np.random.RandomState(seed)

        self.pos_idx = np.where(self.labels == 1)[0]
        self.neg_idx = np.where(self.labels == 0)[0]
        print(f"Number of positive samples: {len(self.pos_idx)}, Number of negative samples: {len(self.neg_idx)}")
        assert len(self.pos_idx) > 0 and len(self.neg_idx) > 0, "Need both classes."

        self.replacement = replacement
        self.rng = rng

        # Define how many batches in an 'epoch'
        self.num_batches = math.ceil(len(self.labels) / batch_size)
    
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        # Shuffle pools each epoch
        self.rng.shuffle(self.pos_idx)
        self.rng.shuffle(self.neg_idx)
        # print the pos and neg ratio
        print(f"Positive samples: {len(self.pos_idx)}, Negative samples: {len(self.neg_idx)}")

        p_ptr = 0
        n_ptr = 0

        for _ in range(self.num_batches):
            if p_ptr + self.n_pos > len(self.pos_idx):
                if self.replacement:
                    # resample with replacement when we run out
                    pos = self.rng.choice(self.pos_idx, size=self.n_pos, replace=True)
                else:
                    # wrap-around without replacement
                    take = self.pos_idx[p_ptr:]
                    need = self.n_pos - len(take)
                    self.rng.shuffle(self.pos_idx)
                    pos = np.concatenate([take, self.pos_idx[:need]])
                    p_ptr = need
            else:
                pos = self.pos_idx[p_ptr:p_ptr + self.n_pos]
                p_ptr += self.n_pos

            if n_ptr + self.n_neg > len(self.neg_idx):
                if self.replacement:
                    neg = self.rng.choice(self.neg_idx, size=self.n_neg, replace=True)
                else:
                    take = self.neg_idx[n_ptr:]
                    need = self.n_neg - len(take)
                    self.rng.shuffle(self.neg_idx)
                    neg = np.concatenate([take, self.neg_idx[:need]])
                    n_ptr = need
            else:
                neg = self.neg_idx[n_ptr:n_ptr + self.n_neg]
                n_ptr += self.n_neg

            batch = np.concatenate([pos, neg])
            self.rng.shuffle(batch)
            yield batch.tolist()

class BalancedBatchSampler_multiclass(data.Sampler):
    """
    Yields batches of indices with a fixed number of samples per class.
    Works with multi-class labels in {0,1,...,num_classes-1}.
    samples_per_class: number of samples per class in each batch
    """
    def __init__(self, labels, batch_size=32, n_classes=2, replacement=True, samples_class=None, seed=0):
        assert n_classes <= batch_size
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.samples_per_class = [0] * n_classes
        if n_classes == 3:
            self.samples_per_class[0] = batch_size // 3
            self.samples_per_class[1] = batch_size // 3
            self.samples_per_class[2] = batch_size - self.samples_per_class[0] - self.samples_per_class[1]
        else:
            raise NotImplementedError("Only n_classes=3 is implemented in this sampler.")


        rng = np.random.RandomState(seed)
        self.class_indices = {}
        for cls in range(n_classes):
            self.class_indices[cls] = np.where(self.labels == cls)[0]
            assert len(self.class_indices[cls]) > 0, f"Need samples for class {cls}."
        self.replacement = replacement
        self.rng = rng
        # Define how many batches in an 'epoch'
        self.num_batches = math.ceil(len(self.labels) / batch_size)
    def __len__(self):
        return self.num_batches
    def __iter__(self):
        # Shuffle pools each epoch
        for cls in range(self.n_classes):
            self.rng.shuffle(self.class_indices[cls])

        class_pointers = {cls: 0 for cls in range(self.n_classes)}

        for _ in range(self.num_batches):
            batch = []
            for cls in range(self.n_classes):
                n_samples = self.samples_per_class[cls]
                cls_idx = self.class_indices[cls]
                ptr = class_pointers[cls]

                if ptr + n_samples > len(cls_idx):
                    if self.replacement:
                        samples = self.rng.choice(cls_idx, size=n_samples, replace=True)
                    else:
                        take = cls_idx[ptr:]
                        need = n_samples - len(take)
                        self.rng.shuffle(cls_idx)
                        samples = np.concatenate([take, cls_idx[:need]])
                        class_pointers[cls] = need
                else:
                    samples = cls_idx[ptr:ptr + n_samples]
                    class_pointers[cls] += n_samples

                batch.extend(samples)
            self.rng.shuffle(batch)
            yield batch


if __name__ == "__main__":
    train_dataset = OsaDataset(root_folder='/export/catch2/data', fold_id=0, split='train', feature_type='fbank', num_classes=2)
    
