# to grab embeddings from Pretrained models
# HuBERT

import os
import torchaudio
import torch
import s3prl.hub as hub
from copy2shm import get_audio_filesizes, block_audio_files

dataset_path = '/export/catch1/data/osa-brahms'
audio_path = 'data/002/2019-09-10/audio/2xvrEu_2019-09-10-23-40-39-711_2019-09-10-23-42-39-709_7.wav'
# model_0 = getattr(hub, 'fbank')()  # use classic FBANK
# model_1 = getattr(hub, 'modified_cpc')()  # build the CPC model with pre-trained weights
# model_2 = getattr(hub, 'tera')()  # build the TERA model with pre-trained weights
# model_3 = getattr(hub, 'wav2vec2')()  # build the Wav2Vec 2.0 model with pre-trained weights


model_HuBERT = getattr(hub, 'hubert_base')()  # build the HuBERT model with pre-trained weights

device = 'cuda'  # or cpu
model_HuBERT = model_HuBERT.to(device)
# wavs = [torch.randn(160000, dtype=torch.float).to(device) for _ in range(16)]
# with torch.no_grad():
#     reps = model_HuBERT(wavs)  # list of (time, feat) tensor

#     print(reps['hidden_state_12'].shape)  # (16, time, feat)
# read csv for audio paths for feature extraction
import csv
csv_file = 'audio_files_blocked.csv'
embed_save_path = 'data/embeddings_hubert_base'
os.makedirs(embed_save_path, exist_ok=True)
with open(csv_file, 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter='\t')
    next(csvreader)  # skip header
    for row in csvreader:
        blk_num_str, filepath, size_str = row
        # process each audio file
        waveform, sample_rate = torchaudio.load(filepath)
        waveform = waveform.to(device)
        with torch.no_grad():
            reps = model_HuBERT(waveform)  # get representations
            embedding = reps['hidden_state_12'].squeeze(0).cpu()  # (time, feat)
        # save embedding
        save_path = filepath.replace(dataset_path+'/data', embed_save_path).replace('.wav', '.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(embedding, save_path)
        print(f"Saved embedding for {filepath} to {save_path}")