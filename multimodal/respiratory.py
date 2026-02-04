##########################################################################
# Xiaolei Xu, 2026
# This package is for reading, processing, and analyzing respiratory data:
# - effort, flow, and snore signals
##########################################################################

import numpy as np
import json
import torch
import scipy
import pandas as pd
import os
import glob
import librosa

class RespiratoryAnalyser:
    # signal configuration
    def __init__(self, sig_config):
        self.dataroot = sig_config.dataroot
        self.channel = sig_config.name
        # dataset structure: dataroot/subject_id/night_id/hst_analysed/channel
        self.fs = sig_config.sampling_rate
        self.audio_fs = 16000 # all the audio files are resampled to 16kHz

        self.plot_folder = 'plots'
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)
    
    def _get_psg_folder(self, night_id):
        # format: dataroot/night_id/hst_analysed/channel.dat
        # night_id 001/2019-09-01
        return f"{self.dataroot}/data/{night_id}/hst_synched/{self.channel}"

    def _get_audio_folder(self, night_id):
        # format: dataroot/night_id/hst_analysed/channel.wav
        return f"{self.dataroot}/data/{night_id}/audio"

    def get_signal(self, night_id):
        sig_folder = self._get_psg_folder(night_id)
        audio_folder = self._get_audio_folder(night_id)
        # data saved as csv in one column splitted as non-equal length segments by -1
        # get all the csv files in the folder and concatenate them
        csv_files = glob.glob(os.path.join(sig_folder, "*.csv"))
        sig_list = []
        audio_lengths = 0
        for csv_file in csv_files:
            sig = pd.read_csv(csv_file, header=None).values.flatten()
            sig = np.array(sig)
            audio_path = csv_file.replace(self.channel,'').replace("hst_synched", "audio").replace(".csv", ".wav")
            audio_length = librosa.get_duration(path=audio_path, sr=self.audio_fs)
            audio_lengths += audio_length
            sig_list.append(sig)
        sig = np.concatenate(sig_list)
        
        # trim or pad the signal to match the audio length
        target_length = int(audio_lengths * self.fs)
        if len(sig) > target_length:
            sig = sig[:target_length]
        elif len(sig) < target_length:
            pad_length = target_length - len(sig)
            sig = np.pad(sig, (0, pad_length), 'constant', constant_values=(0, 0))
        return sig

    def plot_signal(self, signal, title="Respiratory Signal", xlabel="Time (s)", ylabel="Amplitude"):
        import matplotlib.pyplot as plt
        time = np.arange(len(signal)) / self.fs
        plt.figure(figsize=(12, 4))
        plt.plot(time, signal)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.savefig(self.plot_folder + "/" +
            f"{title.replace(' ', '_').lower()}.png")



