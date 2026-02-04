########################################################
# Copyright (c) 2018 Deep Sound
#
# Ning Ma
# 18 Sep 2018
########################################################

import os
from os.path import join as pj
import numpy as np
import pickle
from scipy.fftpack import dct

def rms(x):
    '''Returns the RMS amplitude of signal
    '''
    return np.sqrt(np.mean(x**2))

def rms_normalise(x, target_rms = 0.1):
    '''Normalise waveform to a target RMS
    Args:
        x: a numpy array containing waveform signal [nsamples, nchannels]
        target_rms: target RMS level (default 0.1)
    Returns:
        normalised signal
    '''
    x_rms = np.mean(np.sqrt(np.mean(x**2, axis=0)))
    return x * target_rms / x_rms

def next_power_of_two(n):
    return 1<<(int(n)-1).bit_length()

def power_to_db(x):
    '''Convert power to dB
    '''
    return 10 * np.log10(np.clip(np.array(x), 1e-10, None))

def amp_to_db(x):
    '''Convert amplitude to dB
    '''
    return 20 * np.log10(np.clip(np.array(x), 1e-10, None))

def hz_to_mel(hz):
    '''Convert Frequency in Hertz to Mels
    Args:
        hz: A value in Hertz. This can also be a numpy array.
    Returns
        A value in Mels.
    '''
    return 2595 * np.log10(1 + hz / 700.0)

def mel_to_hz(mel):
    '''Convert a value in Mels to Hertz
    Args:
        mel: A value in Mels. This can also be a numpy array.
    Returns
        A value in Hertz.
    '''
    return 700 * (10 ** (mel / 2595.0) - 1)

def mel_freqs(low_freq, high_freq, n_filters):
    '''Make an array of frequencies based on the Mel scale
    Args:
        low_freq: lowest frequeny (Hz)
        high_freq: highest frequency (Hz)
        n_filters: number of filters
    Returns:
        A numpy array of frequeny values
    '''
    return mel_to_hz(np.linspace(hz_to_mel(low_freq), hz_to_mel(high_freq), n_filters+2))
    
def erb(x):
    return 24.7*(4.37e-3*x+1)

def erb_to_hz(x):
    return (pow(10,x/21.4)-1)/4.37e-3

def hz_to_erb(x):
    return 21.4*np.log10(4.37e-3*x+1)

def erb_freqs(lowcf, highcf, n_filters):
    '''Make an array of centre frequencies based on the ERB scale
    Args:
        lowcf: lowest centre frequeny (Hz)
        highcf: highest centre frequency (Hz)
        n_filters: number of filters
    Returns:
        A numpy array of frequeny values
    '''
    return erb_to_hz(np.linspace(hz_to_erb(lowcf), hz_to_erb(highcf), n_filters))

def preemphasis(sig, preemph=0.97):
    return np.append(sig[0], sig[1:] - preemph * sig[:-1])

def frame_signal(sig, fs, frame_length, frame_shift, winfunc=np.hanning):
    '''Frame a signal and apply window
    Args:
        sig: signal array
        fs: sampling rate in Hz (default 16000)
        frame_length: frame length in seconds
        frame_shift: frame shift in seconds
        winfunc: window function
    Returns
        frames: (n_frames x frame_length)
    '''
    siglen = len(sig)

    frame_length = int(frame_length*fs)
    frame_shift = int(frame_shift*fs)

    # Compute number of frames and padding
    n_frames = int(np.ceil(float(siglen - frame_length) / frame_shift))

    pad_siglen = int(n_frames * frame_shift + frame_length)
    pad_zeros = np.zeros((pad_siglen - siglen))
    pad_signal = np.append(sig, pad_zeros)

    indices = np.tile(np.arange(0, frame_length), (n_frames, 1)) + np.tile(np.arange(0, n_frames * frame_shift, frame_shift), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Apply window
    frames *= winfunc(frame_length)
    return frames

def mel_filters(nfft, low_freq, high_freq, n_filters, fs):
    '''Compute Mel filters
    Args:
        nfft: number of FFT bins
        low_freq: lowest frequency (Hz)
        high_freq: highest frequency (Hz)
        n_filters: number of filters
        fs: sampling rate (Hz)
    Returns
        fb: Mel filterbank
    '''
    fft_bins = (nfft + 1.0) * mel_freqs(low_freq, high_freq, n_filters) // fs # FFT bin indice for the filters
    fb = np.zeros((nfft // 2 + 1, n_filters))
    for m in range(0, n_filters):
        for k in range(int(fft_bins[m]), int(fft_bins[m + 1])):
            fb[k, m] = (k - fft_bins[m]) / (fft_bins[m+1] - fft_bins[m])
        for k in range(int(fft_bins[m + 1]), int(fft_bins[m + 2])):
            fb[k, m] = (fft_bins[m + 2] - k) / (fft_bins[m + 2] - fft_bins[m+1])
    return fb

def compute_powspec(sig, fs=16000, nfft=512, frame_length=0.025, frame_shift=0.01, winfunc=np.hanning, preemph=0.97):
    '''Compute power spectrogram
    Args:
        sig: signal array
        fs: sampling rate in Hz (default 16000)
        nfft: length of the FFT window (default 512)
        frame_length: frame length in seconds (default 0.025 s, 400 samples)
        frame_shift: frame shift in seconds (default 0.01 s, 160 samples)
        winfunc: window function (default numpy.hanning)
        preemph: pre-emphasis factor (default 0.97)
    Returns
        powspec: power spectrogram
    '''
    # Pre-emphasis
    sig = preemphasis(sig, preemph)
    
    # Compute number of frames and padding
    frames = frame_signal(sig, fs, frame_length, frame_shift, winfunc)

    # Find the next power of two
    # nfft = next_power_of_two(frame_length*fs)

    # Magnitude spectrogram
    magspec = np.absolute(np.fft.rfft(frames, nfft))  

    # Power spectrogram
    powspec = (1.0 / nfft) * (magspec ** 2)
    return powspec

def compute_fbank(sig=None, fs=16000, powspec=None, nfft=512, frame_length=0.025, frame_shift=0.01, low_freq=120, high_freq=6000, n_filters=64, winfunc=np.hanning, preemph=0.97):
    '''Compute FBANK features
    Args:
        sig: signal array (default None when powspec is passed in)
        fs: sampling rate in Hz (default 16000)
        nfft: length of the FFT window (default 512)
        powspec: pre-computed power spectrogram (default None)
        frame_length: frame length in seconds (default 0.025 s, 400 samples)
        frame_shift: frame shift in seconds (default 0.01 s, 160 samples)
        low_freq: lowest frequency in Hz (default 120)
        high_freq: highest frequency in Hz (default 6000)
        n_filters: number of filters (default 64)
        winfunc: window function (default numpy.hanning)
        preemph: pre-emphasis factor (default 0.97)
    Returns
        fbank: log-power Mel spectrogram (n_frames x n_filters)
    '''

    # Compute power spectrogram
    if powspec is None:
        powspec = compute_powspec(sig, fs, nfft=nfft, frame_length=frame_length, frame_shift=frame_shift, winfunc=winfunc, preemph=preemph)

    # Compute mel filters
    fb = mel_filters(nfft, low_freq, high_freq, n_filters, fs)

    # Power Mel spectrogram
    fbank = np.dot(powspec, fb)

    # FBANK: log-power Mel spectrogram
    fbank = power_to_db(fbank)
    return fbank

def compute_mfcc(sig=None, fs=16000, fbank=None, n_ceps=20, include_c0=False, cmn=True, cep_lifter=22, nfft=512, frame_length=0.025, frame_shift=0.01, low_freq=120, high_freq=6000, n_filters=64, winfunc=np.hanning, preemph=0.97):
    '''Compute MFCC features
    Args:
        sig: signal array (default None is fbank is passed in)
        fs: sampling rate in Hz (default 16000)
        fbank: pre-computed log-power Mel spectrogram (default None)
        n_ceps: number of cepstral coefficients (default 20)
        cep_lifter: cepstral liftering order (default 22)
        include_c0: if include C0 (default False)
        cmn: cepstral mean normalisation (default True)
        nfft: length of the FFT window (default 512)
        frame_length: frame length in seconds (default 0.025 s, 400 samples)
        frame_shift: frame shift in seconds (default 0.01 s, 160 samples)
        low_freq: lowest frequency in Hz (default 120)
        high_freq: highest frequency in Hz (default 6000)
        n_filters: number of filters (default 64)
        winfunc: window function (default numpy.hanning)
        preemph: pre-emphasis factor (default 0.97)
    Returns
        mfcc: MFCC feature vectors (n_frames x n_ceps)
    '''

    # Compute FBANK features
    if fbank is None:
        fbank = compute_fbank(sig, fs, nfft=nfft, frame_length=frame_length, frame_shift=frame_shift, low_freq=low_freq, high_freq=high_freq, n_filters=n_filters, winfunc=winfunc, preemph=preemph)

    # Apply DCT to get n_ceps MFCCs, omit C0
    if include_c0:
        mfcc = dct(fbank, type=2, axis=1, norm='ortho')[:,:n_ceps]
    else:
        mfcc = dct(fbank, type=2, axis=1, norm='ortho')[:,1:n_ceps]
        n_ceps -= 1

    # Liftering
    lift = 1 + (cep_lifter / 2.0) * np.sin(np.pi * np.arange(n_ceps) / cep_lifter)
    mfcc *= lift

    # Cepstral mean and variance normalisation
    if cmn == True:
        mfcc = (mfcc - np.mean(mfcc, axis=0)) / np.std(mfcc, axis=0)

    return mfcc
