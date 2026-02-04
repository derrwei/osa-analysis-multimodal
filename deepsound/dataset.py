########################################################
# Copyright (c) 2018 Deep Sound
#
# Ning Ma
# 18 Sep 2018
########################################################

import numpy as np
import os
import sys
from os.path import join, basename, dirname, exists, isdir, splitext
import scipy.signal
import tgt
import json
import gzip
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from deepsound import sigproc
import datetime
import seaborn as sns
from itertools import groupby

class FeatureConfig:
    '''Feature configuration class
    '''
    def __init__(self, feat_type='fbank', target_fs=16000, frame_length=.04, frame_shift=.02, 
        low_freq=75, high_freq=7500, n_filters=64, winfunc=np.hanning):
        '''
        Args:
            feat_type: feature type (default 'fbank'): 'powspec', 'fbank', 'mfcc'
            target_fs: target sampling rate in Hz (default 16000)
            frame_length: frame length in s (default 50 ms)
            frame_shift: frame shift in s (default 20 ms)
            low_freq: lowest frequency in Hz (default 120)
            high_freq: highest frequency in Hz (default 7500)
            n_filters: number of filters (default 64)
            winfunc: window function (default numpy.hanning)
        '''
        self.feat_type = feat_type
        # General
        self.target_fs = target_fs
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.nfft = sigproc.next_power_of_two(frame_length*target_fs)
        self.winfunc = winfunc
        self.preemph = 0.97
        # FBANK
        self.low_freq = low_freq
        if high_freq > target_fs/2:
            self.high_freq = int(target_fs/2)
        else:
            self.high_freq = high_freq
        self.n_filters = n_filters
        # MFCC
        self.n_ceps = 20
        self.include_c0 = False
        self.cep_lifter = 22

    def compute_features(self, sig, fs):
        '''Compute features depending on feature type
        Args:
            sig: signal numpy array
            fs: sampling rate in Hz
        Returns:
            Extracted feature matrix (n_frames, n_features)
        '''
        # Check if need to resample signal
        if fs != self.target_fs:
            sig = scipy.signal.resample(sig, int(self.target_fs/fs*len(sig)))

        # Normalise to the same RMS level
        sig = sigproc.rms_normalise(sig) # to be tested the effect of rms normalisation

        # Add small white noise to avoid zeros
        if sig.dtype == 'int16':
            noise = np.random.normal(0, 1, size=len(sig))
        else:
            noise = np.random.normal(0, 1e-9, size=len(sig))

        if self.feat_type == 'powspec':
            return sigproc.compute_powspec(sig+noise, fs=self.target_fs, nfft=self.nfft, frame_length=self.frame_length, frame_shift=self.frame_shift, 
                winfunc=self.winfunc, preemph=self.preemph)
        elif self.feat_type == 'fbank':
            return sigproc.compute_fbank(sig+noise, fs=self.target_fs, nfft=self.nfft, frame_length=self.frame_length, frame_shift=self.frame_shift, 
                low_freq=self.low_freq, high_freq=self.high_freq, n_filters=self.n_filters, winfunc=self.winfunc, preemph=self.preemph)
        elif self.feat_type == 'mfcc':
            return sigproc.compute_mfcc(sig+noise, fs=self.target_fs, nfft=self.nfft, frame_length=self.frame_length, frame_shift=self.frame_shift, 
                low_freq=self.low_freq, high_freq=self.high_freq, n_filters=self.n_filters, winfunc=self.winfunc, preemph=self.preemph,
                n_ceps=self.n_ceps, include_c0=self.include_c0, cep_lifter=self.cep_lifter)
        else:
            raise ValueError('Unknown feature type: {}'.format(self.feat_type))


    def plot_features(self, features):
        '''Plot features according to the feature type
        Args:
            features: [n_frames x n_features]
        '''
        n_frames = features.shape[0]
        sig_len = self.frame_shift * n_frames
        if self.feat_type == 'powspec':
            plt.imshow(features.T, extent=(0,sig_len,self.low_freq,self.high_freq), aspect='auto', origin='lower')
        elif self.feat_type == 'fbank':
            mel_freqs = np.round(sigproc.mel_freqs(self.low_freq, self.high_freq, self.n_filters)).astype(int)
            yticks = [int(n*self.n_filters/5)+1 for n in range(5)] + [self.n_filters-1]
            yticks[0] = 0
            yticklabels = mel_freqs[yticks]
            yticklabels[-1] = self.high_freq
            plt.imshow(features.T, extent=(0,sig_len,0,self.n_filters-1), aspect='auto', origin='lower')
            plt.yticks(yticks, yticklabels)
        ax = plt.gca()
        ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
        plt.xlabel('Time [s]', fontsize=20)
        plt.ylabel('Frequency [Hz]', fontsize=20)

    def plot_annotations(self, annotations):
        '''Plot annotations on top of features
        Args:
            annotations: a list of dicts holding annotations
        '''
        ax = plt.gca()
        face_colours = ['lightskyblue', 'tomato', 'darkgrey', 'whitesmoke']
        plotting_labels = [0, 1]
        label_names = ['Non-apnoea', 'Apnoea']
        n_labels = len(plotting_labels)
        legends = [None] * n_labels
        height = 4
        ypos = self.n_filters
        for i in range(n_labels):
            lab = label_names[i]
            legends[i] = mpatches.Patch(facecolor=face_colours[i], label=lab)

        for item in annotations:
            lab = item['annotation']
            if lab in plotting_labels:
                idx = plotting_labels.index(lab)
                xpos = item['start']
                dur = item['end'] - xpos
                ax.add_patch(mpatches.Rectangle([xpos, ypos], dur, height, ec='None', clip_on=False, fc=face_colours[idx]))
        ax.legend(bbox_to_anchor=(.41, -0.05), handles=legends, ncol=n_labels)
        plt.tight_layout()


class FeatureScalerMinMax:
    '''Normalise features to [0,1]
    '''
    def transform(self, x):
        x_min = np.min(x)
        x_max = np.max(x)
        return (x - x_min) / (x_max - x_min)

    def fit_transform(self, x):
        return self.transform(x)

class FeatureScalerStandard:
    '''Normalise features to zero mean and unit variance
    '''
    def transform(self, x):
        x_mean = np.mean(x)
        x_std = np.std(x)
        return (x - x_mean) / x_std
    
    def fit_transform(self, x):
        return self.transform(x)


def process_features(features, context, stride=1, flatten_context=False, feat_scaler=None):
    '''Process feature vectors (normalisation, context creation)
    Args:
        features: feature matrix [n_frames x n_features]
        context: context size (minimum 3)
        stride: context stride size (default 1)
        flatten_context: If True, then the context window is flattened (default False)
        feat_scaler: feature scaler for normalisation (default None)
    Returns:
        features: processed features
    '''
    half_context = int(context/2)
    n_frames = features.shape[0]

    # Normalise features
    if feat_scaler is not None:
        features = feat_scaler.transform(features)

    # Create context
    n_features = features.shape[1]
    ind = np.tile(np.arange(-half_context*stride, (context-half_context)*stride, stride), (n_frames, 1)) + np.tile(np.arange(0, n_frames, 1), (context, 1)).T
    ind[ind<0] = 0
    ind[ind>=n_frames] = n_frames - 1
    features = features[ind]

    if flatten_context:
        features = features.reshape(n_frames, context * n_features)
    return features


def save_list_to_file(fname, strlist):
    '''Saves a list to a file, one item per line
    '''
    with open(fname, 'w') as f:
        for s in strlist:
            f.write(s + '\n')


def load_list_from_file(fname):
    '''Loads all the lines in a file as a list
    '''
    strlist = list()
    with open(fname, 'r') as f:
        for line in f:
            strlist.append(line.strip())
    return strlist

    
def collapse_label(label):
    '''Collapse labels (0: non-apnoea, 1: apnoea)
    '''
    label = label.strip()
    if label == '' or label == 0:
        return 0 # normal
    else:
        return 1 # apnoea or desat
    

def load_annotations_json(json_file):
    '''Load annotations from a JSON file
    Args:
        json_file: JSON file name
    Returns:
        annotations: A list of dicts holding the annotations
    '''
    with open(json_file) as f:
        annotations = json.load(f)['annotations']
        return annotations


def load_annotations_textgrid(textgrid_file, tier_index=0):
    '''Load annotations from a text grid
    Args:
        textgrid_file: text grid file name
    Returns:
        annotations: A list of dicts holding the annotations
    '''
    annotation = tgt.read_textgrid(textgrid_file, include_empty_intervals=True)

    # Get tier names
    annotation_tier_name = annotation.get_tier_names()
    
    # Access tiers
    if len(annotation_tier_name) <= tier_index:
        return []
        
    annotation_tier = annotation.get_tier_by_name(annotation_tier_name[tier_index])
    
    annotations = []
    for item in annotation_tier:
        lab = collapse_label(item.text)
        annotations.append({'start': item.start_time, 'end': item.end_time, 'annotation': lab, 'event': item.text})
    return annotations


def save_annotations_json(json_file, annotations):
    '''Load annotations from a JSON file
    Args:
        json_file: JSON file name
        annotations: A list of dicts holding the annotations
    Returns:
    '''
    data = {'file': json_file, 'annotations': annotations}
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)


def save_annotations_textgrid(textgrid_file, annotations):
    '''Load annotations from a TextGrid file
    Args:
        textgrid_file: TextGrid file name
        annotations: A list of dicts holding the annotations
    Returns:
    '''
    with open(textgrid_file, 'w') as f:
        n_intervals = len(annotations)
        xmin = annotations[0]['start']
        xmax = annotations[-1]['end']        
        f.write('File type = "ooTextFile"\n')
        f.write('Object class = "TextGrid"\n\n')
        f.write('xmin = {}\n'.format(xmin))
        f.write('xmax = {}\n'.format(xmax))
        f.write('tiers? <exists>\n')
        f.write('size = 1\n')
        f.write('item []:\n')
        f.write('    item [1]:\n')
        f.write('        class = "IntervalTier"\n')
        f.write('        name = "snoring"\n')
        f.write('        xmin = {}\n'.format(xmin))
        f.write('        xmax = {}\n'.format(xmax))
        f.write('        intervals: size = {}\n'.format(n_intervals))
        for i, item in enumerate(annotations):
            f.write('        intervals [{}]:\n'.format(i+1))
            f.write('            xmin = {}\n'.format(item['start']))
            f.write('            xmax = {}\n'.format(item['end']))
            f.write('            text = "{}"\n'.format(item['annotation']))


def textgrid_to_json(textgrid_file, json_file, tier_index=0):
    '''Converts a TextGrid file to the JSON format
    Args:
        textgrid_file: TextGrid file name
        json_file: JSON file name
    '''
    annotations = load_annotations_textgrid(textgrid_file, tier_index=tier_index)
    save_annotations_json(json_file, annotations)
    

def json_to_textgrid(json_file, textgrid_file):
    '''Converts a JSON file to the TextGrid format
    Args:
        json_file: JSON file name
        textgrid_file: TextGrid file name
    '''
    annotations = load_annotations_json(json_file)
    save_annotations_textgrid(textgrid_file, annotations)

def annotations_to_frame_labels(annotations, frame_shift):
    '''Convert annotations to frame-level labels
    Args:
        annotations: a list of dictionaries holding the annotations
        frame_shift: frame shift in seconds
    Returns:
        frame_labels: a list of frame labels
    '''
    frame_labels = []
    for item in annotations:
        # Go through each item
        start_frame = int(item['start'] / frame_shift)
        end_frame = int(item['end'] / frame_shift)
        n_frames = end_frame - start_frame
        frame_labels[start_frame:end_frame] = [item['annotation']] * n_frames
    return np.asarray(frame_labels)

def annotations_to_events_labels(annotations, frame_shift):
    '''Convert annotations to frame-level labels
    Args:
        annotations: a list of dictionaries holding the annotations
        frame_shift: frame shift in seconds
    Returns:
        frame_labels: a list of frame labels
    '''
    events_labels = []
    for item in annotations:
        event = "n" if item['event'] in ["", "0"] else item['event']
        # Go through each item
        start_frame = int(item['start'] / frame_shift)
        end_frame = int(item['end'] / frame_shift)
        n_frames = end_frame - start_frame
        events_labels[start_frame:end_frame] = [event] * n_frames
    return np.asarray(events_labels)

def annotations_to_temporal_events_labels(annotations):
    '''Convert annotations to temporal labels
    Args:
        annotations: a list of dictionaries holding the annotations
    Returns:
        temporal_labels: a list of temporal labels
    '''
    temporal_labels = []
    for item in annotations:
        stat_time = int(item['start'])
        end_time = int(item['end'])
        event = "n" if item['event'] in ["", "0"] else item['event']
        temporal_labels[stat_time:end_time] = [event] * (end_time - stat_time)
    return np.asarray(temporal_labels)

def frame_labels_to_annotations(frame_labels, frame_shift):
    '''Convert frame-level labels to annotations
    Args:
        frame_labels: a list of frame labels
        frame_shift: frame shift in seconds
    Returns:
        annotations: a list of dictionaries holding the annotations
    '''
    annotations = []
    start_time = 0
    lab = frame_labels[0]
    n_frames = len(frame_labels)
    for end_frame in range(1, n_frames):
        # Go through each item
        if end_frame == n_frames - 1 or lab != frame_labels[end_frame]:
            end_time = end_frame * frame_shift
            annotations.append({'start': start_time, 'end': end_time, 'annotation': lab})
            lab = frame_labels[end_frame]
            start_time = end_time
    return annotations

import re
import csv
def sort_nicely( l ):
    ''' Sort the given list in the way that humans expect.
    '''
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )

def list_all_files(root_dir, file_ext=('.wav','.WAV')):
    '''Lists all files in a folder and its subfolders that match file_ext
    Args:
        root_dir: top directory
        file_ext: file extensions, e.g. ('.wav','.WAV') or .npy
    Returns:
        A sorted list of files in root_dir
    '''
    filelist = [join(root, name) for root, dirs, files in os.walk(root_dir) 
        for name in files if name.endswith(file_ext)]
    # Uncomment the next line if need a human-sorted list
    sort_nicely(filelist)
    return filelist

def list_all_nights(snorer_dir):
    night_list = next(os.walk(snorer_dir))[1]
    sort_nicely(night_list)
    return night_list

def save_variables(pgz_file, list_variables):
    '''Save results in a gzipped pickle file
    Args:
        pgz_file: gzipped pickle file
        list_variables: a list of variables
    '''
    with gzip.open(pgz_file, 'wb') as f:
        pickle.dump(list_variables, f)


def load_variables(pgz_file):
    '''Load results from a gzipped pickle file
    Args:
        pgz_file: gzipped pickle file
    Returns:
        A list of variables
    '''
    with gzip.open(pgz_file, 'rb') as f:
        return pickle.load(f)

def compute_record_time(file_name):
    '''Compute recording start datetime from file name
    Args:
        file_name: file name containing a datetime string in the format of '2017-06-12-23-41-31-144'
    Returns:
        rec_start: recording start time as a datetime object, e.g. '2017-06-12 23:41:31'
        seg: segment number starting from 0, e.g. 143
    '''
    file_name = splitext(basename(file_name))[0]
    idx = file_name.rfind('-')
    time_str = file_name[:idx]
    rec_start = datetime.datetime.strptime(time_str, '%Y-%m-%d-%H-%M-%S')
    seg = int(file_name[idx+1:]) - 1
    return rec_start, seg


def compute_record_time_ms(file_name):
    '''Compute recording start datetime from file name including msec
    Args:
        file_name: file name containing a datetime string in the format of '2017-06-12-23-41-31-144'
    Returns:
        rec_start: recording start time as a datetime object, e.g. '2017-06-12 23:41:31.144'
    '''
    time_str = splitext(basename(file_name))[0]
    rec_start = datetime.datetime.strptime(time_str, '%Y-%m-%d-%H-%M-%S-%f')
    return rec_start


def compute_record_night_ms(file_name):
    '''Compute recording start night from file name. If recording starts after midnight,
        returns the day before
    Args:
        file_name: file name containing a datetime string, e.g. '2017-06-12-23-41-31-144'
    Returns:
        rec_night: date string representing the recording night, e.g. '2017-06-12'
    '''
    # work out recording start datetime
    rec_start = compute_record_time_ms(file_name)
    # work out night start datetime
    rec_night = datetime.date(rec_start.year, rec_start.month, rec_start.day)
    if rec_start.hour < 12:
        rec_night = rec_night - datetime.timedelta(days=1)
    return str(rec_night)
    

def compute_record_night(file_name):
    '''Compute recording start night from file name. If recording starts after midnight,
        returns the day before
    Args:
        file_name: file name containing a datetime string, e.g. '2017-06-12-23-41-31-144'
    Returns:
        rec_night: date string representing the recording night, e.g. '2017-06-12'
    '''
    # work out recording start datetime
    rec_start, _ = compute_record_time(file_name)
    # work out night start datetime
    rec_night = datetime.date(rec_start.year, rec_start.month, rec_start.day)
    if rec_start.hour < 12:
        rec_night = rec_night - datetime.timedelta(days=1)
    return str(rec_night)
    

def compute_segment_index(file_name, seg_dur=2, night_hour=21):
    '''Compute the segment index from file name, e.g. 2017-06-12-23-41-31-144.json
    Args:
        file_name: file name containing a time string in the format of '2017-06-12-23-41-31-144'
                    is the 144th segment from 2017-06-12 23:41:31
        seg_dur: segment duration in minute. Default 2 (mins)
        night_hour: night start hour. Default 21 (9pm)
    '''
    # work out recording start datetime
    rec_start, seg = compute_record_time(file_name)
    # work out night start datetime
    night_start = datetime.datetime(rec_start.year, rec_start.month, rec_start.day, night_hour)
    if rec_start.hour < 12:
        night_start = night_start - datetime.timedelta(days=1)
    # work out segment index relative to night_start
    td = rec_start + datetime.timedelta(minutes=seg*seg_dur) - night_start
    return int(td.total_seconds() / (60 * seg_dur))


def plot_night_stats(night_data, night_labels, plot_title=None, plot_file=None, seg_dur=2):

    # Time ticks
    time_labels = ['', '10 PM','11 PM','12 AM','1 AM','2 AM','3 AM','4 AM','5 AM','6 AM','7 AM','8 AM','9 AM','10 AM']
    night_dur = len(time_labels) - 1
    n_night_segs = night_dur * 60 // seg_dur
    time_ticks = range(0, n_night_segs+1, int(60/seg_dur))

    #nightly_snore_counts = np.sum(night_data, axis=0)

    # Night ticks
    snorer_ticks = []
    snorer_labels = []
    prev_lab = ''
    for i, lab in enumerate(night_labels):
        if prev_lab != lab[:4]:
            snorer_ticks.append(i)
            #snorer_labels.append('{}:{:>6.0f}'.format(lab[:4], nightly_snore_counts[i]))
            snorer_labels.append(lab[:4])
            prev_lab = lab[:4]
    #night_ticks = range(len(night_labels))
    n_nights = len(night_labels)

    # Plot heatmap
    fig = plt.figure(figsize=(20,80))
    ax = sns.heatmap(night_data, cmap='Blues', cbar=False, rasterized=True)
    if plot_title is not None:
        ax.set_title(plot_title, fontsize=36, pad=70)
    
    # Add top tick labels
    plt.yticks(snorer_ticks, snorer_labels, fontsize=18)
    plt.xticks(time_ticks, time_labels, fontsize=24)
    plt.xticks(rotation=0)
    ax.tick_params(right=True, top=True, labelbottom=True, labeltop=True, which='major', pad=12)

    # Vertical align tick labels
    plt.setp(ax.yaxis.get_majorticklabels(), va='top')

    # Draw lines to separate snorers
    snorer_ticks2 = snorer_ticks + [n_nights]
    for ypos in snorer_ticks2:
        plt.axhline(y=ypos, color='skyblue', linestyle='-')
    
    # Plot stats at the side
    avg = np.nanmean(night_data, axis=1)
    err = np.nanstd(night_data, axis=1)
    ma = np.nanmax(night_data, axis=1)
    ma2 = np.max(ma) // 10 * 10
    upper = 30
    w = upper / ma2

    # Plot guide grid
    fs = 13
    plt.axvline(x=0, color='darkgray', linestyle='-')
    plt.axvline(x=upper*.5, color='lightgray', linestyle=':')
    plt.axvline(x=upper, color='lightgray', linestyle=':')
    plt.axvline(x=upper*1.5, color='lightgray', linestyle=':')

    ax.text(-1.2, -0.3, '0', color='gray', fontsize=fs)
    ax.text(upper*.5-2.5, -0.3, str(int(ma2*.5)), color='gray', fontsize=fs)
    ax.text(upper-2.5, -0.3, str(int(ma2)), color='gray', fontsize=fs)
    ax.text(upper*1.5-2.5, -0.3, str(int(ma2*1.5)), color='gray', fontsize=fs)

    ax.text(-1.2, n_nights+.9, '0', color='gray', fontsize=fs)
    ax.text(upper*.5-2.5, n_nights+.9, str(int(ma2*.5)), color='gray', fontsize=fs)
    ax.text(upper-2.5, n_nights+.9, str(int(ma2)), color='gray', fontsize=fs)
    ax.text(upper*1.5-2.5, n_nights+.9, str(int(ma2*1.5)), color='gray', fontsize=fs)

    # Plot mean and max data
    night_ticks = np.arange(0.5, n_nights)
    for i in range(1, len(snorer_ticks2)):
        idx = (night_ticks>snorer_ticks2[i-1]) & (night_ticks<snorer_ticks2[i])
        ax.errorbar(avg[idx]*w, night_ticks[idx], xerr=err[idx]*w, fmt='o-', color='#ff7f0e', ecolor='darkgray', capsize=5)
        ax.plot(ma[idx]*w, night_ticks[idx], 'o-', color='darkgray')
    
    #for i, v in enumerate(avg):
    #    ax.text(-4, i+.75, str(int(v)).rjust(2), color='gray', fontsize=fs)
    
    plt.xlim([-3, n_night_segs])
    #plt.ylim([-1, n_nights+1])
    ax.errorbar([], [], [], fmt='o-', color='#ff7f0e', ecolor='darkgray', capsize=4, label='Segment Mean')
    ax.plot([], [], 'o-', color='darkgray', label='Segment Max')
    plt.legend(loc='lower left', bbox_to_anchor=(-.04, 1.008), fontsize=fs) #, frameon=False)

    # Save figure
    plt.tight_layout()
    if plot_file is None:
        plt.show()
    else:
        fig.savefig(plot_file, bbox_inches='tight')


def plot_radar_chart(values, labels, title=None, ax=None):
    n_values = len(labels)

    angles=np.linspace(0, 2*np.pi, n_values, endpoint=False)
    values=np.concatenate((values,[values[0]]))
    angles=np.concatenate((angles,[angles[0]]))

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.1)
    ax.set_thetagrids(angles * 180/np.pi, labels)
    if title is not None:
        ax.set_title(title)
    ax.grid(True)

    '''
    # Initialise the radar plot
    ax = plt.subplot(111, polar=True)
    
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], labels, color='grey', size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    step = int(np.round(max(values) / (n_steps+1)))
    yticks = range(step, step*n_steps+step, step)
    plt.yticks(yticks, yticks, color='grey', size=7)
    
    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    
    # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)
    '''

def count_apnoea_events_from_files(json_files):
    apnoea_events = 0
    for fname in json_files:
        if fname.endswith('.npy'):
            fname = fname.replace('.npy', '.json')
        annotations = load_annotations_json(fname)
        for item in annotations:
            apnoea_events += item['annotation']
    return apnoea_events


def count_apnoea_events_from_json(json_files):
    apnoea_events = 0
    total_seconds = 0
    for fname in json_files:
        if fname.endswith('.npy'):
            fname = fname.replace('.npy', '.json')
        annotations = load_annotations_json(fname)
        for item in annotations:
            apnoea_events += item['annotation']
            total_seconds += item['end'] - item['start']
    return apnoea_events, total_seconds/3600


def count_apnoea_events_from_labels(labels):
    apnoea_events = []
    apnoea_events = sum(1 if sum(g) else 0 for  _, g in groupby(labels))
    return apnoea_events


def combine_csv_cross_valid(result_csv_prefix, result_csv_postfix, nfolds, result_csv_out):

    header_row, all_rows = [], []
    for n in range(nfolds):
        result_csv = '{}{}{}'.format(result_csv_prefix, n+1, result_csv_postfix)
        with open(result_csv, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            header_row = next(csvreader)
            for row in csvreader:
                all_rows.append(row)
    all_rows.sort()

    with open(result_csv_out, mode='w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(header_row)
        csvwriter.writerows(all_rows)
