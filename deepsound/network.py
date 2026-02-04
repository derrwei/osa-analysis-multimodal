########################################################
# Copyright (c) 2018 Deep Sound
#
# Ning Ma
# 18 Sep 2018
########################################################

import numpy as np
import os
from os.path import join
import math
import tensorflow as tf
from tensorflow import keras
import torch
from sklearn import metrics
from deepsound import dataset
import gzip
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import csv
from collections import Counter

class ModelConfig:
    '''Model configuration class'''

    def __init__(self, model_desc, feature_config, segment_length=40, segment_shift=10):
        '''
        Args:
            model_desc: a description of the model, e.g. 'snore_cnn'
            feature_config: a FeatureConfig object describing the features
            segment_length: segment length in sec
            segment_shift: segment shift in sec
        '''
        # Workout segment length in frames
        self.segment_length = segment_length
        self.segment_shift = segment_shift
        self.win_size = int(segment_length / feature_config.frame_shift)
        self.hop_size = int(segment_shift / feature_config.frame_shift)
        self.model_desc = model_desc
        self.feature_config = feature_config
        self.flatten_context = False # flatten the context frames
        self.model_name = make_model_name(model_desc, feature_config.feat_type, self.segment_length, self.segment_shift)
        self.batch_size = 128 
        self.epochs = 50
        self.learning_rate = 1e-3
        self.lr_decay = 0.5
        self.min_learning_rate = 1e-5
        self.hidden_activ = 'relu'
        self.output_activ = 'softmax'
        self.loss = 'categorical_crossentropy'
        self.dropout = 0.3
        self.n_kernels = [64, 64, 64] # CNN kernels
        self.kernel_size = [5, 3] # CNN layers
        self.pool_size = 2 # max pooling layer
        self.dense_nodes = 512 # dense layers
        self.lstm_units = [32, 32] # LSTM layers
        self.feature_scaler = None

        self.class_labels = [0, 1]

        # Define metrics
        # self.metrics = [
        #     keras.metrics.TruePositives(name='tp'),
        #     keras.metrics.FalsePositives(name='fp'),
        #     keras.metrics.TrueNegatives(name='tn'),
        #     keras.metrics.FalseNegatives(name='fn'), 
        #     keras.metrics.BinaryAccuracy(name='accuracy'),
        #     keras.metrics.Precision(name='precision'),
        #     keras.metrics.Recall(name='recall'),
        #     keras.metrics.AUC(name='auc'),
        # ]
        self.metric_names = ['TP', 'FP', 'TN', 'FN', 'Accuracy', 'Precision', 'Recall', 'AUC']

    def n_classes(self):
        '''Return the number of classes
        '''
        return len(self.class_labels)

    def labels_to_class_vector(self, labels):
        '''Convert a list of string labels to a class vector 
        Args:
            labels: a list of string labels, e.g. ['x', 'b', s', 'b', 'x', 'o', 'x', 's']
        Returns:
            class_vector: a vector of integers 0 to (n_classes-1) representing all classes
        '''
        sort_idx = np.argsort(self.class_labels)
        class_vector = sort_idx[np.searchsorted(self.class_labels, labels, sorter=sort_idx)]
        return class_vector

    def class_vector_to_labels(self, class_vector):
        '''Convert a class vector to a list of string labels
        Args:
            class_vector: a vector of integers 0 to (n_classes-1) representing all classes
        Returns:
            A list of string labels, e.g. ['x', 'b', s', 'b', 'x', 'o', 'x', 's']
        '''
        class_labels = np.asarray(self.class_labels)
        return class_labels[class_vector]

    def probs_to_labels(self, probs):
        '''Convert model predictions posteriors to a label vector
        Args:
            probs: a matrix representation of the model prediction posteriors
        Returns:
            A list of string labels, e.g. ['x', 'b', s', 'b', 'x', 'o', 'x', 's']
        '''
        return self.class_vector_to_labels(np.argmax(probs, axis=1))

    def annotations_to_frame_labels(self, annotations):
        '''Convert annotations to frame-level labels
        Args:
            annotations: a list of dictionaries holding the annotations
        Returns:
            frame_labels: a list of frame labels
        '''
        return dataset.annotations_to_frame_labels(annotations, self.feature_config.frame_shift)

    def annotations_to_class_vector(self, annotations):
        '''Convert annotations a class vector
        Args:
            annotations: a list of dictionaries holding the annotations
        Returns:
            class_vector: a vector of integers 0 to (n_classes-1) representing all classes
        '''
        frame_labels = dataset.annotations_to_frame_labels(annotations, self.feature_config.frame_shift)
        return self.labels_to_class_vector(frame_labels)

    def frame_labels_to_annotations(self, frame_labels):
        '''Convert frame-level labels to annotations
        Args:
            frame_labels: a list of frame labels
        Returns:
            annotations: a list of dictionaries holding the annotations
        '''
        return dataset.frame_labels_to_annotations(frame_labels, self.feature_config.frame_shift)

    def class_vector_to_annotations(self, class_vector):
        '''Convert a class vector to annotations
        Args:
            class_vector: a vector of integers 0 to (n_classes-1) representing all classes
        Returns:
            annotations: a list of dictionaries holding the annotations
        '''
        frame_labels = self.class_vector_to_labels(class_vector)
        return dataset.frame_labels_to_annotations(frame_labels, self.feature_config.frame_shift)

    def normalise_features(self, features, fit_transform=False):
        '''Normalise features using feature scaler
        Args:
            features: numpy array of shape (n_segments, win_size, n_features)
            fit_transform: if True, use fit_transform instead of transform
        Returns:
            features: normalised features
        '''
        n_segments, win_size, n_features = features.shape
        features = np.reshape(features, newshape=(-1, n_features))
        if fit_transform:
            features = self.feature_scaler.fit_transform(features)
        else:
            features = self.feature_scaler.transform(features)
        features = np.reshape(features, newshape=(n_segments, win_size, n_features))
        return features

    def load_features_labels_from_list(
            self,
            feat_files,
            concatenate=True,
            undersample_nonapnoea=True,
            load_desat=False,
            osa_dur=15.,
            desat_delay=10.
    ):
        '''Load segment-level features and labels from frame-level feature files
        Args:
            feat_files: feature file list
            concatenate: If False, features and labels are returned as a list coressponding to feat_files (default True)
            undersample_nonapnoea: If True, under sample non-apnoea files by using non-overlapping segments
            load_desat: If True, load oxygen desatuation data as percentage of desat frames
            osa_dur: OSA duration in seconds for deciding segment OSA labels
            desat_delay: delay in seconds for desaturation data
        Returns:
            all_features: concatenated features (n_segments, win_size, n_features)
            all_labels: corresponding frame labels (n_segments, n_outputs) # rewritten to include event, start_time, end_time, feature_file
        '''
 # sec
        desat_delay = int(desat_delay / self.feature_config.frame_shift)
        # Read all data into lists first before concatenating them in one go
        # Otherwise it is REALLY slow to append data on the fly
        n_files = len(feat_files)
        all_features = []
        all_labels = []

        lab_dur = int(osa_dur / self.feature_config.frame_shift)
        # osa_dur_thr = int(1 / self.feature_config.frame_shift)

        if lab_dur > self.win_size:
            lab_dur = self.win_size
        lab_st = (self.win_size - lab_dur) // 2
        if n_files == 0:
            return None, None
        for fname in sorted(feat_files):
            frame_features = np.load(fname).astype(np.float32)
            annotations = dataset.load_annotations_json(fname.replace('.npy', '.json')) if fname.endswith('.npy') else dataset.load_annotations_json(fname.replace('.pt', '.json'))
            frame_labels = dataset.annotations_to_frame_labels(annotations, self.feature_config.frame_shift)

            events_labels = dataset.annotations_to_events_labels(annotations, self.feature_config.frame_shift)

            # Make sure features and labels dimensions agree
            n_frames = min(frame_features.shape[0], len(frame_labels))
            frame_features = frame_features[:n_frames, :]
            frame_labels = frame_labels[:n_frames]
            events_labels = events_labels[:n_frames]

            # Process features (normalisation, create context etc)
            n_frames = frame_features.shape[0]
            if n_frames < self.win_size:
                continue

            # If file contains apnoea, use a smaller hop_size
            hop_size = self.hop_size
            if undersample_nonapnoea:
                # Check if frame_labels contains only zeros
                if not np.any(frame_labels):
                    # non-apnoea files
                    hop_size = self.win_size
                    # continue

            # Normalise features
            if self.feature_scaler is not None:
                frame_features = self.feature_scaler.fit_transform(frame_features)

            # Load Oxygen Desat data
            if load_desat:
                parts = fname.split(os.path.sep)
                tgt_file = os.path.sep.join(parts[:len(parts) - 5] + ['data'] + parts[-3:-1] + ['audio'] + parts[-1:])
                tgt_file = tgt_file.replace('.npy', '.TextGrid')
                desat_annotations = dataset.load_annotations_textgrid(tgt_file, tier_index=3)
                desat_frames = dataset.annotations_to_frame_labels(desat_annotations, self.feature_config.frame_shift)
                desat_frames = desat_frames[:n_frames]
                if len(desat_frames) == 0:
                    continue
                # print(tgt_file)
                # print(len(desat_frames))

            # Create context
            n_segments = int((n_frames - self.win_size + hop_size) / hop_size)
            ind = np.tile(np.arange(0, self.win_size), (n_segments, 1)) + np.tile(
                np.arange(0, n_segments * hop_size, hop_size), (self.win_size, 1)).T
            # ind[ind<0] = 0
            # ind[ind>=n_frames] = n_frames - 1
            all_features.append(frame_features[ind])  # shape = (n_segments, win_size, n_features)
            seg_labels = []
            for i, labels in enumerate(frame_labels[ind]):
                # Use the most common label in the central half
                min_ind, max_ind = np.min(ind[i]), np.max(ind[i])
                min_time, max_time = min_ind * self.feature_config.frame_shift, max_ind * self.feature_config.frame_shift
                lab = Counter(labels[lab_st:lab_st + lab_dur]).most_common(1)[0][0]
                event = Counter(events_labels[ind[i]][lab_st:lab_st + lab_dur]).most_common(1)[0][0]
                lab = [int(lab), event, min_time, max_time, fname.split(os.path.sep)[-1]]
                '''
                if np.sum(labels[lab_st:lab_st+lab_dur]) > osa_dur_thr: # .5 sec
                    lab = 1
                else:
                    lab = 0
                '''
                # if lab == 0:
                #    # if not OSA, also consider dur > 10 sec
                #    osa_dur = np.count_nonzero(labels) * self.feature_config.frame_shift
                #    if osa_dur >= 10:
                #        lab = 1
                if load_desat:
                    desat_ind = ind[i] + desat_delay
                    desat_ind = desat_ind[:self.win_size // 2]
                    desat_ind = desat_ind[desat_ind < n_frames]
                    desat = np.mean(desat_frames[desat_ind])
                    lab.append(desat)
                seg_labels.append(lab)
            all_labels.append(np.array(seg_labels))

        if concatenate:
            # all_features.shape = (n_segments, win_size, n_features)
            # all_labels.shape = (n_segments,)
            return np.concatenate(all_features), np.concatenate(all_labels)
        else:
            return all_features, all_labels

    def compute_features(self, sig, fs):
        '''Wrap function to FeatureConfig.compute_features()
        Args:
            sig: signal numpy array
            fs: sampling rate in Hz
        Returns:
            Extracted feature matrix (n_frames, n_features)
        '''
        return self.feature_config.compute_features(sig, fs)

    def transition_params(self):
        '''Returns transition_params (log transition probs)
        '''
        n_classes = self.n_classes
        trans_probs = np.zeros((n_classes, n_classes)) + self.next_tp
        np.fill_diagonal(trans_probs, self.self_tp)
        return trans_probs

    def classify_features(self, model, features):
        '''Classification using features
        Args:
            model: Keras model
            sig: single channel waveform numpy array
            fs: sampling rate in Hz
        Returns:
            pred_classes a numpy array holding the predicted label indices
        '''
        # Normalise features and create context
        features = dataset.process_features(features, context=self.context, stride=self.stride, 
            flatten_context=self.flatten_context, feat_scaler=self.feature_scaler)

        # Reshaping features for CNNs
        if self.flatten_context == False:
            features = features.reshape(features.shape[0], features.shape[1], features.shape[2], 1)

        print('  feature = {}, min = {:.4f}, max = {:.4f}'.format(features.shape, features.min(), features.max()))

        # Compute posterior probabilities
        probs = model.predict(features)

        # Viterbi decoding
        #best_seq, _ = tfa.text.viterbi_decode(score=np.log(probs.clip(min=1e-10)), transition_params=self.transition_params())
        #pred_classes = np.asarray(best_seq)

        # No Viterbi
        pred_classes = np.argmax(probs, axis=1)

        return pred_classes

    def classify_signal(self, model, sig, fs):
        '''Classification using waveform signals
        Args:
            model: Keras model
            sig: single channel waveform numpy array
            fs: sampling rate in Hz
        Returns:
            pred_classes: a numpy array holding the predicted label indices
        '''
        # Extract features
        features = self.feature_config.compute_features(sig, fs)
        
        # Classify features
        pred_classes = self.classify_features(model, features)

        return pred_classes

    def model_callbacks(self, model_path, monitor='val_auc', mode='max'):
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=model_path, 
            verbose=1, 
            save_weights_only=True,
            monitor=monitor,
            mode=mode,
            save_best_only=True)
        model_early_stopping = keras.callbacks.EarlyStopping(
            verbose=1,
            monitor=monitor,
            mode=mode,
            patience=20,
            restore_best_weights=True)
        model_reduce_learning_rate = keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, 
            mode=mode,
            patience=3,
            factor=self.lr_decay,
            min_lr=self.min_learning_rate)
        model_callbacks = [model_checkpoint, model_early_stopping, model_reduce_learning_rate]
        return model_callbacks

    def create_CNN(self, initial_bias=None):
        # Construct the DNN
        inputs = keras.Input(shape=(self.win_size, self.feature_config.n_filters, 1))

        # CNN1
        x = keras.layers.Conv2D(filters=16, activation='relu', kernel_size=[3,3])(inputs)
        x = keras.layers.BatchNormalization()(x)
        #x = keras.layers.Activation(keras.activations.relu)(x)
        x = keras.layers.MaxPooling2D(pool_size=(4,2))(x)
        x = keras.layers.Dropout(self.dropout)(x)
        # CNN2
        x = keras.layers.Conv2D(filters=32, activation='relu', kernel_size=[3,3])(x)
        x = keras.layers.BatchNormalization()(x)
        #x = keras.layers.Activation(keras.activations.relu)(x)
        x = keras.layers.MaxPooling2D(pool_size=(4,2))(x)
        x = keras.layers.Dropout(self.dropout)(x)
        # CNN3
        x = keras.layers.Conv2D(filters=64, activation='relu', kernel_size=[3,3])(x)
        x = keras.layers.BatchNormalization()(x)
        #x = keras.layers.Activation(keras.activations.relu)(x)
        x = keras.layers.MaxPooling2D(pool_size=(4,2))(x)
        x = keras.layers.Dropout(self.dropout)(x)

        # FC1
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(512, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        #x = keras.layers.Activation(keras.activations.relu)(x)
        x = keras.layers.Dropout(self.dropout)(x)

        #Output
        if initial_bias is not None:
            initial_bias = keras.initializers.Constant(initial_bias)
        outputs = keras.layers.Dense(1, activation='sigmoid', bias_initializer=initial_bias)(x)
        model = keras.Model(inputs, outputs)
        model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=self.learning_rate), metrics=self.metrics)
        return model

    def create_TFCNN(self, initial_bias=None):
        # Same CNN but process the time and frequency domains separately
        # input = [time, frequency, channel]
        inputs = keras.Input(shape=(self.win_size, self.feature_config.n_filters, 1))

        # Time dimension
        x = keras.layers.Conv2D(filters=16, activation='relu', kernel_size=[5,1])(inputs)
        x = keras.layers.Conv2D(filters=16, activation='relu', kernel_size=[5,1])(inputs)
        x = keras.layers.MaxPooling2D(pool_size=(2,1))(x)
        x = keras.layers.Dropout(self.dropout)(x)
        # Frequency dimension
        x = keras.layers.Conv2D(filters=16, activation='relu', kernel_size=[1,3])(inputs)
        x = keras.layers.Conv2D(filters=16, activation='relu', kernel_size=[1,3])(inputs)
        x = keras.layers.MaxPooling2D(pool_size=(1,2))(x)
        x = keras.layers.Dropout(self.dropout)(x)

        # Time dimension
        x = keras.layers.Conv2D(filters=32, activation='relu', kernel_size=[5,1])(inputs)
        x = keras.layers.Conv2D(filters=32, activation='relu', kernel_size=[5,1])(inputs)
        x = keras.layers.MaxPooling2D(pool_size=(2,1))(x)
        x = keras.layers.Dropout(self.dropout)(x)
        # Frequency dimension
        x = keras.layers.Conv2D(filters=32, activation='relu', kernel_size=[1,3])(inputs)
        x = keras.layers.Conv2D(filters=32, activation='relu', kernel_size=[1,3])(inputs)
        x = keras.layers.MaxPooling2D(pool_size=(1,2))(x)
        x = keras.layers.Dropout(self.dropout)(x)
        
        # Both dimensions
        x = keras.layers.Conv2D(filters=64, activation='relu', kernel_size=[5,3])(inputs)
        x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
        x = keras.layers.Dropout(self.dropout)(x)

        # FC1
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(512, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(self.dropout)(x)

        #Output
        if initial_bias is not None:
            initial_bias = keras.initializers.Constant(initial_bias)
        outputs = keras.layers.Dense(1, activation='sigmoid', bias_initializer=initial_bias)(x)
        model = keras.Model(inputs, outputs)
        model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=self.learning_rate), metrics=self.metrics)
        return model
   
    def create_CNN_MT(self, initial_bias=None):
        # Construct the Multi-Tasking CNN
        inputs = keras.Input(shape=(self.win_size, self.feature_config.n_filters, 1))

        # CNN1
        x = keras.layers.Conv2D(filters=16, activation='relu', kernel_size=[3,3])(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(pool_size=(4,2))(x)
        x = keras.layers.Dropout(self.dropout)(x)
        # CNN2
        x = keras.layers.Conv2D(filters=32, activation='relu', kernel_size=[3,3])(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(pool_size=(4,2))(x)
        x = keras.layers.Dropout(self.dropout)(x)
        # CNN3
        x = keras.layers.Conv2D(filters=64, activation='relu', kernel_size=[3,3])(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(pool_size=(4,2))(x)
        x = keras.layers.Dropout(self.dropout)(x)

        # FC1
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(self.dropout)(x)

        # OSA Branch
        osa_x = keras.layers.Dense(128, activation='relu')(x)
        osa_x = keras.layers.Dropout(self.dropout)(osa_x)
        if initial_bias is not None:
            initial_bias = keras.initializers.Constant(initial_bias)
        osa_output = keras.layers.Dense(1, activation='sigmoid', bias_initializer=initial_bias, name='osa_output')(osa_x)


        # SpO2 Branch
        desat_x = keras.layers.Dense(128, activation='relu')(x)
        desat_x = keras.layers.Dropout(self.dropout)(desat_x)
        desat_x = keras.layers.Dense(64, activation='relu')(desat_x)
        desat_x = keras.layers.Dropout(self.dropout)(desat_x)
        desat_output = keras.layers.Dense(1, activation='relu', name='desat_output')(desat_x)

        model = keras.Model(inputs, [osa_output, desat_output])
        
        model.compile(
              loss = {'osa_output': 'binary_crossentropy', 'desat_output': 'mse'},
              loss_weights = {'osa_output': 1., 'desat_output': 1.}, 
              optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), 
              metrics={'osa_output': self.metrics, 'desat_output': keras.metrics.MeanSquaredError(name='mse')})
        return model

model_root = 'models'
def make_model_name(model_desc, feat_type, win_size, hop_size):
    '''Make a model name based on model parameters
    Args:
        model_desc: a description of the model, e.g. 'snore_cnn'
        feat_type: feature type, e.g. 'fbank'
        win_size: window size in s
        hop_size: hop size in s
    '''
    return '{}_{}_win{}s_hop{}s'.format(model_desc, feat_type, int(win_size), int(hop_size))


def temp_model_path():
    '''Get a temporary model path
    '''
    global model_root
    model_dir = join(model_root, 'temp_training_models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir


def save_keras_model(model_name, model, model_config):
    '''Save Keras models to disk
    Args:
        model_name: unique model name
        model: Keras model
        model_config: ModelConfig object
    Returns:
    '''
    global model_root
    if not os.path.exists(model_root):
        os.makedirs(model_root)

    # Construct architecture file and weight file
    arch_file = join(model_root, model_name + '.json')
    config_file = join(model_root, model_name + '.config')
    weight_file = join(model_root, model_name + '.h5')

    # Save config
    pickle.dump(model_config, open(config_file, 'wb'))

    # Serialise model to JSON
    model_json = model.to_json()
    with open(arch_file, 'w') as json_file:
        json_file.write(model_json)

    # Serialise weights to HDF5
    model.save_weights(weight_file)


def load_keras_model(model_name):
    '''Load Keras models from disk
    Args:
        model_name: unique model name
    Returns:
        model: loaded model
        model_config: ModelConfig object describing the model
    '''
    global model_root

    # Construct architecture, weight and config files
    arch_file = join(model_root, model_name + '.json')
    config_file = join(model_root, model_name + '.config')
    weight_file = join(model_root, model_name + '.h5')

    # Load model config first
    model_config = pickle.load(open(config_file, 'rb'))

    # Load json and create model
    with open(arch_file, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = tf.keras.models.model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights(weight_file)

    return model, model_config


def compute_evaluation_metrics(true_labels, pred_labels):
    '''Compute evaluation metrics per class for multilabel classification
    Args:
        true_labels: true label vector, e.g. [1, 0, 1, 2, 2, 3, 0]
        pred_labels: predicted label vector, e.g. [0, 3, 1, 2, 2, 3, 0]
    Returns:
        cm: confusion matrix
        precision: precision rate per class in percentages
        recall: recall rate per class in percentages (also known as sensitivity)
        f1: f-measure scores per class
    '''
    cm = metrics.confusion_matrix(true_labels, pred_labels)

    # Compute metrics
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    #TN = cm.sum() - (FP + FN + TP)
    #accuracy = (TP + TN) / (TP + FP + FN + TN) * 100
    precision = TP / (TP + FP) * 100
    recall = TP / (TP + FN) * 100
    f1 = 2 * precision * recall / (precision + recall)
    
    #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    return cm, precision, recall, f1


def save_results(pgz_file, list_variables):
    '''Save results in a gzipped pickle file
    Args:
        pgz_file: gzipped pickle file
        list_variables: a list of variables
    '''
    with gzip.open(pgz_file, 'wb') as f:
        pickle.dump(list_variables, f)


def load_results(pgz_file):
    '''Load results from a gzipped pickle file
    Args:
        pgz_file: gzipped pickle file
    Returns:
        A list of variables
    '''
    with gzip.open(pgz_file, 'rb') as f:
        return pickle.load(f)


def print_frame_results(class_labels, true_classes, pred_classes, csv_file=None, cm_file=None):
    '''Print frame-level metric results
    Args:
        class_labels: a list of class labels
        true_classes: a numpy array of the reference label vector
        pred_classes: a numpy array of the predicted label vector
        csv_file: save results in a CSV file (default None)
        cm_file: save confusion matrix plot file (default None)
    '''
    # Compute metrics per class
    cm, precision, recall, f1 = compute_evaluation_metrics(true_classes, pred_classes)
    counts = cm.sum(axis=1)

    # Normalise confusion matrix
    cm = cm.astype('float') / counts[:, np.newaxis] * 100

    # Compute overall accuracy
    acc = (true_classes == pred_classes).mean() * 100

    # Print metrics
    n_classes = len(class_labels)
    print('Normalised confusion matrix')
    print('-'*80)
    print('{:>5}'.format('Label'), end='')
    for lab in class_labels:
        print('{:>10}'.format(lab), end='')
    print('{:>10}'.format('Counts'), end='')
    print('{:>10}'.format('Recall'), end='')
    print('{:>10}'.format('Precision'), end='')
    print('{:>10}'.format('F-measure'))

    # Print normalised confusion matrix
    for r in range(n_classes):
        print('{:>10}'.format(class_labels[r]), end='')
        for c in range(n_classes):
            print('{:9.2f}%'.format(cm[r,c]), end='')
        print('{:10d}'.format(counts[r]), end='')
        print('{:9.2f}%'.format(recall[r]), end='')
        print('{:9.2f}%'.format(precision[r]), end='')
        print('{:9.2f}%'.format(f1[r]))
    print('')

    # Print overall accuracy
    print('Overall Accuracy: {:.2f}%'.format(acc))
    print('-'*80)

    if csv_file is not None:
        with open(csv_file, 'w', newline='') as f:
            csvwriter = csv.writer(f, quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['Label'] + class_labels + ['Counts', 'Recall[%]', 'Precision[%]', 'F-measure[%]'])
            for r in range(n_classes):
                row = [class_labels[r]]
                for c in range(n_classes):
                    row += [cm[r,c]]
                row += [counts[r], recall[r], precision[r], f1[r]]
                csvwriter.writerow(row)
            csvwriter.writerow(['Accuracy[%]', acc])
    if cm_file is not None:
        # Plot confusion matrix
        sns.heatmap(cm, square=True, annot=True, fmt='.2f', cbar=False,
                xticklabels=class_labels, yticklabels=class_labels, cmap="YlGnBu", rasterized=True)
        plt.xlabel('predicted label')
        plt.ylabel('true label')
        plt.title('Normalised confusion matrix')

        # Save plot
        plt.savefig(cm_file, bbox_inches='tight')


def plot_metrics(history, fig_file=None):
    metrics =  ['loss', 'auc', 'precision', 'recall']
    fig = plt.figure(figsize=(10,10))
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch,  history.history[metric], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([.4,1])
        else:
            plt.ylim([0,1])
        plt.legend()
    plt.tight_layout()
    if fig_file is None:
        plt.show()
    else:
        plt.savefig(fig_file, bbox_inches='tight')


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,20])
    plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
