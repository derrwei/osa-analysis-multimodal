from sklearn.utils import shuffle
import warnings 
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    # import h5py

import os, sys
from os.path import join, isdir, exists
import argparse
import csv
import numpy as np
import random

import torch
import lightning
import sys
# sys.path.append('../')
# sys.path.append('../../')
from lightning_modules.osa_classification import AudioOnlyLightningModule
import sklearn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from deepsound import dataset, network


# Need this for running CNNs with NVIDIA GeForce RTX 2080 cards
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#tf.compat.v1.enable_eager_execution(config=config)


# ================================
# Parse input
# ================================
#
# python3 ds_train_cnn.py --feature fbank --context 3 --stride 1
parser = argparse.ArgumentParser(description='Deep sleep convolutional neural network.')
parser.add_argument('--fold', default=0, help='fold number')
parser.add_argument('--feature', default='fbank', help='Feature type: fbank, mfcc')
parser.add_argument('--segment_length', default=30, help='segment length in sec', type=float)
parser.add_argument('--segment_shift', default=10, help='segment shift in sec', type=float)
parser.add_argument('--frame_length', default=.05, help='frame length in sec', type=float)
parser.add_argument('--frame_shift', default=.02, help='frame shift in sec', type=float)
parser.add_argument('--model', default='cohort1_cnn', help='Model name: cohort1_cnn, cohort1-cohort2_cnn, etc')
parser.add_argument('--device', default='all', help='all, ios, android')
parser.add_argument('--gpu', default=None, help='gpu id to be used')
parser.add_argument('--train_corpus', nargs='+', default=['cohort1'], help='Training corpus list: cohort1, cohort2')
parser.add_argument('--eval_corpus', nargs='+', default=['cohort1'], help='Evaluation corpus list: cohort1, cohort2')
parser.add_argument('--arch', default='cnn', help='Model architecture: cnn, tfcnn')
parser.add_argument('--model_path', default='', help='Path to the trained model checkpoint')
parser.add_argument('--model_type', default='cnn', help='Model type: cnn, tfcnn')
parser.add_argument('--exp_name', default='test_exp', help='Experiment name')
args = parser.parse_args()

# Specify which GPU(s) to be used
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'  # To avoid MKL errors

def load_model(model_path: str, model_type: str, device: torch.device) -> AudioOnlyLightningModule:
    """Load and return a trained model."""
    print(f"   Loading model from: {model_path}")
    model = AudioOnlyLightningModule(verbose=False, model_type = model_type)
    # print the model architecture
    print(model.model)
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state.get('state_dict', state), strict=False)
    # load mean and std
    # model.mean = state['hyper_parameters']['mean']
    # model.std = state['hyper_parameters']['std']
    # print(f"   Loaded model mean: {model.mean}, std: {model.std}")
    model.eval().to(device)
    return model

# ================================
# Setup paths
# ================================
#
corpus_root = 'data/osa-brahms'
feat_root = join(corpus_root, 'features')
feat_type = args.feature
feature_config = dataset.FeatureConfig(feat_type, frame_length=args.frame_length, frame_shift=args.frame_shift)
feat_dir = join(feat_root, '{}_win{}ms_hop{}ms'.format(feat_type, int(feature_config.frame_length*1000), int(feature_config.frame_shift*1000)))
print('Feature dir: {}'.format(feat_dir))
if not os.path.exists(feat_dir):
    sys.exit('Unable to locate feature directory {}'.format(feat_dir))

exp_name = args.exp_name
result_root = 'results/{}'.format(exp_name)
if not os.path.exists(result_root):
    os.makedirs(result_root)

# ================================
# Load patient data
# ================================
#
# nights = []
# eval_desc = args.device
# for corpus in args.eval_corpus:
#     if args.device == 'all':
#         night_csv = join(f'{corpus_root}/flists', 'osa-{}_nights.csv'.format(corpus))
#     else:
#         if 'cohort1' in args.eval_corpus:
#             night_csv = join(f'{corpus_root}/flists', 'osa-{}_patients_{}.csv'.format(corpus, args.device))
#         else:
#             night_csv = join(f'{corpus_root}/flists', 'osa-{}_nights_{}.csv'.format(corpus, args.device))
#     if not os.path.exists(night_csv):
#         sys.exit('Unable to locate evaluation night csv {}'.format(night_csv))
#     eval_desc += '-' + corpus
#     with open(night_csv, newline='') as csvfile:
#         csvreader = csv.reader(csvfile, delimiter=',')
#         # Skip header
#         next(csvreader)
#         for row in csvreader:
#             nights.append(row[0])

# get nights from txt
nights_txt = 'data/folds/osa_nights_fold_{}.txt'.format(args.fold)
nights = []
with open(nights_txt, 'r') as f:
    for line in f:
        nights.append(line.strip())

print('='*80)
print('Loaded {} nights for evaluation'.format(len(nights)))
print('='*80)

# ================================
# Load DNN models
# ================================

# model_root = 'models'
train_desc = 'osa'
for corpus in args.train_corpus:
    train_desc += '-' + corpus
model_desc = '{}_{}_{}'.format(train_desc, args.arch, args.device)
model_config = network.ModelConfig(model_desc, feature_config, segment_length=args.segment_length, segment_shift=args.segment_shift)
model_config.feature_scaler = dataset.FeatureScalerStandard() #StandardScaler()
model_config.flatten_context = False
model_config.epochs = 30

# # Construct the DNN
# model_dir = join(model_root, model_config.model_name)
# model_path = join(model_dir, 'checkpoint')
# if not exists(model_path):
#     sys.exit('Unable to locate model {}'.format(model_path))

# if args.arch == 'cnn':
#     model = model_config.create_CNN()
# elif args.arch == 'tfcnn':
#     model = model_config.create_TFCNN()
# else:
#     sys.exit('Unknown architecture {}'.format(args.arch))
# print('Loading pre-trained model weights')
# model.load_weights(model_path)
# model_path = "/export/catch2/users/xiaolei/projects/icassp_multimodal/logs/multimodal/fusion_dual/fold3/checkpoints/last.ckpt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Using device: {device}")
model = load_model(model_path=args.model_path, model_type=args.model_type, device=device)


# ================================
# Evaluation
# ================================

model_name = "cnnt"
# result_csv = 'results/{}/eval_{}_fold_{}.csv'.format(args.model_type, args.model_type, args.fold)
result_csv = os.path.join(result_root, 'eval_fold_{}.csv'.format(args.fold))
with open(result_csv, 'w', newline='') as f:
    csvwriter = csv.writer(f, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(['Night', 'Hours', 'Scored AHI', 'Label AHI', 'Pred AHI', 'Loss'] + ['TP','FP','TN','FN','Accuracy','Precision', 'Recall', 'AUC'])

# nights = ['023/2020-01-19', '006/2019-10-03', '006/2019-10-04', '035/2020-01-21', '039/2020-01-25', '039/2020-01-26', '066/2019-10-30']
for num,night_eval in enumerate(nights):
    print('\n==== Evaluating night {}/{}: {}'.format(num+1, len(nights), night_eval))

    feature_files = dataset.list_all_files(join(feat_dir, night_eval), '.npy')
    night_hours = len(feature_files) / 30 # 2-min files
    apnoea_events_ref, _ = dataset.count_apnoea_events_from_json(feature_files)
    te_data, te_labels = model_config.load_features_labels_from_list(feature_files, undersample_nonapnoea=False)
    if te_data is None:
        continue
    te_labels = te_labels[:,1]  # Use apnoea labels only
    te_labels = [0 if label=='n' else 1 for label in te_labels]
    apnoea_events_lab = dataset.count_apnoea_events_from_labels(te_labels)    
    te_data = torch.from_numpy(te_data).float()
    # te_labels = torch.from_numpy(te_labels).float()
    te_labels = torch.tensor(te_labels).long()
    # to gpu
    te_data = te_data.to(device)
    te_labels = te_labels.to(device) 
    batch_size = 256
    test_predictions = []
    test_losses = []
    for batch_idx, i in enumerate(range(0, len(te_data), batch_size)):
        batch_data = te_data[i:i+batch_size]
        # print(batch_data.mean(), batch_data.std())
        # batch_data = batch_data.float()
        batch_labels = te_labels[i:i+batch_size]
        with torch.no_grad():
            output = model._shared_step((batch_data, batch_labels), 'test')
            # print(output['probs'])
            preds = output['preds']
            # probs = output['probs']
            # preds = torch.argmax(probs, dim=1)
            # preds = (torch.argmax(output['probs'], dim=1))
            # print(preds)
            loss = output['loss']
            test_predictions.extend(preds)
            # print(test_predictions)
            # print(test_predictions)
            test_losses.append(loss.item())
        # del batch_data, batch_labels, output
    del te_data
    test_loss = np.mean(test_losses)
    print('  Test loss: {:.4f}'.format(test_loss))
    test_predictions = torch.stack(test_predictions).cpu().numpy()
    # test_predictions = np.vstack(test_predictions)
    # test_predictions = torch.sigmoid(torch.tensor(test_predictions)).numpy()
    # test_predictions = model.forward(torch.tensor(te_data, dtype=torch.float32).to(next(model.parameters()).device)).detach().cpu().numpy()
    apnoea_events_pred = dataset.count_apnoea_events_from_labels(test_predictions)
    te_labels = te_labels.detach().cpu().numpy()
    precision = precision_score(te_labels, test_predictions)
    recall = recall_score(te_labels, test_predictions)
    auc_score = roc_auc_score(te_labels, test_predictions)
    accuracy = accuracy_score(te_labels, test_predictions)
    classification_report_str = classification_report(te_labels, test_predictions)
    tn, fp, fn, tp = confusion_matrix(te_labels, test_predictions).ravel()
    print('Confusion matrix:')
    print(f'  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')
    print('Precision: {:.4f}, Recall: {:.4f}, AUC: {:.4f}'.format(precision, recall, auc_score))
    print('Classification report:')
    print(classification_report_str)

    print('  {}\nhours: {:.2f}\nScored AHI: {:.1f}\nLabel AHI: {:.1f}\n Pred AHI: {:.1f}'.format(
        night_eval,
        night_hours,
        apnoea_events_ref / night_hours, 
        apnoea_events_lab / night_hours, 
        apnoea_events_pred / night_hours))
    print()
    result_row = [
        night_eval,
        night_hours, 
        apnoea_events_ref / night_hours, 
        apnoea_events_lab / night_hours, 
        apnoea_events_pred / night_hours
    ]
    # results = model.evaluate(te_data, te_labels)
    with open(result_csv, 'a', newline='') as f:
        csvwriter = csv.writer(f, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(result_row + list([test_loss, tp,fp,tn,fn, accuracy, precision, recall, auc_score]))
