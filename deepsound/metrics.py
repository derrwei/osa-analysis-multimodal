########################################################
# Copyright (c) 2018 University of Sheffield
#
# Ning Ma (n.ma@sheffield.ac.uk)
# 18 Oct 2018
########################################################

import numpy as np
import os
import sys
from os.path import join, basename
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


def compute_statistical_measures(con_mat):
    '''Compute statistical measures for each class from a confusion matrix
    Args:
        con_mat: confusion matrix
    Returns:
        precision: precision rate in percentages
        recall: recall rate in percentages (aka sensitivity)
        f1: f-measure scores per class
        specificity: recall rate
        accuracy: accuracy rate
    '''
    
    # Compute metrics
    TP = np.diag(con_mat)
    FP = con_mat.sum(axis=0) - TP
    FN = con_mat.sum(axis=1) - TP
    TN = con_mat.sum() - (FP + FN + TP)
    accuracy = (TP + TN) / (TP + FP + FN + TN) * 100
    precision = TP / (TP + FP) * 100
    recall = TP / (TP + FN) * 100
    f1 = 2 * precision * recall / (precision + recall)
    specificity = TN / (TN + FP) * 100
    
    return precision, recall, f1, specificity, accuracy


def compute_error_statistics(ref, hyp):
    '''Computes error statistics (insertions, deletions, substitutions) given two vectors
    Args:
        ref: reference class vector (0...n_classes-1)
        hyp: hypothesis class vector (0...n_classes-1)
    Returns:
        con_mat: class confusion matrix, the diagonal gives the correct hits
        con_ins: array of insertion errors for each class
        con_del: array of delection errors for each class
    '''
    C_HIT = 0
    C_SUB = 1
    C_INS = 2
    C_DEL = 3
    n_classes = max(ref) + 1 # class index starts from 0 ... n_classes - 1
    con_mat = np.zeros((n_classes, n_classes), dtype=np.int16)
    con_ins = np.zeros(n_classes, dtype=np.int16)
    con_del = np.zeros(n_classes, dtype=np.int16)
    len_ref = len(ref)
    len_hyp = len(hyp)
    # Edit distance (Levenshtein distance)
    ed = np.zeros((len_ref + 1, len_hyp + 1), dtype=np.int16)
    # Back tracing
    bt = np.zeros((len_ref + 1, len_hyp + 1), dtype=np.int16)

    # First column represents the case where we achieve zero
    # hypothesis labels by deleting all reference labels.
    ed[1:, 0] = np.arange(1, len_ref + 1)
    bt[1:, 0] = C_DEL

    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis labels into a zero-length reference.
    ed[0, 1:] = np.arange(1, len_hyp + 1)
    bt[0, 1:] = C_INS

    # Dynamic programming
    for i in range(1, len_ref + 1):
        for j in range(1, len_hyp + 1):
            if ref[i-1] == hyp[j-1]:
                ed[i,j] = ed[i-1,j-1]
                bt[i,j] = C_HIT
            else:
                # ed[i-1,j-1]: diagonal move -> SUB
                # ed[i,j-1]: horizontal move -> INS
                # ed[i-1,j]: vertical move -> DEL
                ed[i,j] = ed[i-1,j-1] + 1
                bt[i,j] = C_SUB
                if ed[i,j] > ed[i,j-1] + 1:
                    ed[i,j] = ed[i,j-1] + 1
                    bt[i,j] = C_INS
                if ed[i,j] > ed[i-1,j] + 1:
                    ed[i,j] = ed[i-1,j] + 1
                    bt[i,j] = C_DEL

    # Trace back to collect the statistics
    i = len_ref
    j = len_hyp
    while i > 0 or j > 0:
        if bt[i][j] == C_HIT:
            i -= 1
            j -= 1
            con_mat[ref[i], hyp[j]] += 1
        elif bt[i][j] == C_SUB:
            i -= 1
            j -= 1
            con_mat[ref[i], hyp[j]] += 1
        elif bt[i][j] == C_INS:
            j -= 1
            con_ins[hyp[j]] += 1
        elif bt[i][j] == C_DEL:
            i -= 1
            con_del[ref[i]] += 1

    return con_mat, con_ins, con_del
    

def group_frames_labels(frame_labels):
    '''Merge neighbour frames that have the same labels into a list of groups
    Args:
        frame_labels: frame-level labels
    Returns:
        groups: a list of groups
    '''
    groups = [frame_labels[0]]
    prev_group = frame_labels[0]
    for i in range(1, len(frame_labels)):
        if frame_labels[i] != prev_group:
            prev_group = frame_labels[i]
            groups.append(prev_group)
    return groups


def report_frame_metrics(class_labels, true_classes, pred_classes, cm_file=None):
    '''Report frame-level metric results
    Args:
        class_labels: class labels
        true_classes: a numpy array of the reference label vector
        pred_classes: a numpy array of the predicted label vector
        cm_file: if not None, plot confusion matrix to cm_file
    '''
    true_classes = np.asarray(true_classes)
    pred_classes = np.asarray(pred_classes)
    # Compute metrics per class
    con_mat = metrics.confusion_matrix(true_classes, pred_classes)
    precision, recall, f1, specificity, accuracy = compute_statistical_measures(con_mat)
    counts = np.sum(con_mat, axis=1)

    # Compute overall accuracy
    acc = (true_classes == pred_classes).mean() * 100

    # Print metrics
    n_classes = len(class_labels)
    n_bars = 14 * n_classes + 14 * 7 - 4
    print('')
    print('Frame-level metrics')
    print('='*n_bars)
    # Print overall accuracy
    print('Overall (all classes): Acc = {:.2f}%'.format(acc))
    print('-'*n_bars)
    print('{:>7}'.format(''), end='')
    for lab in class_labels:
        print('{:>14}'.format(lab), end='')
    print('{:>14}'.format('N'), end='')
    print('{:>14}'.format('Recall'), end='')
    print('{:>14}'.format('Precision'), end='')
    print('{:>14}'.format('F-measure'), end='')
    print('{:>14}'.format('Specificity'), end='')
    print('{:>14}'.format('Accuracy'))
    print('-'*n_bars)
    # Print normalised confusion matrix
    for r in range(n_classes):
        print('{:>7}'.format(class_labels[r]), end='')
        for c in range(n_classes):
            print('{:14d}'.format(con_mat[r,c]), end='')
        print('{:14d}'.format(counts[r]), end='')
        print('{:13.2f}%'.format(recall[r]), end='')
        print('{:13.2f}%'.format(precision[r]), end='')
        print('{:13.2f}%'.format(f1[r]), end='')
        print('{:13.2f}%'.format(specificity[r]), end='')
        print('{:13.2f}%'.format(accuracy[r]))
    print('='*n_bars)

    if cm_file is not None:
        # Normalise confusion matrix
        plot_confusion_matrix(con_mat, class_labels, cm_file)


def report_event_metrics(class_labels, true_classes, pred_classes, cm_file=None):
    '''Report event-level metric results
    Args:
        class_labels: class labels
        true_classes: a list of arrays for the reference class labels
        pred_classes: a list of arrays for the predicted class labels
        cm_file: if not None, plot confusion matrix to cm_file
    '''
    if len(true_classes) != len(pred_classes):
        print('The lengths of the two lists are not the same')
        return None
    
    all_con_mat = []
    all_con_ins = []
    all_con_del = []
    for i in range(len(true_classes)):
        # Convert frame-level labels to event labels
        ref = group_frames_labels(true_classes[i])
        hyp = group_frames_labels(pred_classes[i])
        con_mat, con_ins, con_del = compute_error_statistics(ref, hyp)
        all_con_mat.append(con_mat)
        all_con_ins.append(con_ins)
        all_con_del.append(con_del)
        
    all_con_mat = np.sum(all_con_mat, axis=0)
    all_con_ins = np.sum(all_con_ins, axis=0)
    all_con_del = np.sum(all_con_del, axis=0)
    
    H = np.sum(np.diag(all_con_mat))
    S = np.sum(all_con_mat) - H
    I = np.sum(all_con_ins)
    D = np.sum(all_con_del)
    N = S + D + H
    wer = (S + I + D) / N * 100
    corr = H / N * 100
    acc = 100 - wer
    n_classes = len(class_labels)
    n_bars = 14 * n_classes + 14 * 9 - 4
    print('')
    print('Event-level metrics')
    print('='*n_bars)
    print('Overall (all classes): [H={}, S={}, I={}, D={}, N={}] Corr={:.2f}%, Acc={:.2f}%, EER={:.2f}%'.format(H, S, I, D, N, corr, acc, wer))
    print('-'*n_bars)
    print('{:>7}'.format(''), end='')
    for i in range(n_classes):
        print('{:>14}'.format(class_labels[i]), end='')
    print('{:>14}'.format('H'), end='')
    print('{:>14}'.format('S'), end='')
    print('{:>14}'.format('I'), end='')
    print('{:>14}'.format('D'), end='')
    print('{:>14}'.format('N'), end='')
    print('{:>14}'.format('Correct'), end='')
    print('{:>14}'.format('Accuracy'), end='')
    print('{:>14}'.format('EER'))
    print('-'*n_bars)
    # Print confusion matrix
    for i in range(n_classes):
        print('{:>7}'.format(class_labels[i]), end='')
        for j in range(n_classes):
            print('{:14d}'.format(all_con_mat[i,j]), end='')
        H = all_con_mat[i,i]
        S = np.sum(all_con_mat[i,:]) - H
        I = all_con_ins[i]
        D = all_con_del[i]
        N = S + D + H
        corr = H / N * 100
        wer = (S + I + D) / N * 100
        acc = 100 - wer
        print('{:14d}'.format(H), end='')
        print('{:14d}'.format(S), end='')
        print('{:14d}'.format(I), end='')
        print('{:14d}'.format(D), end='')
        print('{:14d}'.format(N), end='')
        print('{:13.2f}%'.format(corr), end='')
        print('{:13.2f}%'.format(acc), end='')
        print('{:13.2f}%'.format(wer))
    print('='*n_bars)

    if cm_file is not None:
        # Normalise confusion matrix
        plot_confusion_matrix(all_con_mat, class_labels, cm_file)

    # Print metrics based on substitutions and hits
    precision, recall, f1, specificity, accuracy = compute_statistical_measures(all_con_mat)
    print('')
    print('Event-level metrics without considering deletions and insertions')
    print('='*80)
    print('{:>7}'.format(''), end='')
    print('{:>14}'.format('Recall'), end='')
    print('{:>14}'.format('Precision'), end='')
    print('{:>14}'.format('F-measure'), end='')
    print('{:>14}'.format('Specificity'), end='')
    print('{:>14}'.format('Accuracy'))
    print('-'*80)
    # Print normalised confusion matrix
    for r in range(n_classes):
        print('{:>7}'.format(class_labels[r]), end='')
        print('{:13.2f}%'.format(recall[r]), end='')
        print('{:13.2f}%'.format(precision[r]), end='')
        print('{:13.2f}%'.format(f1[r]), end='')
        print('{:13.2f}%'.format(specificity[r]), end='')
        print('{:13.2f}%'.format(accuracy[r]))
    print('='*80)


def plot_confusion_matrix(cm, class_labels, cm_file=None):
    '''Plos confustion matrix
    Args:
        cm: confusion matrix
        class_labels: labels for each class
        cm_file: pdf file for saving the plot
    '''
    # Plot confusion matrix
    ax = sns.heatmap(cm*100/np.sum(cm,axis=1)[:,np.newaxis], square=True, annot=True, fmt='.2f', cbar=False,
            xticklabels=class_labels, yticklabels=class_labels, cmap="PuBu", rasterized=True)
    for t in ax.texts:
        t.set_text(t.get_text() + '%')
    plt.xlabel('Predicted class label')
    plt.ylabel('True class label')
    #plt.title('Normalised confusion matrix')

    # Save plot
    if cm_file is not None:
        plt.savefig(cm_file, bbox_inches='tight')
        plt.close()
