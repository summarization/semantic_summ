#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import sys
import codecs

def evaluate(test_file, posteriors, thresh=0.5):
    """
    calculate precision, recall, and f-score
    """
    gold_labels = []
    with codecs.open(test_file, 'r', 'utf-8') as infile:
        for line in infile:
            label = int(line.strip().split('\t')[0]) # first column of test file
            gold_labels.append(label)
    
    pred_labels = []
    with codecs.open(posteriors, 'r', 'utf-8') as infile:
        for line in infile:
            p0 = float(line.strip().split()[0]) # first column of posteriors file
            label = 0 if p0 > thresh else 1
            pred_labels.append(label)
    
    assert len(gold_labels) == len(pred_labels), \
    "gold labels do not align with predicted labels"
    
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    
    for gold, pred in zip(gold_labels, pred_labels):
        if gold == 1 and pred == 1:
            true_pos += 1
        if gold == 0 and pred == 1:
            false_pos += 1
        if gold == 0 and pred == 0:
            true_neg += 1
        if gold == 1 and pred == 0:
            false_neg += 1
    
    # calculate precision, recall, f-score
    prec = true_pos * 100 / (true_pos + false_pos) if true_pos + false_pos > 0 else 0.0
    rec = true_pos * 100 / (true_pos + false_neg) if true_pos + false_neg > 0 else 0.0
    fscore = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    
    print 'true_pos: [%d] false_pos: [%d] true_neg: [%d] false_neg: [%d]' % (true_pos, false_pos, true_neg, false_neg)
    print 'prec: [%.0f%%] rec: [%.0f%%] fscore: [%.0f%%]' % (prec, rec, fscore)
    return

if __name__ == '__main__':
    test_file = '/home/user/Runs/Experiments/EdgeFilter/test_file'
    posteriors_file = '/home/user/Runs/Experiments/EdgeFilter/posteriors'
    thresh = float(sys.argv[1])
    
    evaluate(test_file, posteriors_file, thresh)
