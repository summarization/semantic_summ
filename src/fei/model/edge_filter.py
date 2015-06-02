#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
import re
import logging

from random import random
from collections import Counter

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class EdgeFilter(object):
    
    def __init__(self):
        self.percepts = []
        self.percept_indices = {}
        self.percept_counts = Counter()
        
        self.model = None
        self.model_l1 = 0.0
        self.model_accuracy = 0.0
        self.model_posteriors_iter = None

    def fireFeat(self, template_name, template_val, feat_val, add_feat=True):
        """
        fire a feature
        template_name: e.g., curr_word | template_val: e.g., "today"
        feat_val: e.g., 1 | add_feat: e.g., True
        """
        percept_name = (template_name, template_val)
        
        if add_feat:
            percept_id = self.percept_indices.setdefault(percept_name, len(self.percepts))
            if percept_id == len(self.percepts): self.percepts.append(percept_name)
            self.percept_counts[percept_id] += 1
        else: 
            percept_id = self.percept_indices.get(percept_name)
        
        if percept_id is not None: 
            return {str(percept_id): feat_val}
        return {}

    def extract(self, edge, add_feat=True):
        """
        return fired features
        """
        relation = edge.relation
        node1 = edge.node1
        node2 = edge.node2
        
        dep_node1 = str(len(node1.graphIdx.split('.')))
        dep_node2 = str(len(node2.graphIdx.split('.')))
        
        active_percepts = {}
        
        active_percepts.update(self.fireFeat('relation', relation, 1, add_feat))
        active_percepts.update(self.fireFeat('dep_node1', dep_node1, 1, add_feat))
        active_percepts.update(self.fireFeat('dep_node2', dep_node2, 1, add_feat))
        
        if re.search(r'-\d{2}$', node1.concept):
            active_percepts.update(self.fireFeat('verb_node1', '', 1, add_feat))
        if re.search(r'-\d{2}$', node2.concept):
            active_percepts.update(self.fireFeat('verb_node2', '', 1, add_feat))

        concept1 = re.sub(r'-\d{2}$', '', node1.concept)
        concept2 = re.sub(r'-\d{2}$', '', node2.concept)
        
        active_percepts.update(self.fireFeat('concept1', concept1, 1, add_feat))
        active_percepts.update(self.fireFeat('concept2', concept2, 1, add_feat))
        
        return active_percepts


    def prune(self, train_data, cutoff=1):
        """
        Feature pruning
        """
        # percepts to be removed
        deleted_percept_ids = set()
        
        for percept_id, count in self.percept_counts.items():
            if count < cutoff:
                deleted_percept_ids.add(percept_id)
                # update percept_indices, no need to prune test_data
                del self.percept_indices[self.percepts[percept_id]]
        
        # update training data
        for active_percepts in train_data:
            for percept_id in deleted_percept_ids:
                active_percepts.pop(percept_id, None)
        return
    
    def data_iter(self, edges, labels, add_feat=True):
        """
        paired edges and labels
        """
        for edge, label in zip(edges, labels):
            # down-sample negative edges during training
            if label == 0 and add_feat == True: 
                if random() >= 0.1: continue
            active_percepts = self.extract(edge, add_feat)
            yield (active_percepts, label)

    def trainLR(self, train_iter, l1=0.01):
        """
        run logistic regression using creg
        train_iter: iterator of (dict, float) pairs over train set
        """
        logger.debug('start training LR model...')
        import creg #@UnresolvedImport
        train_data = creg.CategoricalDataset(train_iter)
        self.model = creg.LogisticRegression()
        self.model.fit(train_data, l1)
        self.model_l1 = l1
        return
    
    def testLR(self, test_iter):
        """
        test_iter: iterator of (dict, float) pairs over test set
        return: iterator of posterior probabilities
        """
        logger.debug('start testing LR model...')
        import creg #@UnresolvedImport
        test_data = creg.CategoricalDataset(test_iter)
        self.model_posteriors_iter = self.model.predict_proba(test_data)
        return

    def writeFiles(self, output_dir):
        # write feature mapping
        import os.path
        import codecs
        
        output_file_percepts = os.path.join(output_dir, 'percepts')
        with codecs.open(output_file_percepts, 'w', 'utf-8') as outfile:
            for percept, index in sorted(self.percept_indices.iteritems(), key=lambda x: x[1]):
                count = self.percept_counts[index]
                outfile.write('%d\t%d\t%s\n' % (index, count, '\t'.join([str(w) for w in percept])))

        # write model weights
        output_file_weights = os.path.join(output_dir, 'weights_' + str(self.model_l1))
        with codecs.open(output_file_weights, 'w', 'utf-8') as outfile:
            for label in sorted(self.model.weights):
                for fname, w in sorted(self.model.weights[label].iteritems(), 
                                       key=lambda x: x[1], reverse=True):
                    outfile.write('%s\t%s\t%f\n' % (label, fname, w))

        # write model posteriors
        output_file_posteriors = os.path.join(output_dir, 'posteriors')
        with codecs.open(output_file_posteriors, 'w', 'utf-8') as outfile:
            for posteriors in self.model_posteriors_iter:
                outfile.write('%s\n' % '\t'.join([str(p) for p in posteriors]))
        
        return


    # TODO: parallel feature extraction?
    # TODO: how to deal with data imbalance issue
    # TODO: is it possible to load weights to LR model?



    










