#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import codecs
import os

from math import sqrt
from random import shuffle
from collections import namedtuple
from collections import Counter
from fei.model.feat_vec import FeatureVector
from fei.model.utils import getLogger

logger = getLogger()
PerfScore = namedtuple('PerfScore', 'prec, rec, fscore')


def getPRFScores(intersect, num_selected, num_gold):
    """
    calculate prec, rec, and f-score
    """
    prec, rec, fscore = 0.0, 0.0, 0.0

    if num_selected != 0: 
        prec = (intersect * 100)/num_selected
    if num_gold != 0: 
        rec = (intersect * 100)/num_gold
    if prec != 0.0 and rec != 0.0: 
        fscore = 2 * prec * rec / (prec + rec)
        
    return PerfScore(prec, rec, fscore)


def logPRFScores(text, perf_score):
    """
    generate a log message for prec, rec, and f-score
    """
    message = ('[%s] prec=[%.1f%%] rec=[%.1f%%] fscore=[%.1f%%]' 
               % (text, perf_score.prec, perf_score.rec, perf_score.fscore))
    
    logger.debug(message)
    return message


class ParamEstimator(object):
    
    def learnParamsAdaGrad(self, decoder, corpus, param_file, loss_func, num_passes=10, oracle_len='nolen'):
        """
        learn parameters using Structured Perceptron, Ramp Loss, (Hinge??)
        """        
        logger.debug('start learning parameters...')
        shuffle(corpus) # shuffle corpus
        
        avg_weights = FeatureVector()
        curr_instances = 0
        
        node_perf = PerfScore(0.0, 0.0, 0.0) # node performance
        edge_perf = PerfScore(0.0, 0.0, 0.0) # edge performance
        
        eta = 1.0 # stepsize
        l2reg = 0.0 # 
        node_cost_scaling = 1.0 # cost scaling factor
        edge_cost_scaling = 1.0 # cost scaling factor
        
        sumSq = FeatureVector()
        
        for curr_num_passes in xrange(1, num_passes+1):
            logger.debug('#curr_num_passes#: %d' % curr_num_passes)
            
            for instance in corpus:
                curr_instances += 1
                logger.debug('processing instance %d...' % curr_instances)
                
                # perceptron loss
                if loss_func.startswith('perceptron'):
                    gradient, selected_nodes, selected_edges, score_pred = decoder.decode(instance, oracle_len)
                    plus_feats, oracle_nodes, oracle_edges, score_true = decoder.oracle(instance)
                    
                    curr_loss = score_pred - score_true  # @UnusedVariable
                    gradient -= plus_feats
                    decoder.weights -= eta * gradient
                
                # ramp loss + cost-augmented decoding
                if loss_func.startswith('ramp'):                         
                    node_cost, edge_cost = node_cost_scaling, edge_cost_scaling
                    gradient, _, _, score_plus_cost = decoder.decode(instance, oracle_len, node_cost, edge_cost)
                    plus_feats, selected_nodes, selected_edges, score_minus_cost = decoder.decode(instance, oracle_len, -1.0 * node_cost, -1.0 * edge_cost)
                    _, oracle_nodes, oracle_edges, score_true = decoder.oracle(instance)
                
                    curr_loss = score_plus_cost - score_minus_cost  # @UnusedVariable
                    gradient -= plus_feats
                    
                    for k, v in gradient.iteritems():
                        if v == 0.0: continue
                        sumSq[k] = sumSq.get(k, 0.0) + v * v
                        decoder.weights[k] = decoder.weights.get(k, 0.0) - eta * v / sqrt(sumSq[k])
                    
                    if l2reg != 0.0:
                        for k, v in decoder.weights.iteritems():
                            if v == 0.0: continue
                            value = l2reg * v
                            sumSq[k] = sumSq.get(k, 0.0) + value * value
                            decoder.weights[k] = v - eta * value / sqrt(sumSq[k])
                
                # hinge-loss + cost-augmented decoding
                if loss_func.startswith('hinge'): 
                    node_cost, edge_cost = node_cost_scaling, edge_cost_scaling
                    gradient, selected_nodes, selected_edges, score_plus_cost = decoder.decode(instance, oracle_len, node_cost, edge_cost)
                    plus_feats, oracle_nodes, oracle_edges, score_true = decoder.oracle(instance)
    
                    curr_loss = score_plus_cost - score_true  # @UnusedVariable
                    gradient -= plus_feats

                    for k, v in gradient.iteritems():
                        if v == 0.0: continue
                        sumSq[k] = sumSq.get(k, 0.0) + v * v
                        decoder.weights[k] = decoder.weights.get(k, 0.0) - eta * v / sqrt(sumSq[k])
                    
                    if l2reg != 0.0:
                        for k, v in decoder.weights.iteritems():
                            if v == 0.0: continue
                            value = l2reg * v
                            sumSq[k] = sumSq.get(k, 0.0) + value * value
                            decoder.weights[k] = v - eta * value / sqrt(sumSq[k])
                
                # use gold nodes and edges to calculate P/R/F
                num_gold_nodes, num_gold_edges = instance.gold
                # P/R/F scores of nodes and edges, for current instance
                # Edge recall can not reach %100 since decoding produces only tree structure
                intersect_nodes = set(selected_nodes) & set(oracle_nodes)
                curr_node_perf = getPRFScores(len(intersect_nodes), len(selected_nodes), num_gold_nodes)
                logPRFScores('train_node', curr_node_perf)
                
                intersect_edges = set(selected_edges) & set(oracle_edges)
                curr_edge_perf = getPRFScores(len(intersect_edges), len(selected_edges), num_gold_edges)
                logPRFScores('train_edge', curr_edge_perf)

                # P/R/F scores of nodes and edges, averaged across all curr_instances
                node_perf = PerfScore(*[sum(x) for x in zip(node_perf, curr_node_perf)])
                edge_perf = PerfScore(*[sum(x) for x in zip(edge_perf, curr_edge_perf)])
                
                logPRFScores('train_node_avg', 
                             PerfScore(node_perf.prec/curr_instances, node_perf.rec/curr_instances, 
                                       node_perf.fscore/curr_instances))              
                logPRFScores('train_edge_avg', 
                             PerfScore(edge_perf.prec/curr_instances, edge_perf.rec/curr_instances, 
                                       edge_perf.fscore/curr_instances))
                                
            # averaging weight vectors
            avg_weights += decoder.weights
            
            # output averaged weight vectors to file
            curr_weights = FeatureVector()
            curr_weights += avg_weights * (1/curr_num_passes)
            if param_file:
                with codecs.open(param_file, 'w', 'utf-8') as outfile:
                    outfile.write('#curr_num_passes#: %d\n' % curr_num_passes)
                    outfile.write('%s\n' % curr_weights.toString())

        final_weights = FeatureVector()
        final_weights += avg_weights * (1/num_passes)
        return final_weights
    
    def learnParams(self, decoder, corpus, param_file, loss_func, num_passes=10, oracle_len='nolen'):
        """
        learn parameters using Structured Perceptron, Ramp Loss, (Hinge??)
        """        
        logger.debug('start learning parameters...')
        shuffle(corpus) # shuffle corpus
        
        avg_weights = FeatureVector()
        curr_instances = 0
        
        node_perf = PerfScore(0.0, 0.0, 0.0) # node performance
        edge_perf = PerfScore(0.0, 0.0, 0.0) # edge performance
        
        eta = 1.0 # stepsize
        l2reg = 0.0 # 
        node_cost_scaling = 1.0 # cost scaling factor
        edge_cost_scaling = 1.0 # cost scaling factor
        
        for curr_num_passes in xrange(1, num_passes+1):
            logger.debug('#curr_num_passes#: %d' % curr_num_passes)
            
            for instance in corpus:
                curr_instances += 1
                logger.debug('processing instance %d...' % curr_instances)
                
                # perceptron loss
                if loss_func.startswith('perceptron'):
                    gradient, selected_nodes, selected_edges, score_pred = decoder.decode(instance, oracle_len)
                    plus_feats, oracle_nodes, oracle_edges, score_true = decoder.oracle(instance)
                    
                    curr_loss = score_pred - score_true  # @UnusedVariable
                    gradient -= plus_feats
                    decoder.weights -= eta * gradient
                
                # ramp loss + cost-augmented decoding
                if loss_func.startswith('ramp'):                         
                    node_cost, edge_cost = node_cost_scaling, edge_cost_scaling
                    gradient, _, _, score_plus_cost = decoder.decode(instance, oracle_len, node_cost, edge_cost)
                    plus_feats, selected_nodes, selected_edges, score_minus_cost = decoder.decode(instance, oracle_len, -1.0 * node_cost, -1.0 * edge_cost)
                    _, oracle_nodes, oracle_edges, score_true = decoder.oracle(instance)
                
                    curr_loss = score_plus_cost - score_minus_cost  # @UnusedVariable
                    gradient -= plus_feats
                    
                    # scale parameter vector and update parameters with SGD
                    if l2reg != 0.0: decoder.weights -= decoder.weights * (eta * l2reg)
                    decoder.weights -= eta * gradient
                
                # hinge-loss + cost-augmented decoding
                if loss_func.startswith('hinge'): 
                    node_cost, edge_cost = node_cost_scaling, edge_cost_scaling
                    gradient, selected_nodes, selected_edges, score_plus_cost = decoder.decode(instance, oracle_len, node_cost, edge_cost)
                    plus_feats, oracle_nodes, oracle_edges, score_true = decoder.oracle(instance)
    
                    curr_loss = score_plus_cost - score_true  # @UnusedVariable
                    gradient -= plus_feats

                    # scale parameter vector and update parameters with SGD
                    if l2reg != 0.0: decoder.weights -= decoder.weights * (eta * l2reg)
                    decoder.weights -= eta * gradient
                
                # use gold nodes and edges to calculate P/R/F
                num_gold_nodes, num_gold_edges = instance.gold
                # P/R/F scores of nodes and edges, for current instance
                # Edge recall can not reach %100 since decoding produces only tree structure
                intersect_nodes = set(selected_nodes) & set(oracle_nodes)
                curr_node_perf = getPRFScores(len(intersect_nodes), len(selected_nodes), num_gold_nodes)
                logPRFScores('train_node', curr_node_perf)
                
                intersect_edges = set(selected_edges) & set(oracle_edges)
                curr_edge_perf = getPRFScores(len(intersect_edges), len(selected_edges), num_gold_edges)
                logPRFScores('train_edge', curr_edge_perf)

                # P/R/F scores of nodes and edges, averaged across all curr_instances
                node_perf = PerfScore(*[sum(x) for x in zip(node_perf, curr_node_perf)])
                edge_perf = PerfScore(*[sum(x) for x in zip(edge_perf, curr_edge_perf)])
                
                logPRFScores('train_node_avg', 
                             PerfScore(node_perf.prec/curr_instances, node_perf.rec/curr_instances, 
                                       node_perf.fscore/curr_instances))              
                logPRFScores('train_edge_avg', 
                             PerfScore(edge_perf.prec/curr_instances, edge_perf.rec/curr_instances, 
                                       edge_perf.fscore/curr_instances))
                                
            # averaging weight vectors
            avg_weights += decoder.weights
            
            # output averaged weight vectors to file
            curr_weights = FeatureVector()
            curr_weights += avg_weights * (1/curr_num_passes)
            if param_file:
                with codecs.open(param_file, 'w', 'utf-8') as outfile:
                    outfile.write('#curr_num_passes#: %d\n' % curr_num_passes)
                    outfile.write('%s\n' % curr_weights.toString())

        final_weights = FeatureVector()
        final_weights += avg_weights * (1/num_passes)
        return final_weights

    def predict(self, decoder, corpus, oracle_len):
        """
        structured prediction on test corpus
        """        
        logger.debug('start prediction...')
        
        node_perf = PerfScore(0.0, 0.0, 0.0) # node performance
        edge_perf = PerfScore(0.0, 0.0, 0.0) # edge performance
        
        curr_instances = 0

        for instance in corpus:
            
            curr_instances += 1
            logger.debug('processing instance %d...' % curr_instances)
            
            _, selected_nodes, selected_edges, _ = decoder.decode(instance, oracle_len)
            _, oracle_nodes, oracle_edges, _ = decoder.oracle(instance)
            
            # use number of gold nodes and edges for P/R/F
            num_gold_nodes, num_gold_edges = instance.gold
            # P/R/F scores of nodes and edges, for current instance
            # Edge recall can not reach %100 since decoding produces only tree structure
            intersect_nodes = set(selected_nodes) & set(oracle_nodes)
            curr_node_perf = getPRFScores(len(intersect_nodes), len(selected_nodes), num_gold_nodes)
            logPRFScores('test_node', curr_node_perf)
            
            intersect_edges = set(selected_edges) & set(oracle_edges)
            curr_edge_perf = getPRFScores(len(intersect_edges), len(selected_edges), num_gold_edges)
            logPRFScores('test_edge', curr_edge_perf)

            # P/R/F scores of nodes and edges, averaged across all curr_instances
            node_perf = PerfScore(*[sum(x) for x in zip(node_perf, curr_node_perf)])
            edge_perf = PerfScore(*[sum(x) for x in zip(edge_perf, curr_edge_perf)])
            
            logPRFScores('test_node_avg', 
                         PerfScore(node_perf.prec/curr_instances, node_perf.rec/curr_instances, 
                                   node_perf.fscore/curr_instances))              
            logPRFScores('test_edge_avg', 
                         PerfScore(edge_perf.prec/curr_instances, edge_perf.rec/curr_instances, 
                                   edge_perf.fscore/curr_instances))
                
        return

    def summarize(self, decoder, corpus, oracle_len, output_folder):
        """
        structured prediction on test corpus
        """        
        logger.debug('start summarization...')
        curr_instances = 0
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, mode=0755)

        for instance in corpus:
            
            curr_instances += 1
            logger.debug('processing instance %d...' % curr_instances)
            
            curr_filename = instance.filename
            my_nodes, oracle_nodes, _ = instance.nodes # nodes and selected nodes            
            _, selected_nodes, _, _ = decoder.decode(instance, oracle_len)
            
            # collect system summary
            system_summ = []
            for idx in selected_nodes:
                curr_node = my_nodes[idx]
                # outfile.write('[concept]: %s\n' % curr_node.concept)
                word_spans = Counter()
                for source in curr_node.sources:
                    if source.word_str == '': continue
                    word_str = (source.word_str).lower()
                    word_spans[word_str] = word_spans.get(word_str, 0.0) + 1
                sorted_word_spans = sorted(word_spans.iteritems(), 
                                           key=lambda x: (x[1], len(x[0])), reverse=True)
                if sorted_word_spans:
                    word_str = sorted_word_spans[0][0]
                    system_summ.append(word_str)
            
            # output system summary
            output_filename = os.path.join(output_folder, curr_filename + '_system')
            with codecs.open(output_filename, 'w', 'utf-8') as outfile:
                outfile.write('%s\n' % ' '.join(system_summ))
            
            # collect oracle summary
            oracle_summ = []
            for idx in oracle_nodes:
                curr_node = my_nodes[idx]
                # outfile.write('[concept]: %s\n' % curr_node.concept)
                word_spans = {}
                for source in curr_node.sources:
                    if source.word_str == '': continue
                    word_str = (source.word_str).lower()
                    word_spans[word_str] = word_spans.get(word_str, 0.0) + 1
                sorted_word_spans = sorted(word_spans.iteritems(), 
                                           key=lambda x: (x[1], len(x[0])), reverse=True)
                if sorted_word_spans:
                    word_str = sorted_word_spans[0][0]
                    oracle_summ.append(word_str)
            
            # output oracle summary
            oracle_filename = os.path.join(output_folder, curr_filename + '_oracle')
            with codecs.open(oracle_filename, 'w', 'utf-8') as outfile:
                outfile.write('%s\n' % ' '.join(oracle_summ))
        return


if __name__ == '__main__':
    pass
    
    
    