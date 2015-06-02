#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import os

from fei.model.corpus import buildCorpus, loadFile

# input_dir = '/Users/user/Data/SemanticSumm/Proxy/gold/split/test'
# body_file = 'aligned-amr-release-1.0-test-proxy-body.txt'
# summ_file = 'aligned-amr-release-1.0-test-proxy-summary.txt'

# input_dir = '/Users/user/Data/SemanticSumm/Proxy/gold/split/dev'
# body_file = 'aligned-amr-release-1.0-dev-proxy-body.txt'
# summ_file = 'aligned-amr-release-1.0-dev-proxy-summary.txt'

input_dir = '/Users/user/Data/SemanticSumm/Proxy/gold/split/training'
body_file = 'aligned-amr-release-1.0-training-proxy-body.txt'
summ_file = 'aligned-amr-release-1.0-training-proxy-summary.txt'


def getDocStats(input_dir, body_file, summ_file):
    """
    """
    corpus = buildCorpus(os.path.join(input_dir, body_file), 
                         os.path.join(input_dir, summ_file),
                         w_exp=True)
    
    num_docs = 0
    total_nodes = 0
    total_edges = 0
    selected_nodes = 0
    selected_edges = 0
    
    for inst in corpus:
        num_docs += 1
        my_nodes, oracle_nodes, _ = inst.nodes
        my_edges, oracle_edges = inst.edges
        
        total_nodes += len(my_nodes)
        total_edges += len(my_edges)
        
        selected_nodes += len(oracle_nodes)
        selected_edges += len(oracle_edges)
    
    print 'avg nodes: %.1f' % (total_nodes/num_docs)
    print 'avg edges: %.1f' % (total_edges/num_docs)
    
    print 'selected nodes: %.1f' % (selected_nodes/num_docs)
    print 'selected edges: %.1f' % (selected_edges/num_docs)


def getFileStats(input_dir, input_filename):
    """
    """
    num_docs = 0
    num_nodes = 0
    num_edges = 0
    
    corpus = loadFile(os.path.join(input_dir, input_filename))
    
    for curr_filename in corpus:
        if curr_filename == 'PROXY_AFP_ENG_20030126_0212': continue
        
        num_docs += 1
        nodes, _, edges, _ = corpus[curr_filename]        
        num_nodes += len(nodes)
        num_edges += len(edges)

    num_nodes /= num_docs
    num_edges /= num_docs
    
    result = '''
    ------
    number of total files: %d
    number of unique nodes per document: %.1f
    number of unique edges per document: %.1f
    ''' % (num_docs, num_nodes, num_edges)

    print input_filename
    print result
    return result
    

getDocStats(input_dir, body_file, summ_file)
# getFileStats(input_dir, summ_file)


