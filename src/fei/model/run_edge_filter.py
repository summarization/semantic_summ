#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import codecs
import os
import logging
import re
import itertools

from collections import defaultdict
from fei.model.edge_filter import EdgeFilter
from fei.backup.amr_graph import AmrEdge, AmrNode
from fei.backup.amr_graph import CompareAmrEdge, CompareAmrEdgeWoRel

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def getEdgesFromFile(input_file, comparator):
    """
    load AMR graphs from input file, return edges.
    """
    edges = []
    with codecs.open(input_file, 'r', 'utf-8') as infile:
        lines = [] # TODO: temporary solution              

        base_filename = os.path.basename(input_file)
        line_num = 0
        tokens = []
        node_indices = {}
        flag_collect_nodes = False
        flag_collect_edges = False
        
        for line in infile:
            line = line.strip()
            lines.append(line)
            
            if line == '':
                tokens = []
                node_indices = {}
                flag_collect_nodes = False
                flag_collect_edges = False
                continue
            
            # increase line_num, starts from 1
            # TODO: temporary solution
            if line.startswith('# ::snt'):
                prev_line = lines[-2] if len(lines) >= 2 else ''
                if not prev_line.startswith('Nodes:'):
                    line_num += 1
                continue
            
            # get tokens
            if line.startswith('# ::tok'):
                tokens = line.split()[2:] # drop "# ::tok"
                continue
            
            if line.startswith('Nodes:'):
                flag_collect_nodes = True
                continue
            
            if line.startswith('Edges:'):
                flag_collect_nodes = False
                flag_collect_edges = True
                continue
            
            # get AMR nodes
            if flag_collect_nodes == True and line.startswith('0'):
                try:
                    graphIdx, shortHnd, concept, wordIdx = line.split('\t')
                except ValueError:
                    logger.debug('cannot split line: %s' % line)
                    
                startIdx, endIdx = map(int, wordIdx.split('-'))
                phrase = ' '.join(tokens[startIdx:endIdx])
                curr_node = AmrNode(graphIdx, shortHnd, concept, 
                                    startIdx, endIdx, phrase,
                                    base_filename, line_num, tokens)
                node_indices.setdefault(graphIdx, curr_node)
            
            # get AMR edges
            if flag_collect_edges == True and line.startswith('0'):
                try:
                    graphIdx1, _, relation, graphIdx2, _ = line.split('\t')
                except ValueError:
                    logger.debug('cannot split line: %s' % line)
                    
                node1 = node_indices.get(graphIdx1)
                node2 = node_indices.get(graphIdx2)
                if node1 is None or node2 is None:
                    logger.error('node indices do not align with edge indices: %s' % ' '.join(tokens))
                # ignore edges where two concepts correspond to same words
                if node1.word_str == node2.word_str: continue
                edges.append(comparator(AmrEdge(node1, node2, relation)))

    return edges

def _getEdgesIter(input_path, comparator):
    """
    """
    logger.debug('generate edges from: %s' % input_path)
    
    edges = defaultdict(list)
    if not os.path.exists(input_path): 
        return edges
    
    if os.path.isfile(input_path):
        base_filename = os.path.basename(input_path)
        topic = re.sub(r'-.*$', '', base_filename)
        edges[topic].extends(getEdgesFromFile(input_path, comparator))
    
    if os.path.isdir(input_path):
        for base_filename in os.listdir(input_path):
            topic = re.sub(r'-.*$', '', base_filename)
            input_file = os.path.join(input_path, base_filename)
            edges[topic].extend(getEdgesFromFile(input_file, comparator))
    
    return edges

def getEdgesAndLabels(docs_dir, models_dir, comparator):
    """
    unique edges and labels
    """
    edges = []
    labels = []

    docs_edges = _getEdgesIter(docs_dir, comparator)
    models_edges = _getEdgesIter(models_dir, comparator)
    
    for topic in docs_edges:
        curr_docs_edges = set(docs_edges[topic]) # unique edges
        curr_models_edges = set(models_edges[topic])
        
        for edge in curr_docs_edges:
            label = 1 if edge in curr_models_edges else 0
            edges.append(edge.edge) # bare edge
            labels.append(label) # gold label
            
    return edges, labels

def writeEdgesAndLabels(edges, labels, output_file):
    """
    write train/test instances to file
    """
    with codecs.open(output_file, 'w', 'utf-8') as outfile:
        for edge, label in itertools.izip(edges, labels):
            outfile.write('%s\t%s\n' % (label, edge))
    return

def writeCoverageStats(docs_dir, models_dir, comparator, output_file):
    """
    write coverage statistics to file:
    (1) percentage of unique document edges that were covered by summaries
    (2) percentage of unique summary edges that were covered by documents
    """
    docs_edges = _getEdgesIter(docs_dir, comparator)
    models_edges = _getEdgesIter(models_dir, comparator)
    
    total_docs_edges = 0
    total_models_edges = 0
    covered_docs_edges = 0
    covered_models_edges = 0

    # micro-average across all topics
    for topic in docs_edges:
        curr_docs_edges = set(docs_edges[topic])
        curr_models_edges = set(models_edges[topic])
        
        total_docs_edges += len(curr_docs_edges)
        total_models_edges += len(curr_models_edges)
        
        for edge in curr_docs_edges:
            if edge in curr_models_edges:
                covered_docs_edges += 1
                
        for edge in curr_models_edges:
            if edge in curr_docs_edges:
                covered_models_edges += 1
    
    with codecs.open(output_file, 'w', 'utf-8') as outfile:
        outfile.write('total_docs_edges: %d\n' % total_docs_edges)
        outfile.write('covered_docs_edges: %d (%.0f%%)\n' 
                      % (covered_docs_edges, covered_docs_edges*100/total_docs_edges))
        
        outfile.write('total_models_edges: %d\n' % total_models_edges)
        outfile.write('covered_models_edges: %d (%.0f%%)\n' 
                      % (covered_models_edges, covered_models_edges*100/total_models_edges))
    return

if __name__ == '__main__':
    input_dir = '/home/user/TAC_Fei'
    output_dir = '/home/user/Runs/Experiments/EdgeFilter'
    comparator = CompareAmrEdgeWoRel
    
    train_edges = []
    train_labels = []
    
    for year in ['2009', '2010', '2011']:
        models_dir = os.path.join(input_dir, year, 'models_jamr')
        docs_dir = os.path.join(input_dir, year, 'docs_jamr')
        edges, labels = getEdgesAndLabels(docs_dir, models_dir, comparator)
        assert len(edges) == len(labels)
        train_edges.extend(edges)
        train_labels.extend(labels)
    
    models_dir = os.path.join(input_dir, '2008', 'models_jamr')
    docs_dir = os.path.join(input_dir, '2008', 'docs_jamr')
    test_edges, test_labels = getEdgesAndLabels(docs_dir, models_dir, comparator)
    
    # output train_file, test_file, and coverage stats
    output_file_train = os.path.join(output_dir, 'train_file')
    output_file_test = os.path.join(output_dir, 'test_file')
    writeEdgesAndLabels(train_edges, train_labels, output_file_train)
    writeEdgesAndLabels(test_edges, test_labels, output_file_test)
    
    output_file_coverage = os.path.join(output_dir, 'coverage')
    writeCoverageStats(docs_dir, models_dir, comparator, output_file_coverage)

    # run LR classifier
    myfilter = EdgeFilter()
    train_iter = myfilter.data_iter(train_edges, train_labels, add_feat=True)
    test_iter = myfilter.data_iter(test_edges, test_labels, add_feat=False)
     
    myfilter.trainLR(train_iter, l1=0.5)
    myfilter.testLR(test_iter)
    myfilter.writeFiles(output_dir)
    

    
    






