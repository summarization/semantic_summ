#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
To accommodate modified JAMR output format
"""

from __future__ import division
import os
import re
import codecs
import logging

from fei.backup.amr_graph import AmrNode
from fei.preprocess.triple import Triple
from fei.backup.stats_coverage import getCoverageStats
from fei.backup.stats_coverage import getUncoveredTriples
from collections import defaultdict
from collections import Counter as mset

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def getTriples(input_file):
    """
    get triples from JAMR parsed file
    """
    triples = []

    lines = [] # TODO: temporary solution              
    line_num = -1 # start from 0
    base_filename = os.path.basename(input_file)
    tokens = []
    node_indices = {}
    flag_collect_nodes = False
    flag_collect_edges = False

    with codecs.open(input_file, 'r', 'utf-8') as infile:        
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
                t = Triple(node1.concept, node2.concept, relation, 
                           base_filename, line_num, ' '.join(tokens))
                triples.append(t)

    return triples

def getTriplesFromDocsAndModels(input_dir, dirname):
    """
    return triples from docs and models
    TODO: should return concepts as well
    """
    
    docs_parsed_dir = os.path.join(input_dir, 'docs_' + dirname)
    models_parsed_dir = os.path.join(input_dir, 'models_' + dirname)
    
    models_triples = defaultdict(list)
    docs_triples = defaultdict(list)
    
    # obtain triples from documents
    # use 20 documents per topic (NO distinction between A and B)
    for filename in os.listdir(docs_parsed_dir):
        topic = re.sub(r'-.*$', '', filename)  # "D0848"
        
        input_file = os.path.join(docs_parsed_dir, filename)
        triples = getTriples(input_file)
        docs_triples[topic].extend(triples)
                
    # obtain triples from summaries
    # use 4 summaries per topic (EXISTS distinction between A and B)
    for filename in os.listdir(models_parsed_dir):
        topic = re.sub(r'\..*$', '', filename)  # "D0848-A"
        
        input_file = os.path.join(models_parsed_dir, filename)
        triples = getTriples(input_file)
        models_triples[topic].extend(triples)
        
    return models_triples, docs_triples

def logUncoveredSummaryTriples(input_dir, dirname, output_file):
    """
    log uncovered triples to files
    """
    models_triples, docs_triples = getTriplesFromDocsAndModels(input_dir, dirname)
    uncovered_triples = getUncoveredTriples(models_triples, docs_triples)
    
    with codecs.open(output_file, 'w', 'utf-8') as outfile:
        for t in uncovered_triples:
            outfile.write('[triple] %s (%s)-> %s\n' % (t.concept1, t.relation, t.concept2))
            outfile.write('[tokens] %s\n' % (t.sentence))
            outfile.write('[filename] %s [line_num] %d\n\n' % (t.filename, t.line_num))
    return

if __name__ == '__main__':
    input_dir = '/home/user/TAC_Fei'
    dirname = 'jamr'
    output_dir = '/home/user/Runs/Experiments/UncoveredSummaryEdges/Jamr'
    
    for year in ['2008', '2009', '2010', '2011']:
        curr_dir = os.path.join(input_dir, year)
        output_file = os.path.join(output_dir, year)
        logUncoveredSummaryTriples(curr_dir, dirname, output_file)





