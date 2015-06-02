#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs

from fei.model.amr_graph import AmrGraph
from fei.preprocess.triple import Triple
from fei.preprocess.stats_coverage import getCoverageStats
from fei.preprocess.stats_coverage import getConceptStats
from collections import defaultdict, Counter

def getTriples(input_file):
    """
    get triples from gold-standard AMR annotations
    """
    triples = defaultdict(list)
    extended_triples = defaultdict(list)
    concepts = defaultdict(Counter) # Counter
    
    graph_string = ''   # AMR graph
    tokens = ''       # current tokens
    line_num = -1       # line_num in file
    filename = ''       # current filename
    
    with codecs.open(input_file, 'r', 'utf-8') as infile:
        for line in infile:
            line = line.rstrip()
            
            # process each graph
            if line == '' and graph_string:
                g = AmrGraph()
                graph_tokens = graph_string.split()
    
                nodes, edges = g.getCollapsedNodesAndEdges(graph_tokens) 
#                 nodes = g.getNodes(graph_tokens)
#                 edges = g.getEdges(graph_tokens)
                
                # get triples
                for edge in edges:
                    node1 = edge.node1
                    node2 = edge.node2
                    relation = edge.relation
                    
                    t = Triple(node1.concept, node2.concept, relation, filename, line_num, tokens)
                    triples[filename].append(t)

                # get extended triples (ignore relation, but keep direction)
                for node1 in nodes:
                    for node2 in nodes:
                        t = Triple(node1.concept, node2.concept, '', filename, line_num, tokens)
                        extended_triples[filename].append(t)
                
                # get concepts
                for node in nodes:
                    concepts[filename][node.concept] += 1
            
                # clear cache
                graph_string = ''
                filename = ''
                line_num = -1
                tokens = ''
                continue
            
            if line.startswith('# ::id'):
                tokens = line.split()
                filename, line_num = tokens[2].split('.')
                line_num = int(line_num)
                continue
            
            # get snt-type: summary, body, etc.
            if line.startswith('# ::snt'):
                tokens = line
                continue
            
            # get AMR graph string
            if line and not line.startswith('#'):
                graph_string += line
                continue
            
    return triples, extended_triples, concepts

def countTriples(input_dir, uniq=True):
    """
    """
    models_triples = None
    models_concepts = None
    docs_triples = None
    docs_extended_triples = None
    docs_concepts = None
    
    for curr_filename in os.listdir(input_dir):
        if curr_filename.startswith('aligned'): continue
        
        if 'proxy-summary.txt' in curr_filename:
            summary_file = os.path.join(input_dir, curr_filename)
            models_triples, _, models_concepts = getTriples(summary_file)
        
        if 'proxy-body.txt' in curr_filename:
            body_file = os.path.join(input_dir, curr_filename)
            docs_triples, docs_extended_triples, docs_concepts = getTriples(body_file)
        
    # get coverage statistics
    result1 = getCoverageStats(models_triples, docs_triples, 
                              docs_extended_triples, docs_concepts)
    result2 = getConceptStats(models_concepts, docs_concepts)

    print input_dir
    print result1
    print result2
    return

if __name__ == '__main__':

    # separate summary sentences from body sentences
#     input_dir = '/home/user/Data/Proxy/gold/split'
    input_dir = '/Users/user/Data/SemanticSumm/Proxy/gold/split'
    
    # loop through training/dev/test folders
    for curr_dirname in os.listdir(input_dir):
        curr_dir = os.path.join(input_dir, curr_dirname)
        if not os.path.isdir(curr_dir): continue
        countTriples(curr_dir) # split proxy file

    
#     input_dir = '/home/user/Data/Proxy/gold/unsplit'
#     countTriples(input_dir)
    












    
