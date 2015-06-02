#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs

from collections import namedtuple
from fei.model.amr_graph import AmrEdge
from fei.model.amr_graph import AmrGraph
from fei.model.amr_graph import NodeSource, EdgeSource
from fei.model.utils import getLogger

logger = getLogger()

Instance = namedtuple('Instance', 'filename, nodes, edges, gold')

def buildCorpusAndWriteToFile(body_file, summ_file, w_exp, output_file):
    """
    build corpus and write it to file
    """
    corpus = buildCorpus(body_file, summ_file, w_exp)
    
    total_s_nodes = 0
    total_s_edges = 0
    
    with codecs.open(output_file, 'w', 'utf-8') as outfile:
        for inst in corpus:
            my_nodes, s_nodes, r_nodes = inst.nodes  # @UnusedVariable
            my_edges, s_edges = inst.edges
            curr_filename = inst.filename
            
            total_s_nodes += len(s_nodes)
            total_s_edges += len(s_edges)
            
            outfile.write('%s\n' % curr_filename)
            for k_node, v_node in my_nodes.iteritems():
                tag = 0
                if k_node in s_nodes:
                    tag = 1
                outfile.write('%d %s %s\n' % (tag, k_node, v_node))
            
            for k_edge, v_edge in my_edges.iteritems():
                tag = 0
                if k_edge in s_edges:
                    tag = 1
                outfile.write('%d %s %s\n' % (tag, k_edge, v_edge))
                
    # total selected nodes and edges
    logger.debug('[total_s_nodes]: %d' % total_s_nodes)
    logger.debug('[total_s_edges]: %d' % total_s_edges)
    return

def buildCorpus(body_file, summ_file, w_exp=False):
    """
    build corpus from body and summary files
    """
    logger.debug('building corpus [body file]: %s' % body_file)
    logger.debug('building corpus [summ file]: %s' % summ_file)
    
    corpus = []
    body_corpus = loadFile(body_file)
    summ_corpus = loadFile(summ_file)
    
    for curr_filename in body_corpus:
        # sanity check
        if curr_filename not in summ_corpus:
            logger.error('[no summary sentences]: %s' % curr_filename)
            continue
        
        # nodes: (concept,) -> AmrNode
        # root_nodes: (concept,) -> AmrNode
        # edges: (concept1, concept2) -> AmrEdge
        # exp_edges: (concept1, concept2) -> AmrEdge
        body_nodes, body_root_nodes, body_edges, body_exp_edges = body_corpus[curr_filename]
        summ_nodes, _, summ_edges, _ = summ_corpus[curr_filename]
        num_summ_nodes = len(summ_nodes)
        num_summ_edges = len(summ_edges)
        
        node_idx = 0
        node_indices = {}
        
        my_nodes = {}    # my_nodes: (1,) -> AmrNode
        s_nodes = set() # s_nodes: (1,), (3,), ...
        r_nodes = set() # r_ndoes: (1,), (2,), ...
        
        for anchor, node in body_nodes.iteritems():
            node_idx += 1
            my_nodes[(node_idx,)] = node
            node_indices[anchor[0]] = node_idx
            # node is selected if concept appears in summary
            # node is root node if it appears as root of ANY sentence
            if anchor in summ_nodes: s_nodes.add((node_idx,))
            if anchor in body_root_nodes: r_nodes.add((node_idx,))
        
        my_edges = {}    # my_edges: (1,2) -> AmrEdge
        s_edges = set() # s_edges: (1,2), (3,5), ...
        
        if w_exp: # with edge expansion
            for anchor, edge in body_exp_edges.iteritems():
                idx1 = node_indices[anchor[0]]
                idx2 = node_indices[anchor[1]]
                my_edges[(idx1, idx2)] = edge
                # edge is selected if concept pair appears in summary
                if anchor in summ_edges: s_edges.add((idx1, idx2))
                
        else: # without edge expansion
            for anchor, edge in body_edges.iteritems():
                idx1 = node_indices[anchor[0]]
                idx2 = node_indices[anchor[1]]
                my_edges[(idx1, idx2)] = edge
                if anchor in summ_edges: s_edges.add((idx1, idx2))
                
#             for anchor, edge in body_exp_edges.iteritems():
#                 idx1 = node_indices[anchor[0]]
#                 idx2 = node_indices[anchor[1]]
#                 # edge is selected if concept pair appears in summary
#                 # selected edges are the same with or without edge expansion
#                 if anchor in summ_edges: s_edges.add((idx1, idx2))
                
        inst = Instance(curr_filename, (my_nodes, s_nodes, r_nodes), (my_edges, s_edges), 
                        (num_summ_nodes, num_summ_edges))
        corpus.append(inst)
        
    # return list of Instances
    return corpus

    
def loadFile(input_filename):
    """
    load AMR parsed file, re-index AMR nodes and edges.
    return corpus of nodes and edges
    """
    graph_str = ''  # AMR parse
    info_dict = {}  # AMR meta info
    
    doc_filename = ''
    corpus = {}     # filename -> (nodes, root_nodes, edges, exp_edges)
    
    doc_nodes = {}  # (concept,) -> AmrNode
    doc_root_nodes = {}  # (concept,) -> AmrNode
    doc_edges = {}  # (concept1, concept2) -> AmrEdge
    doc_exp_edges = {} # (concept1, concept2) -> AmrEdge
    
    with codecs.open(input_filename, 'r', 'utf-8') as infile:
        for line in infile:
            line = line.rstrip()
            
            if line == '':
                # no AMR graph for current sentence
                if graph_str == '': 
                    info_dict = {}
                    continue
                
                # get nodes and edges (linked)
                g = AmrGraph()
                nodes, edges = g.getCollapsedNodesAndEdges(graph_str.split())
                
                # index nodes by graph_idx
                node_indices = {}
                for node in nodes:
                    graph_idx = node.graph_idx
                    node_indices.setdefault(graph_idx, node)

                # (1) use gold AMR annotation as input
                if not 'alignments' in info_dict:
                    
                    # get sentence info
                    sentence = info_dict['snt'] # tokenized sentence
                    filename, line_num = info_dict['id'].split('.')
                    line_num = int(line_num)
                    
                    # add source info to nodes
                    for node in nodes:
                        node_source = NodeSource(node.graph_idx, 0, 0, '', filename, line_num, sentence)
                        node.sources.append(node_source)
                        
                    # add source info to edges
                    for edge in edges:
                        edge_source = EdgeSource(edge.relation, filename, line_num, sentence)
                        edge.sources.append(edge_source)
                        
                else: # (2) use alignment file as input
                    
                    # get sentence info
                    sentence = info_dict['tok'] # tokenized sentence
                    tokens = sentence.split()
                    filename, line_num = info_dict['id'].split('.')
                    line_num = int(line_num)
                    
                    # add source info to edges
                    for edge in edges:
                        edge_source = EdgeSource(edge.relation, filename, line_num, sentence)
                        edge.sources.append(edge_source)
                    
                    # add alignment and other source info to nodes
                    alignments_str = info_dict['alignments']
                    
                    for alignment in alignments_str.split():
                        word_part, graph_part = alignment.split('|')
                        start_idx, end_idx = map(int, word_part.split('-'))
                        graph_indices = graph_part.split('+')
                        
                        for graph_idx in graph_indices:
                            curr_node = node_indices.get(graph_idx, None)
                            if curr_node is None: continue
                            
                            # add node source info
                            new_start_idx = start_idx
                            new_end_idx = end_idx
                            # change existing start_idx/end_idx to broadest coverage
                            if curr_node.sources: 
                                curr_node_source = curr_node.sources.pop()
                                if new_start_idx > curr_node_source.start_idx:
                                    new_start_idx = curr_node_source.start_idx
                                if new_end_idx < curr_node_source.end_idx:
                                    new_end_idx = curr_node_source.end_idx
                            # update new node source
                            new_node_source = NodeSource(curr_node.graph_idx, new_start_idx, new_end_idx, 
                                                         ' '.join(tokens[new_start_idx:new_end_idx]), 
                                                         filename, line_num, sentence)
                            curr_node.sources.append(new_node_source)

                    # add source info to [unaligned] nodes 
                    for node in nodes:
                        if node.sources: continue
                        node_source = NodeSource(node.graph_idx, 0, 0, '', filename, line_num, sentence)
                        node.sources.append(node_source)

                # start of new file
                if filename != doc_filename:
                    if doc_filename != '':
                        corpus[doc_filename] = (doc_nodes, doc_root_nodes, doc_edges, doc_exp_edges)
                    doc_filename = filename
                    doc_nodes = {}
                    doc_root_nodes = {}
                    doc_edges = {}
                    doc_exp_edges = {}

                # keep track of redirected nodes
                redirect_dict = {}
                
                # merge nodes
                first_node = True
                for node in nodes:
                    curr_anchor = tuple((node.concept,)) # tricky tuple
                    if curr_anchor in doc_nodes:
                        old_node = doc_nodes[curr_anchor]
                        old_node.sources.extend(node.sources)
                        redirect_dict[node] = old_node
                    else:
                        doc_nodes[curr_anchor] = node
                    # root node of sentence
                    if first_node == True:
                        doc_root_nodes[curr_anchor] = doc_nodes[curr_anchor]
                        first_node = False
                
                # merge edges
                edge_indices = {} # index edge by concepts
                for edge in edges:
                    # update node linkage
                    if edge.node1 in redirect_dict:
                        edge.node1 = redirect_dict[edge.node1]
                    if edge.node2 in redirect_dict:
                        edge.node2 = redirect_dict[edge.node2]
                        
                    curr_anchor = tuple((edge.node1.concept, edge.node2.concept)) # ignore relation
                    edge_indices[curr_anchor] = edge
                    
                    if curr_anchor in doc_edges:
                        old_edge = doc_edges[curr_anchor]
                        old_edge.sources.extend(edge.sources)
                    else:
                        doc_edges[curr_anchor] = edge
                
                # expand edges, nodes in each sentence are fully connected
                for node1 in nodes:
                    for node2 in nodes:
                        curr_anchor = tuple((node1.concept, node2.concept))
                        redirect_node1 = doc_nodes[(node1.concept,)]
                        redirect_node2 = doc_nodes[(node2.concept,)]
                        
                        # expanded edge exists
                        if curr_anchor in doc_exp_edges:
                            # update node linkage
                            old_edge = doc_exp_edges[curr_anchor]
                            old_edge.node1 = redirect_node1
                            old_edge.node2 = redirect_node2
                            # update edge sources
                            if curr_anchor in edge_indices: # true edge
                                edge = edge_indices[curr_anchor]
                                old_edge.sources.extend(edge.sources)
                            else: # NULL edge
                                edge_source = EdgeSource('NULL', filename, line_num, sentence)
                                old_edge.sources.append(edge_source)
                                
                        else: # expanded edge does not exist, build a new edge
                            if curr_anchor in edge_indices: # true edge
                                edge = edge_indices[curr_anchor]
                                new_edge = AmrEdge(node1=redirect_node1, node2=redirect_node2, relation=edge.relation)
                                new_edge.sources.extend(edge.sources)
                            else: # NULL edge
                                new_edge = AmrEdge(node1=redirect_node1, node2=redirect_node2, relation='NULL')
                                edge_source = EdgeSource('NULL', filename, line_num, sentence)
                                new_edge.sources.append(edge_source)
                            doc_exp_edges[curr_anchor] = new_edge

                # clear cache
                graph_str = ''
                info_dict = {}
                continue
            
            if line.startswith('#'):
                fields = line.split('::')
                for field in fields[1:]:
                    tokens = field.split()
                    info_name = tokens[0]
                    info_body = ' '.join(tokens[1:])
                    info_dict[info_name] = info_body
                continue
            
            graph_str += line
            
    # add nodes and edges to the last file
    corpus[doc_filename] = (doc_nodes, doc_root_nodes, doc_edges, doc_exp_edges)
    # return loaded corpus
    return corpus    

if __name__ == '__main__':
    
    input_dir = '/Users/user/Data/SemanticSumm/Proxy/gold/split/dev'
    
    body_file = 'aligned-amr-release-1.0-dev-proxy-body.txt'
    summ_file = 'aligned-amr-release-1.0-dev-proxy-summary.txt'
    
    buildCorpusAndWriteToFile(os.path.join(input_dir, body_file),
                              os.path.join(input_dir, summ_file),
                              w_exp=True, output_file='output_file')
        
    
    
    




    
