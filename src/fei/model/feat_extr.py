#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import os
import codecs

from collections import Counter
from fei.model.feat_vec import FeatureVector
from fei.model.corpus import buildCorpus
from fei.model.utils import getLogger

logger = getLogger()


# TODO: weights of concepts calculated on TAC datasets??

class FeatureExtractor(object):
    """
    extract features for nodes and edges
    """
    def __init__(self):
        self.node_feat_funcs = [self.ffNodeBias, self.ffNodeFreq, self.ffNodePos, 
                                self.ffNodeConcept, self.ffNodeDepth,
                                self.ffNodeSpan, self.ffNodeCollapsedEntity]
        
        self.edge_feat_funcs = [self.ffEdgeBias, self.ffEdgeFreq, 
                                self.ffEdgeIsNull, self.ffEdgeNonNullFreq, 
                                self.ffEdgeRel, self.ffEdgePos, 
                                self.ffEdgeNodeFreq, self.ffEdgeNodePos, 
                                self.ffEdgeNodeConcept, self.ffEdgeNodeDepth, 
                                self.ffEdgeNodeSpan, self.ffEdgeNodeCollapsedEntity]
        
        self.curr_filename = ''
        self.curr_feats = {}
        
        return
    
    def getNodeFeats(self, k_node, v_node, tag, curr_filename, my_nodes, my_edges):
        """
        extract features for one node.
        cache features before conjoining with node tag.
        k_node: (1,), v_node: AmrNode
        my_nodes: (1,) -> AmrNode, my_edges: (2,1) -> AmrEdge
        """
        feat_vec = FeatureVector()
        
        # node features have been extracted
        if self.curr_filename == curr_filename and k_node in self.curr_feats:
            feat_vec = self.curr_feats[k_node]
        
        else: # new node
            # if this is a new file, clear cache
            if self.curr_filename != curr_filename:
                self.curr_filename = curr_filename
                self.curr_feats = {}
            
            # extract features and add to cache
            for feat_func in self.node_feat_funcs:
                feat_vec += feat_func(v_node)
            self.curr_feats[k_node] = feat_vec
        
        # conjoin features with tag
        new_feat_vec = FeatureVector()
        for k, v in feat_vec.iteritems():
            new_feat_vec[(str(tag),) + k] = v

        # return node features conjoined with tag            
        return new_feat_vec
    
    def getEdgeFeats(self, k_edge, v_edge, tag, curr_filename, my_nodes, my_edges):
        """
        extract features for one edge.
        cache features before conjoining with edge tag.
        k_edge: (1,2), v_edge: AmrEdge
        my_nodes: (1,) -> AmrNode, my_edges: (2,1) -> AmrEdge
        """
        feat_vec = FeatureVector()
        
        # edge features have been extracted
        if self.curr_filename == curr_filename and k_edge in self.curr_feats:
            feat_vec = self.curr_feats[k_edge]
        
        else: # new edge
            # if this is a new file, clear cache
            if self.curr_filename != curr_filename:
                self.curr_filename = curr_filename
                self.curr_feats = {}
            
            # extract features and add to cache
            for feat_func in self.edge_feat_funcs:
                feat_vec += feat_func(v_edge)
            self.curr_feats[k_edge] = feat_vec
        
        # conjoin features with tag
        new_feat_vec = FeatureVector()
        for k, v in feat_vec.iteritems():
            new_feat_vec[(str(tag),) + k] = v

        # return edge features conjoined with tag            
        return new_feat_vec
    
    def ffEdgeBias(self, edge):
        """
        add a bias term to edge
        """
        feat_vec = FeatureVector()
        feat_vec[('e', 'bias')] = 1.0
        return feat_vec
    
    def ffEdgeIsNull(self, edge):
        """
        extract a binary feature indicating a NULL edge or not
        """
        feat_vec = FeatureVector()
        is_null = 1.0
        for source in edge.sources:
            if source.relation != 'NULL':
                is_null = 0.0
                break
        feat_vec[('e', 'is_null')] = is_null
        return feat_vec
    
    def ffEdgeNonNullFreq(self, edge):
        """
        extract a binary feature for edge frequency (Non-NULL edges)
        freq == 0, freq >= 1, freq >= 2, freq >= 5, freq >= 10
        """
        feat_vec = FeatureVector()
        
        edge_freq = 0
        for source in edge.sources:
            if source.relation != 'NULL':
                edge_freq += 1

        # binary feature for edge frequency
        feat_vec[('e', 'freq', 'non_null', '0')] = 1.0 if edge_freq == 0 else 0.0
        feat_vec[('e', 'freq', 'non_null', '1')] = 1.0 if edge_freq >= 1 else 0.0
        feat_vec[('e', 'freq', 'non_null', '2')] = 1.0 if edge_freq >= 2 else 0.0
        feat_vec[('e', 'freq', 'non_null', '5')] = 1.0 if edge_freq >= 5 else 0.0
        feat_vec[('e', 'freq', 'non_null', '10')] = 1.0 if edge_freq >= 10 else 0.0
        
        return feat_vec
    
    def ffEdgeFreq(self, edge):
        """
        extract a binary feature for edge frequency.
        freq == 0, freq >= 1, freq >= 2, freq >= 5, freq >= 10
        """
        feat_vec = FeatureVector()
        
        # edge frequency
        edge_freq = len(edge.sources)
        
        # binary feature for edge frequency
        feat_vec[('e', 'freq', '0')] = 1.0 if edge_freq == 0 else 0.0
        feat_vec[('e', 'freq', '1')] = 1.0 if edge_freq >= 1 else 0.0
        feat_vec[('e', 'freq', '2')] = 1.0 if edge_freq >= 2 else 0.0
        feat_vec[('e', 'freq', '5')] = 1.0 if edge_freq >= 5 else 0.0
        feat_vec[('e', 'freq', '10')] = 1.0 if edge_freq >= 10 else 0.0
        
        return feat_vec
    
    def ffEdgeRel(self, edge): 
        """
        extract a binary feature for edge relation.
        """
        feat_vec = FeatureVector()
        edge_freq = len(edge.sources)
        
        # primary and secondary relation (edge relation entropy)?
        rel_freq = Counter()
        for source in edge.sources:
            rel_freq[source.relation] += 1
        rels = rel_freq.most_common()
        
        if rels: # primary relation
            (rel, count) = rels[0]
            feat_vec[('e', 'rel', 'fst', rel)] = 1.0
            
            # relative frequency
            per_fst_rel = count/edge_freq
            
            feat_vec[('e', 'rel', 'fst', rel, 'p1')] = 1.0 if per_fst_rel >= 0.5 else 0.0
            feat_vec[('e', 'rel', 'fst', rel, 'p2')] = 1.0 if per_fst_rel >= 0.66 else 0.0
            feat_vec[('e', 'rel', 'fst', rel, 'p3')] = 1.0 if per_fst_rel >= 0.75 else 0.0

            # secondary relation and relative frequency
            if len(rels) > 1:
                (sec_rel, sec_count) = rels[1]
                feat_vec[('e', 'rel', 'sec', sec_rel)] = 1.0
                
                # relative frequency
                per_sec_rel = sec_count/edge_freq
                
                feat_vec[('e', 'rel', 'sec', sec_rel, 'p1')] = 1.0 if per_sec_rel >= 0.25 else 0.0
                feat_vec[('e', 'rel', 'sec', sec_rel, 'p2')] = 1.0 if per_sec_rel >= 0.33 else 0.0
                feat_vec[('e', 'rel', 'sec', sec_rel, 'p3')] = 1.0 if per_sec_rel >= 0.5 else 0.0

                # combine first and secondary relation
                feat_vec[('e', 'rel', 'fst', rel, 'sec', sec_rel)] = 1.0
            
        return feat_vec
    
    def ffEdgePos(self, edge):
        """
        extract features from edge occurrences.

        """
        feat_vec = FeatureVector()
        
        # foremost position in all edge occurrences
        # average position across all edge occurrences
        foremost = 1e10
        average = 0.0
        edge_freq = len(edge.sources)
        
        for source in edge.sources:
            average += source.line_num
            if foremost > source.line_num:
                foremost = source.line_num
                
        if edge_freq > 0: 
            average /= edge_freq
        
        if edge_freq > 0: # edge exists
            # foremost occurrence
            feat_vec[('e', 'posit', 'fmst', '5')] = 1.0 if foremost >= 5 else 0.0
            feat_vec[('e', 'posit', 'fmst', '6')] = 1.0 if foremost >= 6 else 0.0
            feat_vec[('e', 'posit', 'fmst', '7')] = 1.0 if foremost >= 7 else 0.0
            feat_vec[('e', 'posit', 'fmst', '10')] = 1.0 if foremost >= 10 else 0.0
            feat_vec[('e', 'posit', 'fmst', '15')] = 1.0 if foremost >= 15 else 0.0
            # average occurrence
            feat_vec[('e', 'posit', 'avg', '5')] = 1.0 if average >= 5 else 0.0
            feat_vec[('e', 'posit', 'avg', '6')] = 1.0 if average >= 6 else 0.0
            feat_vec[('e', 'posit', 'avg', '7')] = 1.0 if average >= 7 else 0.0
            feat_vec[('e', 'posit', 'avg', '10')] = 1.0 if average >= 10 else 0.0
            feat_vec[('e', 'posit', 'avg', '15')] = 1.0 if average >= 15 else 0.0
            
        return feat_vec
            
    def ffEdgeNodeFreq(self, edge):
        """
        extract node frequency features for an edge
        """
        feat_vec = FeatureVector()
        
        node1 = edge.node1
        node2 = edge.node2
        
        for k, v in self.ffNodeFreq(node1).iteritems():
            feat_vec[('e', 'n1') + k] = v
            
        for k, v in self.ffNodeFreq(node2).iteritems():
            feat_vec[('e', 'n2') + k] = v
        
        return feat_vec
    
    def ffEdgeNodeConcept(self, edge):
        """
        extract node concept features for edge
        """
        feat_vec = FeatureVector()
        
        node1 = edge.node1
        node2 = edge.node2
        
        for k, v in self.ffNodeConcept(node1).iteritems():
            feat_vec[('e', 'n1') + k] = v
            
        for k, v in self.ffNodeConcept(node2).iteritems():
            feat_vec[('e', 'n2') + k] = v
            
        return feat_vec

    def ffEdgeNodeDepth(self, edge):
        """
        extract node depth features for edge
        """
        feat_vec = FeatureVector()
        
        node1 = edge.node1
        node2 = edge.node2
        
        for k, v in self.ffNodeDepth(node1).iteritems():
            feat_vec[('e', 'n1') + k] = v
            
        for k, v in self.ffNodeDepth(node2).iteritems():
            feat_vec[('e', 'n2') + k] = v
        
        node1_foremost = 1e10
        node2_foremost = 1e10
        
        for source in node1.sources:
            node_depth = len((source.graph_idx).split('.'))
            if node1_foremost > node_depth:
                node1_foremost = node_depth

        for source in node2.sources:
            node_depth = len((source.graph_idx).split('.'))
            if node2_foremost > node_depth:
                node2_foremost = node_depth
            
        # concatenate foremost occurrence of node1 and node2
        feat_vec[('e', 'n1', 'dep', 'fmst', str(node1_foremost), 'n2', 'dep', 'fmst', str(node2_foremost))] = 1.0
        return feat_vec
    
    
    def ffEdgeNodePos(self, edge):
        """
        extract node position features for edge
        """
        feat_vec = FeatureVector()
        
        node1 = edge.node1
        node2 = edge.node2
        
        for k, v in self.ffNodePos(node1).iteritems():
            feat_vec[('e', 'n1') + k] = v
            
        for k, v in self.ffNodePos(node2).iteritems():
            feat_vec[('e', 'n2') + k] = v

        node1_foremost = 1e10
        node2_foremost = 1e10
        
        for source in node1.sources:
            if node1_foremost > source.line_num:
                node1_foremost = source.line_num

        for source in node2.sources:
            if node2_foremost > source.line_num:
                node2_foremost = source.line_num

        # concatenate foremost occurrence of node1 and node2
        feat_vec[('e', 'n1', 'posit', 'fmst', str(node1_foremost), 'n2', 'posit', 'fmst', str(node2_foremost))] = 1.0
            
        return feat_vec
    
    def ffEdgeNodeCollapsedEntity(self, edge):
        """
        extract features from collapsed concept node
        """
        feat_vec = FeatureVector()

        node1 = edge.node1
        node2 = edge.node2
        
        for k, v in self.ffNodeSpan(node1).iteritems():
            feat_vec[('e', 'n1') + k] = v
            
        for k, v in self.ffNodeSpan(node2).iteritems():
            feat_vec[('e', 'n2') + k] = v
        
        return feat_vec

    def ffEdgeNodeSpan(self, edge):
        """
        extract features from node spans of edge
        """
        feat_vec = FeatureVector()

        node1 = edge.node1
        node2 = edge.node2
        
        for k, v in self.ffNodeSpan(node1).iteritems():
            feat_vec[('e', 'n1') + k] = v
            
        for k, v in self.ffNodeSpan(node2).iteritems():
            feat_vec[('e', 'n2') + k] = v

        node1_longest = -1
        node2_longest = -1
        
        for source in node1.sources:
            node_span = source.end_idx - source.start_idx
            if node1_longest < node_span:
                node1_longest = node_span

        for source in node2.sources:
            node_span = source.end_idx - source.start_idx
            if node2_longest < node_span:
                node2_longest = node_span

        # concatenate foremost occurrence of node1 and node2
        feat_vec[('e', 'n1', 'span', 'lgst', str(node1_longest), 'n2', 'span', 'lgst', str(node2_longest))] = 1.0

        return feat_vec

    
    def ffNodeBias(self, node):
        """
        add a bias term to node
        """
        feat_vec = FeatureVector()
        feat_vec[('n', 'bias')] = 1.0
        return feat_vec

    def ffNodeFreq(self, node):
        """
        extract node frequency features
        """
        feat_vec = FeatureVector()
                
        # node frequency
        node_freq = len(node.sources)
        
        feat_vec[('n', 'freq', '0')] = 1.0 if node_freq == 0 else 0.0
        feat_vec[('n', 'freq', '1')] = 1.0 if node_freq >= 1 else 0.0
        feat_vec[('n', 'freq', '2')] = 1.0 if node_freq >= 2 else 0.0
        feat_vec[('n', 'freq', '5')] = 1.0 if node_freq >= 5 else 0.0
        feat_vec[('n', 'freq', '10')] = 1.0 if node_freq >= 10 else 0.0
                
        return feat_vec
    
    def ffNodeConcept(self, node):
        """
        extract node concept feature
        """
        feat_vec = FeatureVector()
        
        # node concept
        feat_vec[('n', 'cpt', node.concept)] = 1.0
        
        return feat_vec
    
    def ffNodeDepth(self, node):
        """
        depth of node in graph topology
        """
        feat_vec = FeatureVector()
        
        # foremost occurrence of node1, node2
        # average occurrence position of node1, node2
        node_foremost = 1e10
        node_average = 0.0
        node_freq = len(node.sources)
        
        for source in node.sources:
            node_depth = len((source.graph_idx).split('.'))
            node_average += node_depth
            if node_foremost > node_depth:
                node_foremost = node_depth
        
        if node_freq > 0:
            node_average /= node_freq
        
        if node_freq > 0:
            feat_vec[('n', 'dep', 'fmst', '1')] = 1.0 if node_foremost >= 1 else 0.0
            feat_vec[('n', 'dep', 'fmst', '2')] = 1.0 if node_foremost >= 2 else 0.0
            feat_vec[('n', 'dep', 'fmst', '3')] = 1.0 if node_foremost >= 3 else 0.0
            feat_vec[('n', 'dep', 'fmst', '4')] = 1.0 if node_foremost >= 4 else 0.0
            feat_vec[('n', 'dep', 'fmst', '5')] = 1.0 if node_foremost >= 5 else 0.0
            
            feat_vec[('n', 'dep', 'avg', '1')] = 1.0 if node_average >= 1 else 0.0
            feat_vec[('n', 'dep', 'avg', '2')] = 1.0 if node_average >= 2 else 0.0
            feat_vec[('n', 'dep', 'avg', '3')] = 1.0 if node_average >= 3 else 0.0
            feat_vec[('n', 'dep', 'avg', '4')] = 1.0 if node_average >= 4 else 0.0
            feat_vec[('n', 'dep', 'avg', '5')] = 1.0 if node_average >= 5 else 0.0
        
        return feat_vec
    
    def ffNodePos(self, node):
        """
        extract node position features
        """
        feat_vec = FeatureVector()
        
        # foremost occurrence of node
        # average occurrence position of node
        node_foremost = 1e10
        node_average = 0.0
        node_freq = len(node.sources)
        
        for source in node.sources:
            node_average += source.line_num
            if node_foremost > source.line_num:
                node_foremost = source.line_num
        
        if node_freq > 0:
            node_average /= node_freq
        
        if node_freq > 0:
            # foremost occurrence
            feat_vec[('n', 'posit', 'fmst', '5')] = 1.0 if node_foremost >= 5 else 0.0
            feat_vec[('n', 'posit', 'fmst', '6')] = 1.0 if node_foremost >= 6 else 0.0
            feat_vec[('n', 'posit', 'fmst', '7')] = 1.0 if node_foremost >= 7 else 0.0
            feat_vec[('n', 'posit', 'fmst', '10')] = 1.0 if node_foremost >= 10 else 0.0
            feat_vec[('n', 'posit', 'fmst', '15')] = 1.0 if node_foremost >= 15 else 0.0
            # average occurrence
            feat_vec[('n', 'posit', 'avg', '5')] = 1.0 if node_average >= 5 else 0.0
            feat_vec[('n', 'posit', 'avg', '6')] = 1.0 if node_average >= 6 else 0.0
            feat_vec[('n', 'posit', 'avg', '7')] = 1.0 if node_average >= 7 else 0.0
            feat_vec[('n', 'posit', 'avg', '10')] = 1.0 if node_average >= 10 else 0.0
            feat_vec[('n', 'posit', 'avg', '15')] = 1.0 if node_average >= 15 else 0.0
        
        return feat_vec
    
    def ffNodeSpan(self, node):
        """
        extract features from node spans
        """
        feat_vec = FeatureVector()
        
        # longest span of node
        # average span of node
        node_longest = -1
        node_average = 0.0
        node_freq = len(node.sources)
        
        for source in node.sources:
            node_span = source.end_idx - source.start_idx
            node_average += node_span
            if node_longest < node_span:
                node_longest = node_span
        
        if node_freq > 0:
            node_average /= node_freq
        
        if node_freq > 0:
            feat_vec[('n', 'span', 'lgst', '0')] = 1.0 if node_longest >= 0 else 0.0
            feat_vec[('n', 'span', 'lgst', '1')] = 1.0 if node_longest >= 1 else 0.0
            feat_vec[('n', 'span', 'lgst', '2')] = 1.0 if node_longest >= 2 else 0.0
            feat_vec[('n', 'span', 'lgst', '5')] = 1.0 if node_longest >= 5 else 0.0
            feat_vec[('n', 'span', 'lgst', '10')] = 1.0 if node_longest >= 10 else 0.0
            
            feat_vec[('n', 'span', 'avg', '0')] = 1.0 if node_longest >= 0 else 0.0
            feat_vec[('n', 'span', 'avg', '1')] = 1.0 if node_longest >= 1 else 0.0
            feat_vec[('n', 'span', 'avg', '2')] = 1.0 if node_longest >= 2 else 0.0
            feat_vec[('n', 'span', 'avg', '5')] = 1.0 if node_longest >= 5 else 0.0
            feat_vec[('n', 'span', 'avg', '10')] = 1.0 if node_longest >= 10 else 0.0
        
        return feat_vec
                
    def ffNodeCollapsedEntity(self, node):
        """
        extract features from collapsed concept node
        """
        feat_vec = FeatureVector()
        
        # named entity or not
        feat_vec[('n', 'nam-ent')] = 1.0 if '_' in node.concept else 0.0
        feat_vec[('n', 'date-ent')] = 1.0 if (node.concept).startswith('date-entity') else 0.0
            
        return feat_vec   
    

if __name__ == '__main__':
    
    input_dir = '/Users/user/Data/SemanticSumm/Proxy/gold/split/dev/'
    body_file = 'aligned-amr-release-1.0-dev-proxy-body.txt'
    summ_file = 'aligned-amr-release-1.0-dev-proxy-summary.txt'
    
    corpus = buildCorpus(os.path.join(input_dir, body_file),
                         os.path.join(input_dir, summ_file))
    feat_extr = FeatureExtractor()
    feat_vec = FeatureVector()

    for inst in corpus:
        curr_filename = inst.filename
        my_nodes, s_nodes = inst.nodes
        my_edges, s_edges = inst.edges
        
#         logger.debug('extracting features for file: %s' % curr_filename)
#         for k_edge, v_edge in my_edges.iteritems():
#             for tag in [0,1]:
#                 feat_vec += feat_extr.getEdgeFeats(k_edge, v_edge, tag, curr_filename, my_nodes, my_edges)
                
        logger.debug('extracting features for file: %s' % curr_filename)
        for k_node, v_node in my_nodes.iteritems():
            for tag in [0,1]:
                feat_vec += feat_extr.getNodeFeats(k_node, v_node, tag, curr_filename, my_nodes, my_edges)
    
    with codecs.open('output_file', 'w', 'utf-8') as outfile:
        outfile.write('%s\n' % feat_vec.toString())
        
    
    
    


















