#!/usr/bin/env python
# -*- coding: utf-8 -*-



import re

from collections import namedtuple
from collections import Counter
from fei.model.utils import getLogger

logger = getLogger()


# information obtained from file
NodeSource = namedtuple('NodeSource', 'graph_idx, start_idx, end_idx, word_str, filename, line_num, sentence')
EdgeSource = namedtuple('EdgeSource', 'relation, filename, line_num, sentence')

class AmrNode(object):
    def __init__(self, graph_idx=None, short_hnd=None, concept=None):
        self.graph_idx = graph_idx
        self.short_hnd = short_hnd
        self.concept = concept
        self.sources = [] # list of NodeSource
        
    def __repr__(self):
        return '%s %s %s' % (self.graph_idx, self.short_hnd, self.concept)
    
    def toString(self):
        """
        convert AmrNode to string, include full node information
        """
        node_info = []
        if self.concept: node_info.append('[concept]: %s' % self.concept)
        
        if self.sources: # omit sentence info
            for idx, source in enumerate(self.sources):
                if source.graph_idx: node_info.append('[source_id]: %d [graph_idx]: %s' % (idx, source.graph_idx))
                if source.start_idx: node_info.append('[source_id]: %d [start_idx]: %d' % (idx, source.start_idx))
                if source.end_idx: node_info.append('[source_id]: %d [end_idx]: %d' % (idx, source.end_idx))
                if source.word_str: node_info.append('[source_id]: %d [word_str]: %s' % (idx, source.word_str))
                if source.filename: node_info.append('[source_id]: %d [filename]: %s' % (idx, source.filename))
                if source.line_num: node_info.append('[source_id]: %d [line_num]: %d' % (idx, source.line_num))
                
        # final string
        return '\n'.join(node_info) 


class AmrEdge(object):
    def __init__(self, node1=AmrNode(), node2=AmrNode(), relation=None):
        self.node1 = node1
        self.node2 = node2
        self.relation = relation
        self.sources = [] # list of EdgeSource
        return
    
    def __repr__(self):
        return '%s (%s)-> %s' % (self.node1.short_hnd, self.relation, self.node2.short_hnd)
    
    def toString(self):
        """
        convert AmrEdge to string, include full edge information
        """
        edge_info = []
        if self.node1: 
            if self.node1.concept: edge_info.append('[node1_concept]: %s' % self.node1.concept)
        if self.node2:
            if self.node2.concept: edge_info.append('[node2_concept]: %s' % self.node2.concept)
            
        if self.sources: # omit sentence info
            for idx, source in enumerate(self.sources):
                if source.relation: edge_info.append('[source_id]: %d [relation]: %s' % (idx, source.relation))
                if source.filename: edge_info.append('[source_id]: %d [filename]: %s' % (idx, source.filename))
                if source.line_num: edge_info.append('[source_id]: %d [line_num]: %d' % (idx, source.line_num))
                
        # final string            
        return '\n'.join(edge_info)

class CompareAmrEdge(object):
    """
    two edges are equal if their nodes and relation are the same
    """
    def __init__(self, edge):
        self.edge = edge
    
    def __hash__(self):
        return hash(tuple(sorted([self.edge.node1.concept, self.edge.node2.concept]) + [self.edge.relation]))
    
    def __eq__(self, other):
        if type(other) == type(self):
            if other.edge.relation == self.edge.relation:
                if ((other.edge.node1.concept == self.edge.node1.concept 
                     and other.edge.node2.concept == self.edge.node2.concept)
                    or (other.edge.node1.concept == self.edge.node2.concept
                        and other.edge.node2.concept == self.edge.node1.concept)):
                    return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __repr__(self):
        return self.edge.__repr__()


class CompareAmrEdgeWoRel(object):
    """
    two edges are equal if their nodes are the same
    """
    def __init__(self, edge):
        self.edge = edge
    
    def __hash__(self):
        return hash(tuple(sorted([self.edge.node1.concept, self.edge.node2.concept])))
    
    def __eq__(self, other):
        if type(other) == type(self):
            if ((other.edge.node1.concept == self.edge.node1.concept 
                 and other.edge.node2.concept == self.edge.node2.concept)
                or (other.edge.node1.concept == self.edge.node2.concept
                    and other.edge.node2.concept == self.edge.node1.concept)):
                return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return self.edge.__repr__()


class AmrGraph(object):
    def __init__(self):
        return
    
    def getNodes(self, tokens):
        """
        obtain a set of nodes from an AMR graph.
        each node represents a concept.
        """
        nodes, curr_idx = self._getNodesIter(None, tokens[:], 0) # copy tokens
        assert curr_idx == len(tokens)
        return [node for node in self._flatten(nodes)]
    
    def getEdges(self, tokens):
        """
        obtain a set of edges from an AMR graph.
        each edge connects two concepts.
        """
        # get nodes
        nodes = self.getNodes(tokens)
        
        # index nodes by short_hnd
        node_indices = {}
        for node in nodes:
            short_hnd = node.short_hnd
            node_indices.setdefault(short_hnd, node)
        
        # get edges
        edges, curr_idx = self._getEdgesIter(None, tokens[:], 0, node_indices) # copy tokens
        assert curr_idx == len(tokens)-1 or curr_idx == len(tokens)
        
        # share nodes across edges
        for edge in edges:
            curr_short_hnd = edge.node1.short_hnd
            if curr_short_hnd in node_indices:
                edge.node1 = node_indices[curr_short_hnd]
            
            curr_short_hnd = edge.node2.short_hnd
            if curr_short_hnd in node_indices:
                edge.node2 = node_indices[curr_short_hnd]
        
        return edges

    def _flatten(self, curr_list):
        """
        _flatten arbitrarily nested list
        """
        for i in curr_list:
            if isinstance(i, list) or isinstance(i, tuple):
                for j in self._flatten(i):
                    yield j
            else:
                yield i

    def _getNodesIter(self, graph_idx, tokens, curr_idx):
        """
        obtain a set of nodes from an AMR graph.
        each node represents a concept.
        """
        nodes = [] # list of nodes
        
        while curr_idx < len(tokens):
            t = tokens[curr_idx]

            if t.endswith(')'):
                tokens[curr_idx] = t[:-1]
                return (nodes, curr_idx)
            
            elif t.startswith('('):
                curr_node = AmrNode()
                curr_node.short_hnd = t[1:]
                curr_idx += 2
                curr_node.concept = re.sub(r'\)+$', '', tokens[curr_idx])
                curr_graph_idx = str(len(nodes)) if not graph_idx else graph_idx + '.' + str(len(nodes))
                curr_node.graph_idx = curr_graph_idx
                nodes.append([curr_node])
                
                # recursion
                new_nodes, new_idx = self._getNodesIter(curr_graph_idx, tokens, curr_idx)
                if new_nodes: nodes.append(nodes.pop() + new_nodes)
                curr_idx = new_idx
                
            else: 
                curr_idx += 1
        
        return nodes, curr_idx
    
    def _getEdgesIter(self, curr_node, tokens, curr_idx, node_indices):
        """
        obtain a set of edges from an AMR graph.
        each edge connects two concepts.
        """
        edges = [] # list of edges
        
        while curr_idx < len(tokens):
            t = tokens[curr_idx]

            # get current node if not provided
            if not curr_node and t.startswith('('):
                curr_node = AmrNode()
                curr_node.short_hnd = t[1:]
                curr_idx += 2
                    
            elif t.endswith(')'):
                tokens[curr_idx] = t[:-1]
                return (edges, curr_idx)
                
            elif t.startswith(':'):
                curr_edge = AmrEdge()
                curr_edge.node1 = curr_node
                curr_edge.relation = t[1:]
                curr_idx += 1
                
                new_t = re.sub(r'\)+$', '', tokens[curr_idx])
                
                # get second node for current edge
                # second node is a new concept
                if new_t.startswith('('):
                    new_node = AmrNode()
                    new_node.short_hnd = tokens[curr_idx][1:]
                    curr_edge.node2 = new_node
                    edges.append(curr_edge)
                    curr_idx += 2
                    
                    # recursion
                    new_edges, new_idx = self._getEdgesIter(new_node, tokens, curr_idx, node_indices)
                    if new_edges: edges.extend(new_edges)
                    curr_idx = new_idx
                
                # second node refers to an old concept (no recursion)
                elif new_t in node_indices:
                    new_node = node_indices[new_t]
                    curr_edge.node2 = new_node
                    edges.append(curr_edge)
                
            else: 
                curr_idx += 1
        
        return edges, curr_idx

    def getCollapsedNodesAndEdges(self, tokens):
        """
        collapse "name" and "date-entity" concepts
        collapse ":name" relation
        """
        # get nodes
        nodes = self.getNodes(tokens)
        
        # index nodes by short_hnd
        node_indices = {}
        for node in nodes:
            short_hnd = node.short_hnd
            node_indices.setdefault(short_hnd, node)
        
        # get edges
        edges, curr_idx = self._getEdgesIter(None, tokens[:], 0, node_indices) # copy tokens
        assert curr_idx == len(tokens)-1 or curr_idx == len(tokens)
        
        # share nodes across edges
        for edge in edges:
            curr_short_hnd = edge.node1.short_hnd
            if curr_short_hnd in node_indices:
                edge.node1 = node_indices[curr_short_hnd]
    
            curr_short_hnd = edge.node2.short_hnd
            if curr_short_hnd in node_indices:
                edge.node2 = node_indices[curr_short_hnd]
        
        # collapse "name" and "date-entity" concepts
        curr_short_hnd = None
        curr_concept = ''
        start_collapse = False
        
        for i, t in enumerate(tokens):
            # nested concepts may exist, re-start collapsing
            if t.startswith('('):
                curr_concept = ''
                start_collapse = False
                curr_short_hnd = t[1:]
                continue
            
            # assume "name" and "date-entity" do not contain nested concepts
            # if this is not the case, collapse process needs to be rewritten
            if (t == 'name' or t == 'date-entity') and tokens[i-1] == '/':
                start_collapse = True
                curr_concept = t # original concept
                continue
            
            if start_collapse == True:
                # remove ending brackets, concatenate tokens
                t_clean = re.sub(r'\)+$', '', t)
                curr_concept = '%s_%s' % (curr_concept, t_clean)
                    
                if t.endswith(')'): 
                    node_indices[curr_short_hnd].concept = curr_concept
                    curr_concept = ''
                    start_collapse = False
        
        # count node occurrences
        node_counts = Counter()
        for edge in edges:
            node_counts[edge.node1.short_hnd] += 1
            node_counts[edge.node2.short_hnd] += 1

        # collapse edges with "name" as relation
        new_edges = []
        for edge in edges:
            # right-hand-side node linked only to current edge
            if edge.relation == 'name' and node_counts[edge.node2.short_hnd] == 1:
                edge.node1.concept += '_' + edge.node2.concept
                if edge.node2.short_hnd in node_indices:
                    del node_indices[edge.node2.short_hnd]
            else: new_edges.append(edge)

        # in-place removal of collapsed nodes ("name" as relation)
        nodes[:] = [node for node in nodes if node.short_hnd in node_indices]
                            
        return nodes, new_edges
  
if __name__ == '__main__':

    graph_string = '''
    (r / reopen-01
          :ARG1 (u / university :name (n / name :op1 "Naif" :op2 "Arab" :op3 "Academy" :op4 "for" :op5 "Security" :op6 "Sciences")
                :purpose (o / oppose-01
                      :ARG1 (t / terror))
                :mod (e / ethnic-group :name (n3 / name :op1 "Arab")
                      :mod (p / pan)))
          :time (d / date-entity :year 2002 :month 1 :day 5)
          :frequency (f / first
                :time (s / since
                      :op1 (a3 / attack-01
                            :ARG1 (c / country :name (n2 / name :op1 "US"))
                            :time (d2 / date-entity :year 2001 :month 9)))))
    '''
    
    graph_string = '''
    (l / likely
        :polarity -
        :domain (g / give-01
                    :ARG0 (t / tour-01
                            :ARG0 (o / organization
                                    :name (n / name
                                            :op1 "IAEA"))
                            :ARG1 (l2 / laboratory
                                    :ARG0-of (p / produce-01
                                                :ARG1 (i / isotope))))
                    :ARG1 (p2 / picture
                            :mod (f / full)
                            :consist-of (a / ambition
                                            :mod (w / weapon
                                                    :mod (n2 / nucleus))
                                            :ARG1-of (s / suspect-01)
                                            :poss (c / country
                                                :name (n3 / name
                                                        :op1 "North"
                                                        :op2 "Korea"))))))
    '''
  
    graph_string = '''
    (s / schedule-01
        :ARG1 (p / project
            :mod (m / monetary-quantity
                    :unit (d / dollar)
                    :quant 4600000000))
        :ARG2 (c / complete-01
            :ARG1 p)
        :ARG3 (d2 / date-entity
            :year 2003))
    '''
  
    graph_string = '''
    (a2 / and
          :op1 (c2 / cross-02
                :ARG0 (g / guerrilla)
                :ARG1 (l3 / location :name (n6 / name :op1 "Line" :op2 "of" :op3 "Control")
                      :ARG0-of (d / divide-02
                            :ARG1 (w / world-region :name (n4 / name :op1 "Kashmir")))))
          :op2 (c4 / carry-03
                :ARG0 g
                :ARG1 (a / attack-01
                      :ARG0 g
                      :location (s2 / side
                            :part-of n4
                            :part-of (c5 / country :name (n5 / name :op1 "India"))))))
    '''
  
    graph_string = '''
    (s / say-01
          :ARG0 (m2 / military :name (n / name :op1 "Russian" :op2 "Spaces" :op3 "Forces"))
          :ARG1 (s2 / state-01
                :ARG1 (p / possible
                      :domain (t / track-01
                            :ARG0 (c / complex :name (n3 / name :op1 "Okno" :ARG0-of (m / mean-01
                                              :ARG1 (w / window))))
                            :ARG1 (o / object
                                  :location (r / relative-position
                                        :op1 (p2 / planet :name (n6 / name :op1 "Earth"))
                                        :quant (d / distance-quantity :quant 40000
                                              :unit (k / kilometer))))))
                :ARG1-of (c2 / carry-01
                      :ARG0 (a / agency :name (n8 / name :op1 "Interfax")
                            :mod (n2 / news
                                  :mod (m3 / military))))))
    '''
  
    g = AmrGraph()
    tokens = graph_string.split()

    nodes, edges = g.getCollapsedNodesAndEdges(tokens)    
    print 
    for node in nodes: print node
    print
    for edge in edges: print '%s (%s)-> %s' % (edge.node1, edge.relation, edge.node2)
    





    
    