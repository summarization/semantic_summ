#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gurobipy as gp
import codecs

from numpy import random
from fei.model.feat_vec import FeatureVector
from fei.model.feat_extr import FeatureExtractor
from fei.model.utils import getLogger

logger = getLogger()  

class Decoder(object):
    """
    Implement of decoder for structured prediction
    """
    def __init__(self):
        self.gilp = GurobiILP()
        self.weights = FeatureVector()
        self.feat_extr = FeatureExtractor()
        return
    
    def decode(self, instance, oracle_len='nolen', node_cost=None, edge_cost=None):
        """
        an instance includes:
        my_nodes: (1,) -> AmrNode1, (2,) -> AmrNode2, ...
        my_edges: (1,2) -> AmrEdge1, (2,1) -> AmrEdge2,...
        selected_nodes: (1,), (3,),... nodes contained in summary graph
        selected_edges: (1,2), (3,1),... edges contained in summary graph
        """
        logger.debug('start feature extraction...')
        
        curr_filename = instance.filename
        my_nodes, oracle_nodes, root_nodes = instance.nodes # nodes and selected nodes
        my_edges, oracle_edges = instance.edges # edges and selected edges
        num_gold_nodes, num_gold_edges = instance.gold # number of gold nodes and edges
        
        node_weights = {}
        edge_weights = {}
        
        # get edge weights
        num_nonnegative_edges = 0
        for k_edge, v_edge in my_edges.iteritems():
            for tag in [0,1]:
                edge_feats = self.feat_extr.getEdgeFeats(k_edge, v_edge, tag, curr_filename, my_nodes, my_edges)
                edge_weights[k_edge + (tag,)] = self.weights.dot(edge_feats)
                # cost-augmented decoding
                if edge_cost is not None:
                    if tag == 0 and k_edge not in oracle_edges:      # true negative
                        curr_edge_cost = 0
                    if tag == 1 and k_edge in oracle_edges:          # true positive
                        curr_edge_cost = 0
                    if tag == 1 and k_edge not in oracle_edges:      # false positive
                        curr_edge_cost = edge_cost 
                    if tag == 0 and k_edge in oracle_edges:          # false negative
                        curr_edge_cost = edge_cost
                    edge_weights[k_edge + (tag,)] += curr_edge_cost
                if tag == 1 and edge_weights[k_edge + (tag,)] > 0.0: num_nonnegative_edges += 1
                
        # count number of non-negative edges
        logger.debug('[num_nonnegative_edges]: %d' % num_nonnegative_edges)
        
        # get node weights
        num_nonnegative_nodes = 0
        for k_node, v_node in my_nodes.iteritems():
            for tag in [0,1]:
                node_feats = self.feat_extr.getNodeFeats(k_node, v_node, tag, curr_filename, my_nodes, my_edges)
                node_weights[k_node + (tag,)] = self.weights.dot(node_feats)
                # cost-augmented decoding
                if node_cost is not None: 
                    if tag == 0 and k_node not in oracle_nodes:
                        curr_node_cost = 0
                    if tag == 1 and k_node in oracle_nodes:
                        curr_node_cost = 0
                    if tag == 1 and k_node not in oracle_nodes:      # false positive
                        curr_node_cost = node_cost
                    if tag == 0 and k_node in oracle_nodes:          # false negative
                        curr_node_cost = node_cost
                    node_weights[k_node + (tag,)] += curr_node_cost
                if tag == 1 and node_weights[k_node + (tag,)] > 0.0: num_nonnegative_nodes += 1
                
        # count number of non-negative nodes
        logger.debug('[num_nonnegative_nodes]: %d' % num_nonnegative_nodes)
        
        # run Gurobi ILP decoder using node and edge weights
        # optionally set the decoded summary length (#nodes and #edges)
        logger.debug('start ILP decoding...')
        num_selected_nodes = num_gold_nodes if oracle_len == 'nodes' else 0
        num_selected_edges = num_gold_edges if oracle_len == 'edges' else 0
        selected_nodes, selected_edges, score_pred = self.gilp.decode(node_weights, edge_weights, root_nodes,
                                                                      num_selected_nodes=num_selected_nodes,
                                                                      num_selected_edges=num_selected_edges)
        logger.debug('[num_gold_nodes]: %d' % num_gold_nodes)
        logger.debug('[num_selected_nodes]: %d' % len(selected_nodes))
        logger.debug('[num_gold_edges]: %d' % num_gold_edges)
        logger.debug('[num_selected_edges]: %d' % len(selected_edges))
        
        # features that are associated with the decoded graph
        feat_vec = FeatureVector()
        
        for k_edge, v_edge in my_edges.iteritems():
            tag = 1 if k_edge in selected_edges else 0  # use decoded tag
            feat_vec += self.feat_extr.getEdgeFeats(k_edge, v_edge, tag, curr_filename, my_nodes, my_edges)
            
        for k_node, v_node in my_nodes.iteritems():
            tag = 1 if k_node in selected_nodes else 0  # use decoded tag
            feat_vec += self.feat_extr.getNodeFeats(k_node, v_node, tag, curr_filename, my_nodes, my_edges)
            
        # return features associated with decoded graph
        return feat_vec, selected_nodes, selected_edges, score_pred
    
    def oracle(self, instance):
        """
        an instance includes:
        my_nodes: (1,) -> AmrNode1, (2,) -> AmrNode2, ...
        my_edges: (1,2) -> AmrEdge1, (2,1) -> AmrEdge2,...
        root_nodes: (1,), (3,),... nodes that are root of sentence
        selected_nodes: (1,), (3,),... nodes contained in summary graph
        selected_edges: (1,2), (3,1),... edges contained in summary graph
        """
        logger.debug('start oracle decoding...')
        
        curr_filename = instance.filename
        my_nodes, oracle_nodes, _ = instance.nodes # nodes and selected nodes
        my_edges, oracle_edges = instance.edges # edges and selected edges
        
        # features that are associated with oracle graph
        feat_vec = FeatureVector()
        
        for k_edge, v_edge in my_edges.iteritems():
            tag = 1 if k_edge in oracle_edges else 0 # use oracle tag
            feat_vec += self.feat_extr.getEdgeFeats(k_edge, v_edge, tag, curr_filename, my_nodes, my_edges)
        
        for k_node, v_node in my_nodes.iteritems():
            tag = 1 if k_node in oracle_nodes else 0 # use oracle tag
            feat_vec += self.feat_extr.getNodeFeats(k_node, v_node, tag, curr_filename, my_nodes, my_edges)
        
        score_true = self.weights.dot(feat_vec)
        
        # return features associated with oracle graph
        return feat_vec, oracle_nodes, oracle_edges, score_true
    

class GurobiILP(object):
    
    def __init__(self):
        """
        initialize Gurobi ILP decoder
        """
        self.model = gp.Model()
        self.nodeVars = {}
        self.edgeVars = {}
        self.dummyEdgeVars = {}
        self.exclConstrs = {}
        self.num_selected_nodes = 0.0
        self.num_selected_edges = 0.0
        self.num_nodes = 0.0
        return
    
    def clear(self):
        """
        re-initialize
        """
        self.model = gp.Model()
        self.nodeVars = {}
        self.edgeVars = {}
        self.dummyEdgeVars = {}
        self.exclConstrs = {}
        self.num_selected_nodes = 0.0
        self.num_selected_edges = 0.0
        self.num_nodes = 0.0
        return
    
    def setNumNodes(self, num_nodes):
        """
        set number of nodes in the graph
        """
        self.num_nodes = num_nodes
        return

    def setNumSelectedNodes(self, num_selected_nodes):
        """
        set number of nodes that should be selected
        """
        self.num_selected_nodes = num_selected_nodes
        return
    
    def setNumSelectedEdges(self, num_selected_edges):
        """
        set number of edges that should be selected
        """
        self.num_selected_edges = num_selected_edges
        return
    
    def addGraphVars(self):
        """
        add graph variables
        """      
        num_nodes = self.num_nodes
          
        # node variables
        for i in xrange(1, num_nodes+1):
            for t in [0,1]:
                self.nodeVars[i,t] = self.model.addVar(vtype=gp.GRB.BINARY, name='n_%d_%d' % (i,t))
        
        # edge variables
        for i in xrange(1, num_nodes+1):
            for j in xrange(1, num_nodes+1):
                for t in [0,1]:
                    self.edgeVars[i,j,t] = self.model.addVar(vtype=gp.GRB.BINARY, name='e_%d_%d_%d' % (i,j,t))
        
        # integrate new variables
        self.model.update()
        return
    
    def addDummyEdges(self, root_nodes):
        """
        add dummy edge variables
        """
        num_nodes = self.num_nodes
        
        # dummy edge variables
        for i in xrange(1, num_nodes+1):
            for t in [0,1]:
                self.dummyEdgeVars[i,t] = self.model.addVar(vtype=gp.GRB.BINARY, name='e_0_%d_%d' % (i,t))

        # integrate new variables
        self.model.update()
        
        # exclusion constraints
        for i in xrange(1, num_nodes+1):
            if (i,) not in root_nodes:
                self.exclConstrs[i] = self.model.addConstr(gp.LinExpr(self.dummyEdgeVars[i,1]) == 0.0)
                
        return
    
    def getObjective(self, node_weights, edge_weights):
        """
        generate objective expression
        node_weights: (1,) -> 0.1, edge_weights: (1,2) -> 0.2
        self.nodeVars: (1,) -> nodeVar, self.edgeVars: (1,2) -> edgeVar
        """
        n_weights = [v for _, v in sorted(node_weights.iteritems())]
        n_vars = [v for _, v in sorted(self.nodeVars.iteritems())]
        
        e_weights = [v for _, v in sorted(edge_weights.iteritems())]
        e_vars = [v for _, v in sorted(self.edgeVars.iteritems())]
        
        # sum the weights of nodes and edges
        return gp.LinExpr(n_weights, n_vars) + gp.LinExpr(e_weights, e_vars)
    
    def addGraphConstrs(self):
        """
        add basic graph constraints
        """
        num_nodes = self.num_nodes
        
        # an edge/node is either selected or not
        for i in xrange(1, num_nodes+1):
            self.model.addConstr(gp.quicksum(self.nodeVars[i,t] for t in [0,1]) == 1.0)
        
        for i in xrange(1, num_nodes+1):
            for j in xrange(1, num_nodes+1):
                self.model.addConstr(gp.quicksum(self.edgeVars[i,j,t] for t in [0,1]) == 1.0)
                
        for i in xrange(1, num_nodes+1):
            self.model.addConstr(gp.quicksum(self.dummyEdgeVars[i,t] for t in [0,1]) == 1.0)
        
        # one or more edges from ROOT to any node 
        self.model.addConstr(gp.quicksum(self.dummyEdgeVars[i,1] for i in xrange(1, num_nodes+1)) >= 1.0)
        
        # at most one edge between any pair of nodes (no self-loop)
        for i in xrange(1, num_nodes+1):
            for j in xrange(1, num_nodes+1):
                self.model.addConstr(gp.LinExpr(self.edgeVars[i,j,1] + self.edgeVars[j,i,1]) <= 1.0)
        
        # an edge may only be included if both its endpoints are included
        for i in xrange(1, num_nodes+1):
            self.model.addConstr(gp.LinExpr(self.nodeVars[i,1] - self.dummyEdgeVars[i,1]) >= 0.0)
        
        for i in xrange(1, num_nodes+1):
            for j in xrange(1, num_nodes+1):
                self.model.addConstr(gp.LinExpr(self.nodeVars[i,1] - self.edgeVars[i,j,1]) >= 0.0)
                self.model.addConstr(gp.LinExpr(self.nodeVars[j,1] - self.edgeVars[i,j,1]) >= 0.0)
        return
    
    def addFlowConstrs(self):
        """
        add commodity flow constraints
        """
        num_nodes = self.num_nodes
        
        # edge flow variables
        edgeFlowVars = {}
        for i in xrange(1, num_nodes+1):
            for j in xrange(1, num_nodes+1):
                edgeFlowVars[i,j] = self.model.addVar(vtype=gp.GRB.INTEGER, name='f_%d_%d' % (i,j))
        
        # dummy edge flow variables
        dummyEdgeFlowVars = {}
        for i in xrange(1, num_nodes+1):
            dummyEdgeFlowVars[i] = self.model.addVar(vtype=gp.GRB.INTEGER, name='f_0_%d' % i)
        
        # integrate flow variables
        self.model.update()
        
        # ROOT node must send out 1 unit of flow for each node present
        self.model.addConstr(gp.quicksum(dummyEdgeFlowVars.values()) 
                             - gp.quicksum(self.nodeVars[i,1] for i in xrange(1, num_nodes+1)) == 0.0)
        
        # flow may only be sent over an edge if the edge is included in the solution
        # TODO: edge may exist only if flow is greater than zero
        for i in xrange(1, num_nodes+1):
            for j in xrange(1, num_nodes+1):
                self.model.addConstr(gp.LinExpr(self.edgeVars[i,j,1] * num_nodes - edgeFlowVars[i,j]) >= 0.0)
#                 self.model.addConstr(gp.LinExpr(edgeFlowVars[i,j] * num_nodes - self.edgeVars[i,j,1]) >= 0.0)
        
        for i in xrange(1, num_nodes+1):
            self.model.addConstr(gp.LinExpr(self.dummyEdgeVars[i,1] * num_nodes - dummyEdgeFlowVars[i]) >= 0.0)
#             self.model.addConstr(gp.LinExpr(dummyEdgeFlowVars[i] * num_nodes - self.dummyEdgeVars[i,1]) >= 0.0)
        
        # each node must consume exactly one unit of flow, if it is included in solution
        for i in xrange(1, num_nodes+1):
            incoming = gp.quicksum(edgeFlowVars[src,i] for src in xrange(1, num_nodes+1)) + dummyEdgeFlowVars[i]
            outgoing = gp.quicksum(edgeFlowVars[i,dst] for dst in xrange(1, num_nodes+1))
            self.model.addConstr(incoming - outgoing - self.nodeVars[i,1] == 0)

        return

    def addLenConstrs(self):
        """
        add length constraints
        """
        num_nodes = self.num_nodes
        num_selected_nodes = self.num_selected_nodes
        num_selected_edges = self.num_selected_edges
        
        if num_selected_nodes > 0.0:
            self.model.addConstr(gp.quicksum(self.nodeVars[i,1] for i in xrange(1, num_nodes+1)) == num_selected_nodes)

        if num_selected_edges > 0.0:
            self.model.addConstr(gp.quicksum(gp.quicksum(self.edgeVars[i,j,1] for i in xrange(1, num_nodes+1)) 
                                             for j in xrange(1, num_nodes+1)) == num_selected_edges)
        return
    
    def addExclusionConstrs(self, edge_weights):
        """
        exclude edges that are not included in original graph
        """
        num_nodes = self.num_nodes
        
        # exclude edges that are not included in original graph
        for i in xrange(1, num_nodes+1):
            for j in xrange(1, num_nodes+1):
                if (i,j,1) not in edge_weights:
                    self.exclConstrs[i,j] = self.model.addConstr(gp.LinExpr(self.edgeVars[i,j,1]) == 0.0)
                    for t in [0,1]: edge_weights[i,j,t] = 0.0
       
        return
    
    def addIncomingEdgeConstrs(self):
        """
        restrict each node has one incoming edge or less
        """
        num_nodes = self.num_nodes
        
        # set each node has one incoming edge or less
        for i in xrange(1, num_nodes+1):
            incoming = gp.quicksum(self.edgeVars[src,i,1] for src in xrange(1, num_nodes+1)) + self.dummyEdgeVars[i,1]
            self.model.addConstr(incoming <= 1.0) 
        return
    
    def getNumVars(self, num_nodes):
        """
        calculate number of variables in the ILP model
        """
        num_graph_vars = (num_nodes * 2) * 2 + (num_nodes * num_nodes * 2)
        num_flow_vars = num_nodes + num_nodes * num_nodes
        
        return num_graph_vars + num_flow_vars
    
    def decode(self, node_weights, edge_weights, root_nodes, 
               num_selected_nodes=None, num_selected_edges=None, output_file=None):
        """
        decoding using Gurobi
        """
        # clear ILP model
        self.clear()
        
        # set number of nodes in graph
        num_nodes = int(len(node_weights)/2)
        self.setNumNodes(num_nodes)
        
        # set number of nodes to be selected
        if num_selected_nodes: 
            self.setNumSelectedNodes(num_selected_nodes)
        
        # set number of edges to be selected
        if num_selected_edges:
            self.setNumSelectedEdges(num_selected_edges)
        
        # add variables and constraints
        logger.debug('adding ILP variables...')
        self.addGraphVars()
        self.addDummyEdges(root_nodes)
        
        logger.debug('adding ILP graph constraints...')
        self.addGraphConstrs()
        
        logger.debug('adding ILP flow constraints...')
        self.addFlowConstrs()
        
        logger.debug('adding ILP other constraints...')
        self.addLenConstrs()
        self.addIncomingEdgeConstrs()
        self.addExclusionConstrs(edge_weights) # edge_weights changed here

        # set objective
        logger.debug('adding ILP objective...')
        obj = self.getObjective(node_weights, edge_weights)
        self.model.setObjective(obj, gp.GRB.MAXIMIZE)
        self.model.update()
        
        # run solver, get objective and runtime
        logger.debug('start ILP optimizer...')
        self.model.optimize()
        score_pred = self.model.ObjVal  # get objective
        runtime = self.model.Runtime    # get runtime
        logger.debug('[num_variables]: %d' % self.getNumVars(num_nodes))
        logger.debug('[runtime]: %f' % runtime)
        logger.debug('[objective]: %f' % score_pred)
        
        # get nodes and edges
        nodes = set()
        edges = set()
        for v in self.model.getVars():
            varname = v.Varname
            # selected nodes
            if varname.startswith('n') and varname.endswith('1') and v.X > 0.0:
                _, n_idx, _ = varname.split('_')
                nodes.add((int(n_idx),))
            # selected edges
            if varname.startswith('e') and varname.endswith('1') and v.X > 0.0:
                if varname.startswith('e_0'): continue # exclude (0,1), (0,2)... 
                _, src_idx, dst_idx, _ = varname.split('_')
                edges.add((int(src_idx), int(dst_idx)))
        
        # write weights and decoding results to output file
        if output_file is not None:
            with codecs.open(output_file, 'w', 'utf-8') as outfile:
                outfile.write('[Runtime]: %f\n' % self.model.Runtime)
                outfile.write('[ObjVal]: %f\n' % self.model.ObjVal)
                # node weights
                for (i,t), v in sorted(node_weights.iteritems()):
                    outfile.write('%d %d\t%f\n' % (i,t,v))
                # edge weights
                for (i,j,t), v in sorted(edge_weights.iteritems()):
                    outfile.write('%d %d %d\t%f\n' % (i,j,t,v))
                # selected nodes and edges
                for v in self.model.getVars():
                    if ((v.Varname).startswith('f') or (v.Varname).endswith('1')) and v.X > 0.0:
                        outfile.write('%s\t%f\n' % (v.Varname, v.X))
                # final line
                outfile.write('\n')
                 
        return nodes, edges, score_pred

if __name__ == '__main__':
    decoder = GurobiILP()
    
    node_weights = {}
    edge_weights = {}
    num_nodes = 6
    
    for i in xrange(1, num_nodes+1):
        for t in [0,1]:
            node_weights[i,t] = random.random()
    
    for i in xrange(1, num_nodes+1):
        for j in xrange(1, num_nodes+1):
            for t in [0,1]:
                edge_weights[i,j,t] = random.random()
            
    nodes, edges, score = decoder.decode(node_weights, edge_weights, output_file='test_output')
    for node in nodes: print node
    for edge in edges: print edge
    print score














