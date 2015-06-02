#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
import os
import re
import codecs
import logging

from fei.preprocess.triple import Triple
from fei.backup.stats_coverage import getCoverageStats
from collections import defaultdict
from collections import Counter as mset

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def getTriples(input_file):
    """
    get triples from JAMR parsed file
    """
    base_filename = os.path.basename(input_file)
    line_num = -1 # start from 0
    triples = []
    
    with codecs.open(input_file, 'r', 'utf-8') as infile:
        for line in infile:
            line = line.strip()
            
            if line.startswith('# ::snt'): 
                line_num += 1
                continue

            if line.startswith('(') and line.endswith(')'):
                line = line[1:-1] # remove parentheses
                triple = re.sub(r'[0-9A-Za-z\-]+ \/ ', '', line)
                
                try:
                    c1, r, c2 = triple.split(', ')
                    t = Triple(c1.lower(), c2.lower(), r.lower(), base_filename, line_num, tokens=None)
                    triples.append(t)
                    
                except ValueError: pass
                
    return triples

def getExtendedTriples(input_file):
    """
    extract list of triples from JAMR parsed file
    """
    extended_triples = []
    
    cached_concepts = []
    cached_indices = {}
    line_num = -1 # start from 0
    base_filename = os.path.basename(input_file)
    
    with codecs.open(input_file, 'r', 'utf-8') as infile:
        for line in infile:
            line = line.strip()
            
            if line.startswith('# ::snt'): 
                line_num += 1
                continue
            
            if line == '' and cached_indices:
                for i, c1 in enumerate(cached_concepts):
                    for j, c2 in enumerate(cached_concepts):
                        if j <= i: continue
                        t = Triple(c1, c2, '', base_filename, line_num, tokens=None)
                        extended_triples.append(t)
                cached_concepts = []
                cached_indices = {}
                continue
                
            if line.startswith('(') and line.endswith(')'):
                line = line[1:-1] # remove parentheses
                triple = re.sub(r'[0-9A-Za-z\-]+ \/ ', '', line)
                
                try:
                    c1, _, c2 = triple.split(', ')
                    idx1, idx2 = re.findall(r'([0-9A-Za-z]+) \/', line)
                    
                    if idx1 not in cached_indices:
                        cached_indices[idx1] = 1
                        cached_concepts.append(c1.lower())
                        
                    if idx2 not in cached_indices:
                        cached_indices[idx2] = 1
                        cached_concepts.append(c2.lower())
                        
                except ValueError: pass
                
    return extended_triples


def getConcepts(input_file):
    """
    get concepts from file
    """
    concepts = mset() # use Counter for concepts
    cached_indices = {}

    with codecs.open(input_file, 'r', 'utf-8') as infile:
        for line in infile:
            line = line.strip()

            if line == '' and cached_indices:
                cached_indices = {}
                continue
                
            if line.startswith('(') and line.endswith(')'):
                line = line[1:-1] # remove parentheses
                triple = re.sub(r'[0-9A-Za-z\-]+ \/ ', '', line)
                
                try:
                    c1, _, c2 = triple.split(', ')
                    idx1, idx2 = re.findall(r'([0-9A-Za-z]+) \/', line)
                    
                    if idx1 not in cached_indices:
                        cached_indices[idx1] = 1
                        concepts[c1.lower()] += 1
                        
                    if idx2 not in cached_indices:
                        cached_indices[idx2] = 1
                        concepts[c2.lower()] += 1
                        
                except ValueError: pass
                
    return concepts

def countTriples(input_dir, dirname, uniq=False):
    """
    calculate percentage of triples in the summary that are also covered
    by the documents
    foldername: turbo
    """
    
    docs_parsed_dir = os.path.join(input_dir, 'docs_' + dirname)
    models_parsed_dir = os.path.join(input_dir, 'models_' + dirname)
    
    models_triples = defaultdict(list)
    docs_triples = defaultdict(list)
    docs_extended_triples = defaultdict(list)
    docs_concepts = defaultdict(mset) # Counter
    
    # obtain triples from documents
    # use 20 documents per topic (NO distinction between A and B)
    for filename in os.listdir(docs_parsed_dir):
        topic = re.sub(r'-.*$', '', filename)  # "D0848"
        
        input_file = os.path.join(docs_parsed_dir, filename)
        triples = getTriples(input_file)
        extended_triples = getExtendedTriples(input_file)
        concepts = getConcepts(input_file)
        docs_triples[topic].extend(triples)
        docs_extended_triples[topic].extend(extended_triples)
        docs_concepts[topic] += concepts
                
    # obtain triples from summaries
    # use 4 summaries per topic (EXISTS distinction between A and B)
    for filename in os.listdir(models_parsed_dir):
        topic = re.sub(r'\..*$', '', filename)  # "D0848-A"
        
        input_file = os.path.join(models_parsed_dir, filename)
        triples = getTriples(input_file)
        models_triples[topic].extend(triples)
    
    return getCoverageStats(models_triples, docs_triples, 
                            docs_extended_triples, docs_concepts, uniq)


if __name__ == '__main__':
    input_dir = '/home/user/TAC'
    dirname = 'jamr'
    
    for year in ['2008', '2009', '2010', '2011']:
        curr_dir = os.path.join(input_dir, year)
        
        (p1, p2, p3, p4, p5) = countTriples(curr_dir, dirname, uniq=False)
        print 'year: %s folder: %s uniq: %s' % (year, dirname, False)
        print '\texact match: %.2f (%.2f)' % (p1, p1)
        print '\tmatch two concepts: %.2f (%.2f)' % (p2, p1+p2)
        print '\tmatch two concepts same tokens: %.2f (%.2f)' % (p3, p1+p2+p3)
        print '\tmatch two concepts same document set: %.2f (%.2f)' % (p4, p1+p2+p3+p4)
        print '\tno match: %.2f (%.2f)\n' % (p5, 100)
        
        (p1, p2, p3, p4, p5) = countTriples(curr_dir, dirname, uniq=True)
        print 'year: %s folder: %s uniq: %s' % (year, dirname, True)
        print '\texact match: %.2f (%.2f)' % (p1, p1)
        print '\tmatch two concepts: %.2f (%.2f)' % (p2, p1+p2)
        print '\tmatch two concepts same tokens: %.2f (%.2f)' % (p3, p1+p2+p3)
        print '\tmatch two concepts same document set: %.2f (%.2f)' % (p4, p1+p2+p3+p4)
        print '\tno match: %.2f (%.2f)\n' % (p5, 100)




