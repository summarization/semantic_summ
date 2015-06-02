#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import logging

from collections import defaultdict
from fei.preprocess.triple import CompareTriplesExactMatch
from fei.preprocess.triple import CompareTriplesWoRelation

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def getConceptStats(models_concepts, docs_concepts):
    """
    calculate average number of concepts in summary and document set
    """
    num_topics = 0
    num_uniq_m_concepts = 0.0
    num_uniq_d_concepts = 0.0
    per_covered_m_concepts = 0.0
    
    # important to iterate through model files
    for topic in models_concepts:
        num_topics += 1
        num_uniq_m_concepts += len(models_concepts[topic])
        num_uniq_d_concepts += len(docs_concepts[topic])
        
        m_concepts = set(models_concepts[topic])
        d_concepts = set(docs_concepts[topic])
        intersect = m_concepts & d_concepts
        per_covered_m_concepts += len(intersect)*100/len(m_concepts)
        
    num_uniq_m_concepts /= num_topics
    num_uniq_d_concepts /= num_topics
    per_covered_m_concepts /= num_topics
    
    result = '''
    ------
    number of total files: %d
    number of unique concepts per summary: %.1f
    number of unique concepts per document set: %.1f
    percentage of covered summary concepts: %.1f%%
    ''' % (num_topics, num_uniq_m_concepts, num_uniq_d_concepts, per_covered_m_concepts)
    
    return result

def getCoverageStats(models_triples, docs_triples, docs_extended_triples=None, docs_concepts=None):
    """
    considers only unique triples
    """
    
    counts = defaultdict(lambda: defaultdict(int))
    
    # important to iterate through model files
    for topic in models_triples:
        
        m_triples = models_triples[topic]
        m_triples_exact = set()
        
        # get triples from documents
        d_triples = docs_triples[topic]
        d_triples_exact = set(CompareTriplesExactMatch(t) for t in d_triples)
        d_triples_wo_rel = set(CompareTriplesWoRelation(t) for t in d_triples)
        counts[topic]['uniq_d_triples'] = len(d_triples_exact)
        
        # get extended triples
        d_extended_triples = None
        if docs_extended_triples is not None: 
            d_extended_triples = docs_extended_triples[topic]
            d_extended_triples = set(CompareTriplesWoRelation(t) for t in d_extended_triples)
        
        # get concepts
        d_concepts = None
        if docs_concepts is not None:
            d_concepts = docs_concepts[topic]

        for t in m_triples:
            
            counts[topic]['num_m_triples'] += 1
            t_exact = CompareTriplesExactMatch(t)
            if t_exact in m_triples_exact: continue
            
            m_triples_exact.add(t_exact)
            counts[topic]['uniq_m_triples'] += 1
             
            # if there is exact match
            if t_exact in d_triples_exact:
                counts[topic]['exact_match'] += 1
                
            else: # if only two concepts match (w/o relation)
                t_wo_rel = CompareTriplesWoRelation(t)
                if t_wo_rel in d_triples_wo_rel:
                    counts[topic]['wo_relation'] += 1
                    
                else: # if two concepts are in same sentence
                    if d_extended_triples is not None and t_wo_rel in d_extended_triples:
                        counts[topic]['same_sent'] += 1
                        
                    else: # if two concepts are in different sentences
                        if d_concepts is not None and t.concept1 in d_concepts and t.concept2 in d_concepts:
                            counts[topic]['diff_sent'] += 1
                            
                        else: # at least one concept is not in document set
                            counts[topic]['no_match'] += 1

    num_topics = 0
    num_uniq_m_triples = 0.0
    num_uniq_d_triples = 0.0
    per_uniq_m_triples = 0.0
    per_exact_match = 0.0
    per_wo_relation = 0.0
    per_same_sent = 0.0
    per_diff_sent = 0.0
    per_no_match = 0.0
    
    for topic in counts:
        num_topics += 1
        num_uniq_m_triples += counts[topic]['uniq_m_triples']
        num_uniq_d_triples += counts[topic]['uniq_d_triples']
        per_uniq_m_triples += counts[topic]['uniq_m_triples']/float(counts[topic]['num_m_triples'])
        per_exact_match += counts[topic]['exact_match']/float(counts[topic]['uniq_m_triples'])
        per_wo_relation += counts[topic]['wo_relation']/float(counts[topic]['uniq_m_triples'])
        per_same_sent += counts[topic]['same_sent']/float(counts[topic]['uniq_m_triples'])
        per_diff_sent += counts[topic]['diff_sent']/float(counts[topic]['uniq_m_triples'])
        per_no_match += counts[topic]['no_match']/float(counts[topic]['uniq_m_triples'])

    num_uniq_m_triples /= num_topics
    num_uniq_d_triples /= num_topics
    per_uniq_m_triples /= num_topics
    per_exact_match /= num_topics
    per_wo_relation /= num_topics
    per_same_sent /= num_topics
    per_diff_sent /= num_topics
    per_no_match /= num_topics
    
    result = '''
    ------
    number of total files: %d
    number of unique triples per summary: %.1f
    number of unique triples per document set: %.1f
    percentage of unique triples among all triples in summary: %.1f%%
    ------
    unique summary triples (exact match): %.1f%%
    unique summary triples (w/o relation): %.1f%%
    unique summary triples (same sentence): %.1f%%
    unique summary triples (diff sentence): %.1f%%
    unique summary triples (no match): %.1f%%
    ''' % (num_topics, num_uniq_m_triples, num_uniq_d_triples, per_uniq_m_triples*100,
           per_exact_match*100, per_wo_relation*100, per_same_sent*100, per_diff_sent*100, per_no_match*100)
    
    return result




