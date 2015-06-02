#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

class Triple(object):
    def __init__(self, 
                 concept1=None, concept2=None, relation=None,
                 filename=None, line_num=None, sentence=None):
        """
        initialization
        two concepts and relation should be lowercased
        """
        self.concept1 = concept1
        self.concept2 = concept2
        self.relation = relation
        self.filename = filename
        self.line_num = line_num
        self.sentence = sentence
    
    def __repr__(self):
        return '%s (%s)-> %s' % (self.concept1, self.relation, self.concept2)
        
class CompareTriplesExactMatch(object):
    """
    two concepts are the same, 
    relation and direction of relation should be the same as well
    """
    def __init__(self, triple):
        self.triple = triple

    def __hash__(self):
        return hash((self.triple.concept1, self.triple.concept2, self.triple.relation))

    def __eq__(self, other):
        """
        require match of relation, both concepts, and direction
        """
        if type(other) == type(self):
            if other.triple.relation == self.triple.relation:
                if (other.triple.concept1 == self.triple.concept1 
                    and other.triple.concept2 == self.triple.concept2):
                    return True
        return False
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __repr__(self):
        return '%s (%s)-> %s' % (self.triple.concept1, self.triple.relation, self.triple.concept2)

class CompareTriplesWoRelation(object):
    """
    two concepts are the same, ignore order
    equal triples should have equal values for both __hash__ and __eq__
    """
    def __init__(self, triple):
        self.triple = triple

    def __hash__(self):
        return hash((self.triple.concept1, self.triple.concept2))

    def __eq__(self, other):
        if type(other) == type(self):
            if (other.triple.concept1 == self.triple.concept1 
                and other.triple.concept2 == self.triple.concept2):
                return True
        return False
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __repr__(self):
        return '%s -> %s' % (self.triple.concept1, self.triple.concept2)


if __name__ == '__main__':
    t1 = Triple('c1', 'c2', 'r1')
    t2 = Triple('c2', 'c1', 'r2')
    t3 = Triple('c1', 'c3', 'r3')
    set1 = set(CompareTriplesWoRelation(t) for t in [t1, t2])
    set2 = set(CompareTriplesWoRelation(t) for t in [t1, t3])
    print set1
    print set2
    print set1 & set2







