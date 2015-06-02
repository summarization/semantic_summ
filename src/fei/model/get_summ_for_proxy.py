#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import codecs
import logging
import os
import re

from collections import defaultdict

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def getSummForProxy(summ_file, output_folder):
    """
    generate one summary for each file, used for ROUGE evaluation
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, mode=0755)
        
    info_dict = {}
    summ_dict = defaultdict(lambda: [])
    
    # collect summary sentences from input file
    with codecs.open(summ_file, 'r', 'utf-8') as infile:
        for line in infile:
            line = line.rstrip()
            
            if line == '':
                filename, _ = info_dict['id'].split('.')
                sentence = info_dict['tok'].lower()
                words = [w for w in sentence.split() if re.search('[A-Za-z0-9]', w)]
                summ_dict[filename].append(' '.join(words))
                info_dict = {}
                continue
            
            if line.startswith('#'):
                fields = line.split('::')
                for field in fields[1:]:
                    tokens = field.split()
                    info_name = tokens[0]
                    info_body = ' '.join(tokens[1:])
                    info_dict[info_name] = info_body
                    
    # write summary sentences to files
    for curr_filename in summ_dict:
        output_filename = os.path.join(output_folder, curr_filename)
        with codecs.open(output_filename, 'w', 'utf-8') as outfile:
            lines = summ_dict[curr_filename]
            if not lines: 
                logger.debug('[%s] has no summary sentences.')
                continue
            for line in lines:
                outfile.write('%s\n' % line)
                
    return

if __name__ == '__main__':
    summ_file = '/Users/user/Data/SemanticSumm/Proxy/gold/split/test/aligned-amr-release-1.0-test-proxy-summary.txt'
    output_folder = '/Users/feiliu/Experiments/Semantic_Summ/goldstandard_summary'
    getSummForProxy(summ_file, output_folder)








