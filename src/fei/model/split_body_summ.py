#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import codecs
import numpy
import logging

from collections import Counter

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def getLines(input_file, snt_type):
    """
    get lines (AMR graphs) from input file.
    these are lines of certain snt-type: body, summary, topic, country, date
    """
    lines = []
    snt_type = '::snt-type ' + snt_type
    snt_flag = False
    
    with codecs.open(input_file, 'r', 'utf-8') as infile:
        for line in infile:
            line = line.rstrip() # keep indentation

            if not line:
                if snt_flag == True:
                    lines.append(line) # keep an empty line
                    snt_flag = False
                continue
                
            if line.startswith('# ::id') and snt_type in line:
                snt_flag = True
                lines.append(line)
                continue
            
            if snt_flag == True:
                lines.append(line)  
    return lines

def getStats(input_file):
    """
    calculate 1) number of documents; 2) number of sentences;
    3) number of body/summary sentences.
    """
    total_body_sents = Counter()
    total_summary_sents = Counter()
    total_sents = Counter()
    
    with codecs.open(input_file, 'r', 'utf-8') as infile:
        for line in infile:
            line = line.rstrip()
            
            if line.startswith('# ::id'):
                filename = line.split()[2]
                filename = re.sub(r'\.[0-9]+$', '', filename)
                
                # file has no summary; needs to be excluded
                if filename == 'PROXY_AFP_ENG_20030126_0212': continue
                
                total_sents[filename] += 1
                if ' ::snt-type body' in line:
                    total_body_sents[filename] += 1
                if ' ::snt-type summary' in line:
                    total_summary_sents[filename] += 1
    
    num_docs = len(total_sents)
    
    sents_avg = numpy.mean(numpy.array(total_sents.values()))
    sents_std = numpy.std(numpy.array(total_sents.values()))

    body_avg = numpy.mean(numpy.array(total_body_sents.values()))
    body_std = numpy.std(numpy.array(total_body_sents.values()))
    
    summary_avg = numpy.mean(numpy.array(total_summary_sents.values()))
    summary_std = numpy.std(numpy.array(total_summary_sents.values()))
    
    logger.debug('--------')
    logger.debug('[filename]: %s' % (os.path.basename(input_file)))
    logger.debug('[num_docs]: %d' % (num_docs))
    logger.debug('[num_sents]: %.1f (+/-%.1f)' % (sents_avg, sents_std))
    logger.debug('[num_body_sents]: %.1f (+/- %.1f)' % (body_avg, body_std))
    logger.debug('[num_summary_sents]: %.1f (+/- %.1f)' % (summary_avg, summary_std))

    return

def splitProxyFile(input_dir):
    """
    split the "proxy" file in input_dir
    """
    for curr_filename in os.listdir(input_dir):
        if 'proxy.txt' not in curr_filename: continue
        curr_file = os.path.join(input_dir, curr_filename)
        
        body_lines = getLines(curr_file, snt_type='body')
        summary_lines = getLines(curr_file, snt_type='summary')
        getStats(curr_file)
    
        # file contains only "body" sentences
        output_file = re.sub(r'\.txt$', '-body.txt', curr_file)
        with codecs.open(output_file, 'w', 'utf-8') as outfile:
            outfile.write('%s\n' % '\n'.join(body_lines))
        
        # file contains only "summary" sentences
        output_file = re.sub(r'\.txt$', '-summary.txt', curr_file)
        with codecs.open(output_file, 'w', 'utf-8') as outfile:
            outfile.write('%s\n' % '\n'.join(summary_lines))

if __name__ == '__main__':
    
#     # separate summary sentences from body sentences
#     input_dir = '/home/user/Data/Proxy/gold/split'
#      
#     # loop through training/dev/test folders
#     for curr_dirname in os.listdir(input_dir):
#         curr_dir = os.path.join(input_dir, curr_dirname)
#         if not os.path.isdir(curr_dir): continue
#         splitProxyFile(curr_dir) # split proxy file
     
#     input_dir = '/home/user/Data/Proxy/gold/unsplit'
#     splitProxyFile(input_dir)
    
#     # calculate statistics on num_sentences
#     input_dir = '/Users/user/Data/SemanticSumm/Proxy/gold/split/dev'
#     input_filename = 'amr-release-1.0-dev-proxy.txt'
#     getStats(os.path.join(input_dir, input_filename))
    
    input_dir = '/home/user/Data/Proxy/jamr/split/test'
    splitProxyFile(input_dir)
    
    


