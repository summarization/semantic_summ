#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def getJamrForProxy(gold_file, jamr_file, output_file):
    """
    replace gold AMR annotations with JAMR parses 
    """
    all_parses = [] # JAMR parses
    curr_parse = [] # line starts with '# ::tok'
                    # line starts with '# ::alignments'
                    # line contains the JAMR parse
    
    # obtain JAMR parses
    with codecs.open(jamr_file, 'r', 'utf-8') as infile:
        for line in infile:
            line = line.rstrip()
            if line == '' and curr_parse:
                all_parses.append(curr_parse)
                curr_parse = []
                continue
            if (line.startswith('# ::tok') 
                or line.startswith('# ::alignments')
                or line.startswith('(')
                or line.startswith(' ')):
                curr_parse.append(line)
    # include the last parse
    if curr_parse: all_parses.append(curr_parse)
    logger.debug('[Number of JAMR parses]: %d', len(all_parses))

    all_metas = []  # gold meta information
    curr_meta = []  # line starts with '# ::id'
                    # line starts with '# ::snt'
                    # line starts with '# ::save-date'
    
    # obtain meta information from gold AMR annotations
    with codecs.open(gold_file, 'r', 'utf-8') as infile:
        for line in infile:
            line = line.rstrip()
            if line == '' and curr_meta:
                all_metas.append(curr_meta)
                curr_meta = []
                continue
            if (line.startswith('# ::id') 
                or line.startswith('# ::snt')
                or line.startswith('# ::save-date')):
                curr_meta.append(line)
    # include the last meta info
    if curr_meta: all_metas.append(curr_meta)
    logger.debug('[Number of gold metas]: %d', len(all_metas))
    
    assert len(all_parses) == len(all_metas)
    
    # write the combined information to file
    with codecs.open(output_file, 'w', 'utf-8') as outfile:
        for meta, parse in zip(all_metas, all_parses):
            for m in meta: outfile.write('%s\n' % m)
            for p in parse: outfile.write('%s\n' % p)
            outfile.write('\n')
    
    return


if __name__ == '__main__':
    gold_file = '/home/user/Data/Proxy/jamr/split/test/amr-release-1.0-test-proxy.txt'
    jamr_file = '/home/user/Data/Proxy/jamr/split/test/amr-release-1.0-test-proxy.txt.jamr'
    output_file = '/home/user/Data/Proxy/jamr/split/test/jamr-aligned-amr-release-1.0-test-proxy.txt'
    getJamrForProxy(gold_file, jamr_file, output_file)











    
    
