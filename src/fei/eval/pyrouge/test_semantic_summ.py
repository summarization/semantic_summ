#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from fei.eval.pyrouge.Rouge155 import Rouge155 
import os

if __name__ == "__main__":

    rouge_dir = '/Users/user/Softwares/ROUGE/RELEASE-1.5.5'
    rouge_args = '-e /Users/user/Softwares/ROUGE/RELEASE-1.5.5/data -n 4 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -a -x -l 100'
     
    rouge = Rouge155(rouge_dir, rouge_args)
    rouge.model_dir = '/Users/user/Experiments/Semantic_Summ/goldstandard_summary'
    rouge.model_filename_pattern = 'PROXY_[A-Z]{3}_ENG_#ID#'
    
    # system results
#     summ_dir = '/Users/user/Experiments/Semantic_Summ/20141201_AdaGrad_5_Passes_Summ_Jamr/'
    summ_dir = '/Users/user/Experiments/Semantic_Summ/20141203_Summ_Results/'
    for foldername in os.listdir(summ_dir):
        if not (foldername.startswith('summ') or foldername.startswith('jamr')): continue
        curr_folder = os.path.join(summ_dir, foldername)
        if not os.path.isdir(curr_folder): continue
    
        rouge.system_dir = curr_folder
        rouge.system_filename_pattern = 'PROXY_[A-Z]{3}_ENG_([0-9_]+)_system'
        
        rouge_output = rouge.evaluate()    
        output_dict = rouge.output_to_dict(rouge_output)
        
        print '[system][foldername]: %s' % foldername
        print '[prec]: %.1f%% [rec]: %.1f%% [fscore]: %.1f%%' % (output_dict['rouge_1_precision'] * 100,
                                                                 output_dict['rouge_1_recall'] * 100,
                                                                 output_dict['rouge_1_f_score'] * 100)
    # oracle results
    for foldername in os.listdir(summ_dir):
        if not (foldername.startswith('summ') or foldername.startswith('jamr')): continue
        curr_folder = os.path.join(summ_dir, foldername)
        if not os.path.isdir(curr_folder): continue
    
        rouge.system_dir = curr_folder
        rouge.system_filename_pattern = 'PROXY_[A-Z]{3}_ENG_([0-9_]+)_oracle'
        
        rouge_output = rouge.evaluate()    
        output_dict = rouge.output_to_dict(rouge_output)
        
        print '[oracle][foldername]: %s' % foldername
        print '[prec]: %.1f%% [rec]: %.1f%% [fscore]: %.1f%%' % (output_dict['rouge_1_precision'] * 100,
                                                                 output_dict['rouge_1_recall'] * 100,
                                                                 output_dict['rouge_1_f_score'] * 100)


