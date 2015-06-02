#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import sys
from fei.model.utils import getLogger

LOG_FILE = 'semantic_summ.log'
if len(sys.argv) > 4: LOG_FILE = 'log_%s_%s_passes_len_%s_exp_%s' % (sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
logger = getLogger(log_file=LOG_FILE)

from fei.model.corpus import buildCorpus
from fei.model.decoder import Decoder
from fei.model.learning import ParamEstimator

def train(body_file, summ_file, param_file, loss_func, num_passes, oracle_len, w_exp):
    """
    run summarizer, learn structured prediction parameters
    """    
    logger.debug('start training...')
    logger.debug('[settings]: %s_%d_passes_len_%s_exp_%d' % (loss_func, num_passes, oracle_len, w_exp))
    corpus = buildCorpus(body_file, summ_file, w_exp)
    
    # learn parameters
    decoder = Decoder()
    estimator = ParamEstimator()
    final_weights = estimator.learnParamsAdaGrad(decoder, corpus, param_file, loss_func, num_passes, oracle_len)
    
    # output parameters to file
    with codecs.open(param_file, 'w', 'utf-8') as outfile:
        outfile.write('#num_passes#: %d\n' % num_passes)
        outfile.write('%s\n' % final_weights.toString())
    return

def test(body_file, summ_file, param_file, oracle_len, w_exp):
    """
    run summarizer, perform structured prediction
    """
    logger.debug('start testing...')
    logger.debug('[settings]: len_%s_exp_%d' % (oracle_len, w_exp))
    corpus = buildCorpus(body_file, summ_file, w_exp)
    
    # load parameters from file
    decoder = Decoder()
    decoder.weights.load(param_file)
    
    # perform structured prediction
    estimator = ParamEstimator()
    estimator.predict(decoder, corpus, oracle_len)
    
    return

def summ(body_file, summ_file, param_file, oracle_len, w_exp, jamr=False):
    """
    run summarizer, perform structured prediction
    """
    logger.debug('start testing...')
    logger.debug('[settings]: len_%s_exp_%d' % (oracle_len, w_exp))
    corpus = buildCorpus(body_file, summ_file, w_exp)
    
    # load parameters from file
    decoder = Decoder()
    decoder.weights.load(param_file)
    
    # perform structured prediction
    estimator = ParamEstimator()
    output_folder = param_file.replace('params', 'summ')
    if jamr == True: output_folder = param_file.replace('params', 'jamr_summ')
    estimator.summarize(decoder, corpus, oracle_len, output_folder)
    
    return

if __name__ == '__main__':
    
    # local
#     train_body_file = '/Users/user/Data/SemanticSumm/Proxy/gold/split/training/amr-release-1.0-training-proxy-body.txt'
#     train_summ_file = '/Users/user/Data/SemanticSumm/Proxy/gold/split/training/amr-release-1.0-training-proxy-summary.txt'
#     dev_body_file = '/Users/user/Data/SemanticSumm/Proxy/gold/split/dev/amr-release-1.0-dev-proxy-body.txt'
#     dev_summ_file = '/Users/user/Data/SemanticSumm/Proxy/gold/split/dev/amr-release-1.0-dev-proxy-summary.txt'
#     test_body_file = '/Users/user/Data/SemanticSumm/Proxy/gold/split/test/amr-release-1.0-test-proxy-body.txt'
#     test_summ_file = '/Users/user/Data/SemanticSumm/Proxy/gold/split/test/amr-release-1.0-test-proxy-summary.txt'
#      
#     num_passes = 1
#     loss_func = 'ramp'
#     oracle_len = 'edges'
#     w_exp = 0
#     param_file = 'params_%s_%d_passes_len_%s_exp_%s' % (loss_func, num_passes, oracle_len, w_exp)
#  
#     train(dev_body_file, dev_summ_file, param_file, loss_func, num_passes, oracle_len, w_exp)
#     test(dev_body_file, dev_summ_file, param_file, oracle_len, w_exp)
    
    # supercomputer (train/test on gold)
    train_body_file = '/home/user/Data/Proxy/gold/split/training/aligned-amr-release-1.0-training-proxy-body.txt'
    train_summ_file = '/home/user/Data/Proxy/gold/split/training/aligned-amr-release-1.0-training-proxy-summary.txt'
    dev_body_file = '/home/user/Data/Proxy/gold/split/dev/aligned-amr-release-1.0-dev-proxy-body.txt'
    dev_summ_file = '/home/user/Data/Proxy/gold/split/dev/aligned-amr-release-1.0-dev-proxy-summary.txt'
    test_body_file = '/home/user/Data/Proxy/gold/split/test/aligned-amr-release-1.0-test-proxy-body.txt'
    test_summ_file = '/home/user/Data/Proxy/gold/split/test/aligned-amr-release-1.0-test-proxy-summary.txt'
    jamr_test_body_file = '/home/user/Data/Proxy/jamr/split/test/jamr-aligned-amr-release-1.0-test-proxy-body.txt'

    param_file = 'params_%s_%s_passes_len_%s_exp_%s' % (sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    loss_func = sys.argv[1]
    num_passes = int(sys.argv[2])
    oracle_len = sys.argv[3]
    w_exp = int(sys.argv[4])
    
    # train model
    train(train_body_file, train_summ_file, param_file, loss_func, num_passes, oracle_len, w_exp) 
    test(dev_body_file, dev_summ_file, param_file, oracle_len, w_exp)
     
    # test on gold and jamr
    test(test_body_file, test_summ_file, param_file, oracle_len, w_exp)
    test(jamr_test_body_file, test_summ_file, param_file, oracle_len, w_exp)
     
    # summarize on gold and jamr
    summ(test_body_file, test_summ_file, param_file, oracle_len, w_exp)
    summ(jamr_test_body_file, test_summ_file, param_file, oracle_len, w_exp, jamr=True)

        
        
        
    
    
