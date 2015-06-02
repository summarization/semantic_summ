#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

def getLogger(log_name='', log_file='file.log'):
    """
    get logger
    """
    # logging.getLogger() is a singleton
    logger = logging.getLogger(log_name)
    formatter = logging.Formatter(logging.BASIC_FORMAT)

    if not len(logger.handlers):

        # add stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
        # add file handler
        file_handler_info = logging.FileHandler(log_file, mode='w')
        file_handler_info.setFormatter(formatter)
        file_handler_info.setLevel(logging.DEBUG)
        logger.addHandler(file_handler_info)
    
        logger.setLevel(logging.DEBUG)

    return logger

