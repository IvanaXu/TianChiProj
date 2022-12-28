#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import logging
import os
from logging import handlers
from dataclasses import dataclass, asdict

LOG_FORMAT = "%(asctime)s %(message)s"
formatter = logging.Formatter(LOG_FORMAT) #, datefmt="%H:%M:%S"

class Logger:
    log_dir = os.environ['HOME']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    CONFIG_OUTPUT_FILE = f"{log_dir}/matrix.log"

    @staticmethod
    def config_output_file(file_path:str) -> None:
        Logger.CONFIG_OUTPUT_FILE = file_path if file_path else Logger.CONFIG_OUTPUT_FILE

    @staticmethod
    def get_logger(name:str, log_file:str=None, level=logging.INFO, propagate:bool=False):
        """To setup as many loggers as you want"""
        name = name.split(".")[-1]
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if logger.hasHandlers():
            return logger
        
        # std stream print
        handler = logging.StreamHandler()        
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # file stream print
        if log_file is None:
            log_file = Logger.CONFIG_OUTPUT_FILE
        filehandler = handlers.RotatingFileHandler(log_file, mode='a', maxBytes=1024*1024*1024, backupCount=20)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
        logger.propagate = propagate

        return logger
    