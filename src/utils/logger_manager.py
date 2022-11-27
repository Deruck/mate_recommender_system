import os
import logging
import datetime
from pathlib import Path

logger_name = "mate_rec_logger"

class LoggerManager:
    
    @staticmethod
    def set_logger(log_file_path: Path):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter(
                fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')
        )

        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter(
                fmt='%(asctime)s - %(filename)s - %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')
        )
        logger.addHandler(console_handler)

    @staticmethod
    def get_logger() -> logging.Logger:
        return logging.getLogger(logger_name)