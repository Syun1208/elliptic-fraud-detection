import os
import sys
import datetime
import logging
import colorlog
from pathlib import Path

from src.utils.constants import *



FILE = Path(__file__).resolve()
WORK_DIR = FILE.parents[2]


class Logger:

    __slots__ = ['log_dir', 'log_clear_days', 'running_date']

    def __init__(
        self,
        log_dir: str,
        log_clear_days: int
    ) -> None:
        self.log_dir = os.path.join(WORK_DIR, log_dir)
        self.log_clear_days = log_clear_days
        self.running_date = str(datetime.date.today())

    def handle_log_files(self):
        if os.path.exists(self.log_dir):
            log_files = os.listdir(self.log_dir)
            for file in log_files:
                file_path = os.path.join(self.log_dir, file)
                file_date = os.path.getmtime(file_path)
                if datetime.datetime.now() - datetime.datetime.fromtimestamp(file_date) > datetime.timedelta(days=self.log_clear_days):
                    os.remove(file_path)
        else:
            os.makedirs(self.log_dir)


    def get_tracking(self, module_name):
        # Create a logger object with the name of the module
        logger = logging.getLogger(module_name)

        # Set the logging level
        logger.setLevel(logging.DEBUG)

        # Create a formatter object
        formatter = colorlog.ColoredFormatter(
            '%(purple)s%(asctime)s%(reset)s - %(blue)s%(name)s%(reset)s - %(log_color)s%(levelname)s%(reset)s - %(green)s%(message)s%(reset)s',
            reset=True,
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        save_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Create a console handler
        console_handler = colorlog.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # # Create a file handler
        os.makedirs(self.log_dir, exist_ok=True)
        # self.handle_log_files()
        
        file_handler = logging.FileHandler('{}/{}.log'.format(self.log_dir, self.running_date))
        file_handler.setFormatter(save_formatter)
        logger.addHandler(file_handler)

        return logger