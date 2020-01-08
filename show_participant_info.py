#!/usr/bin/env python3
from pathops import dir_must_exist
import os
import dill
import numpy as np
import pdb
import json
from natsort import natsorted
import random
random.seed(42)
np.random.seed(42)
import itertools
import copy
import logging
from loggerops import create_logger, log_newline
import shutil
import os
from pathlib import Path
from datetime import datetime
import re

import argparse

logger = logging.getLogger(__name__)
nowtime = datetime.now()

from gen_participants import Participant

from config import server, socketio, participants

def main(i):

    key = "participant_{}".format(i)
    participant = Participant(participant_dir="./participant_data/{}".format(key),
                number=i)
    participant.load('parameters')
    # Log all parameters of the current participant
    for key, val in participant.parameters.items():
        if type(val) is np.ndarray:
            val = val.tolist()
        trunc_str = re.sub(r'^(.{75}).*$', '\g<1>...', f"{key:<25}{val}")
        logger.info(f"{trunc_str: <78}")
    logger.info("-"*78)

if __name__ == '__main__':
    logs_dir = Path('./logs/')
    logs_dir.mkdir(exist_ok=True)
    logfile_dir =  logs_dir / __file__
    logfile_dir.mkdir(exist_ok=True)
    logfile_name = nowtime.strftime("%m-%d-%Y_%H-%M-%S")+'.log'
    logger = create_logger(
        logger_streamlevel=10,
        log_filename=str(logfile_dir/logfile_name),
        logger_filelevel=10
    )
    parser = argparse.ArgumentParser(description='Show info about participant',)
    parser.add_argument('index', type=int,
                        help='participant index')
    args=parser.parse_args()
    main(args.index)
