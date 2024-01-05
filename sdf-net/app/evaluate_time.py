# The MIT License (MIT)
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import glob
import logging as log

from lib.trainer import Trainer
from lib.options import parse_options

from pathlib import Path

# Set logger display format
log.basicConfig(format='[%(asctime)s] [INFO] %(message)s', 
                datefmt='%d/%m %H:%M:%S',
                level=log.INFO)

if __name__ == "__main__":
    args, args_str = parse_options()
    dataset_root = Path(args.dataset_path)
    log.info(f'Training objects in {dataset_root}')
    assert os.path.isdir(dataset_root)
    obj_files = sorted([f for f in os.listdir(dataset_root) if f.endswith('.obj')])
    for file in obj_files:
        file_path = dataset_root / file
        log.info(f'Training on dataset: {file_path}')
        args.dataset_path = str(file_path)
        args.exp_name = f"train_{str(file)[:-4]}"
        trainer = Trainer(args, args_str)
        trainer.loss_lods = list(range(0, trainer.args.num_lods))
        for i in range(2):
            trainer.test_occupancy(i, log_f1=False)
