#!/bin/bash
python training/run_experiment.py --gpu=1 --save --subsample_fraction 0.1 '{"dataset": "IamParagraphsDataset","model": "LineDetectorModel", "network": "fcn", "train_args": {"batch_size": 64, "epochs": 32}}'
