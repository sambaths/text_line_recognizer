#!/bin/bash
pipenv run python training/run_experiment.py --gpu=0 --save '{"dataset": "IamParagraphsDataset","model": "LineDetectorModel", "network": "fcn", "train_args": {"batch_size": 64, "epochs": 32}}'
