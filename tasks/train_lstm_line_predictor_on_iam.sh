#!/bin/bash
pipenv run python training/run_experiment.py --save --gpu=0 '{"dataset": "IamLinesDataset", "model": "LineModelCtc", "network": "line_lstm_ctc", "train_args":{ "batch_size" : 64, "epochs": 32}}'
