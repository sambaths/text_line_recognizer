#!/bin/bash
pipenv run python training/run_experiment.py --save --gpu=0 '{"dataset": "EmnistLinesDataset", "model": "LineModelCtc", "network": "line_lstm_ctc"}'
