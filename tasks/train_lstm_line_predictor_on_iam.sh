#!/bin/bash
pipenv run python training/run_experiment.py --save --nowandb '{"dataset": "IamLinesDataset", "model": "LineModelCtc", "network": "line_lstm_ctc", "train_args":{ "batch_size" : 128, "epochs": 1, "lr_decay": 0.75}, "network_args": {"dropout_amount": 0.25, "num_layers": 3, "pooling": "avg", "padding":"same"}}'
