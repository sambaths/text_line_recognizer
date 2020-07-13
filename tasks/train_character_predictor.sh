#!/bin/bash
pipenv run python training/run_experiment.py --save --gpu=0 '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp", "train_args": {"batch_size": 256}}'
