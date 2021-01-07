import torch
import numpy as np

import data_loader
from FullyConnectedModel import EvidentialRegression
from EvidentialTrainer import EvidentialTrainer


datasets = ['boston', 'concrete', 'energy-efficiency', 'kin8nm',
            'naval', 'power-plant', 'protein', 'wine', 'yacht']
datasets = ['boston', 'concrete', 'kin8nm', 'naval',
            'power-plant', 'protein', 'wine', 'yacht']
num_trials = 20
num_epochs = 40

h_params = {
    'yacht': {'learning_rate': 5e-4, 'batch_size': 1},
    'naval': {'learning_rate': 5e-4, 'batch_size': 1},
    'concrete': {'learning_rate': 5e-3, 'batch_size': 1},
    'energy-efficiency': {'learning_rate': 2e-3, 'batch_size': 1},
    'kin8nm': {'learning_rate': 1e-3, 'batch_size': 1},
    'power-plant': {'learning_rate': 1e-3, 'batch_size': 2},
    'boston': {'learning_rate': 1e-3, 'batch_size': 8},
    'wine': {'learning_rate': 1e-4, 'batch_size': 32},
    'protein': {'learning_rate': 1e-3, 'batch_size': 64},
}


RMSE = np.zeros((len(datasets), num_trials))
NLL = np.zeros((len(datasets), num_trials))
for di, dataset in enumerate(datasets):
    batch_size = h_params[dataset]["batch_size"]
    learning_rate = h_params[dataset]["learning_rate"]
    for n in range(num_trials):
        (x_train, y_train), (x_test, y_test), y_scale = data_loader.load_dataset(
            dataset, return_as_tensor=True)
        num_iterations = num_epochs * x_train.shape[0]//batch_size

        model = EvidentialRegression(x_train.size(1))
        trainer = EvidentialTrainer(model, learning_rate=learning_rate)
        rmse, nll = trainer.train(x_train, y_train, x_test, y_test, y_scale.item(
        ), iters=num_iterations, batch_size=batch_size, verbose=True)
        del model
        
        print("Trial: {} Saving {} {}".format(n, rmse, nll))
        RMSE[di, n] = rmse
        NLL[di, n] = nll

RESULTS = np.hstack((RMSE, NLL))
mu = RESULTS.mean(axis=-1)
error = np.std(RESULTS, axis=-1)
