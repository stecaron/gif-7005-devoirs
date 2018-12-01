#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 21:03:40 2018

@author: stephanecaron
"""

###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 4, Code utilitaire
#
###############################################################################

import gzip
import random

import numpy as np

from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn


CODES_DE_SECTION = {
    'BACC': 4101,
    'GRAD': 7005
}


class VolcanoesDataset:
    """
    Cette classe sert Ã  dÃ©finir le dataset Volcanoes pour PyTorch
    dataset venant de Francisco Mena sur kaggle : https://bit.ly/2DasPF1

    Args:
        path (str): le chemin du fichier .pt du dataset
    """

    def __init__(self, path):
        # garde les paramÃ¨tres en mÃ©moire
        self.path = path

        # charger les donnÃ©es
        with gzip.open(path, 'rb') as f:
            self.data = torch.load(f)
        self.targets = np.array(list(zip(*self.data))[1])

    def __getitem__(self, index):
        return self.data[index]


class VolcanoesLoader:
    """
    Cette classe sert Ã  charger le dataset Volcanoes pour PyTorch
    lors de l'entraÃ®nement en Ã©quilibrant le dataset

    Args:
        dataset (VolcanoesDataset): le dataset Ã  utiliser
        batch_size (int): la taille de la batch Ã  utiliser
    """

    def __init__(self, dataset, batch_size, balanced=True, random_seed=42):
        # Garde les paramÃ¨tres en mÃ©moire
        self.dataset = dataset
        self.batch_size = batch_size
        self.balanced = balanced

        # Calcul les indices et la tailles des exemples positifs
        # et nÃ©gatif. Comme Ã§a, on peut rebalancer le dataset
        self.pos_idx = np.where(self.dataset.targets == 1)[0].tolist()
        self.neg_idx = np.where(self.dataset.targets == 0)[0].tolist()
        self.pos_size = len(self.pos_idx)
        self.neg_size = len(self.neg_idx)

        # DÃ©finit la random seed
        self.random = random.Random(random_seed)

    def __next__(self):
        # VÃ©rifie si l'epoch est finie
        if self.i_batch == self.nb_batch:
            raise StopIteration

        # Calcul les indices de la batch
        start = self.i_batch * self.batch_size
        end = start + self.batch_size
        idx = self.indices[start:end]

        # SÃ©lectionne la batch
        X, y = [], []
        for i in idx:
            X.append(self.dataset.data[i][0])
            y.append(self.dataset.data[i][1])

        # Convertit les listes en TorchTensor
        X = torch.stack(X)
        y = torch.stack(y).float().view(-1, 1)

        # IncrÃ©mente le compteur
        self.i_batch += 1

        return X, y

    def __iter__(self):
        # Choisi la bonne mÃ©thode
        if self.balanced:
            # Ã‰chantillonne un nombre d'exemples nÃ©gatifs pour l'epoch
            neg_sample = self.random.sample(self.neg_idx, k=self.pos_size)

            # CrÃ©e l'ensemble d'indices pour l'epoch
            self.indices = self.pos_idx + neg_sample
            self.random.shuffle(self.indices)
        else:
            self.indices = list(range(len(self.dataset.data)))
            self.random.shuffle(self.indices)

        # Calcul le nombre de batch et dÃ©bute le compteur
        self.nb_batch = len(self.indices) // self.batch_size
        self.i_batch = 0

        return self


class VolcanoesConv(nn.Module):
    """
    Cette classe sert Ã  dÃ©finir la convolution
    utilisÃ©e dans le cadre du rÃ©seau VolcanoesNet
    """

    def __init__(self, ch_in, ch_out, kernel):
        super().__init__()
        padding = kernel // 2
        self.model = nn.Conv2d(ch_in, ch_out, \
            kernel, stride=2, padding=padding)

    def forward(self, x):
        return self.model(x)


def compute_accuracy(model, dataloader, device='cpu'):
    training_before = model.training
    model.eval()

    all_predictions = []
    all_targets = []
    for i_batch, batch in enumerate(dataloader):
        images, targets = batch

        images = images.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            predictions = model(images)

        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    if all_predictions[0].shape[-1] > 1:
        predictions_numpy = np.concatenate(all_predictions, axis=0)
        predictions_numpy = predictions_numpy.argmax(axis=1)
        targets_numpy = np.concatenate(all_targets, axis=0)
    else:
        predictions_numpy = np.ravel(all_predictions)
        targets_numpy = np.ravel(all_targets)
        predictions_numpy[predictions_numpy >= 0.5] = 1.0
        predictions_numpy[predictions_numpy < 0.5] = 0.0

    if training_before:
        model.train()

    return (predictions_numpy == targets_numpy).mean()


def compute_confusion_matrix(model, dataloader, device='cpu'):
    training_before = model.training
    model.eval()

    all_predictions = []
    all_targets = []
    dataloader.balanced = False
    for i_batch, batch in enumerate(dataloader):
        images, targets = batch

        images = images.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            predictions = model(images)

        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    dataloader.balanced = True

    predictions_numpy = np.ravel(all_predictions)
    targets_numpy = np.ravel(all_targets)

    predictions_numpy[predictions_numpy >= 0.5] = 1.0
    predictions_numpy[predictions_numpy < 0.5] = 0.0

    if training_before:
        model.train()

    matrix = confusion_matrix(targets_numpy, predictions_numpy)
    matrix = matrix / matrix.sum(axis=1)[:, np.newaxis]

    return matrix