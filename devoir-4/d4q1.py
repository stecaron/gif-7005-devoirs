#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 21:02:45 2018

@author: stephanecaron
"""

###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 4, Question 1
#
###############################################################################
############################## INSTRUCTIONS ###################################
###############################################################################
#
# - RepÃ©rez les commentaires commenÃ§ant par TODO : ils indiquent une tÃ¢che que
#       vous devez effectuer.
# - Vous ne pouvez PAS changer la structure du code, importer d'autres
#       modules / sous-modules, ou ajouter d'autres fichiers Python
# - Ne touchez pas aux variables, TMAX*, ERRMAX* et _times, Ã  la fonction
#       checkTime, ni aux conditions vÃ©rifiant le bon fonctionnement de votre 
#       code. Ces structures vous permettent de savoir rapidement si vous ne 
#       respectez pas les requis minimum pour une question en particulier. 
#       Toute sous-question n'atteignant pas ces minimums se verra attribuer 
#       la note de zÃ©ro (0) pour la partie implÃ©mentation!
#
###############################################################################

import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torchvision

from d4utils import CODES_DE_SECTION
from d4utils import VolcanoesConv
from d4utils import VolcanoesDataset, VolcanoesLoader
from d4utils import compute_accuracy, compute_confusion_matrix


# TODO Logistique
# Mettre 'BACC' ou 'GRAD'
SECTION = 'GRAD'

# TODO Logistique
# Mettre son numÃ©ro d'Ã©quipe ici
NUMERO_EQUIPE = 24

# CrÃ©e la random seed
RANDOM_SEED = CODES_DE_SECTION[SECTION] + NUMERO_EQUIPE


class VolcanoesNet(nn.Module):
    """
    Cette classe dÃ©finit un rÃ©seau
    plainement convolutionnel simple
    permettant de classifier des images
    satellite de Venus.
    """

    def __init__(self):
        super().__init__()
        
        # TODO Q1A
        # DÃ©finir ici les couches de convolution
        # comme il est dÃ©crit dans l'Ã©noncÃ© du
        # devoir
        self.conv1 = VolcanoesConv(ch_in=1, ch_out=32, kernel=5)
        self.conv2 = VolcanoesConv(ch_in=32, ch_out=64, kernel=3)
        self.conv3 = VolcanoesConv(ch_in=64, ch_out=64, kernel=3)
        self.conv4 = VolcanoesConv(ch_in=64, ch_out=64, kernel=3)
        self.conv5 = VolcanoesConv(ch_in=64, ch_out=64, kernel=3)

        # TODO Q1A
        # DÃ©finir les couches de normalisation
        # permettant de maintenir les valeurs
        # du rÃ©seau Ã  des valeurs raisonnables
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(64)
        self.norm3 = nn.BatchNorm2d(64)
        self.norm4 = nn.BatchNorm2d(64)
        self.norm5 = nn.BatchNorm2d(64)


        # TODO Q1A
        # DÃ©ninir la couche linÃ©aire de sortie
        self.lin1 = nn.Linear(64, 1)


    def conv_forward(self, x):
        # TODO Q1B
        # Ã‰crire les lignes de code pour l'infÃ©rence
        # des couches de convolution, avant l'average 
        # pooling
        y = F.relu(self.norm1(self.conv1(x)))
        y = F.relu(self.norm2(self.conv2(y)))
        y = F.relu(self.norm3(self.conv3(y)))
        y = F.relu(self.norm4(self.conv4(y)))
        y = F.relu(self.norm5(self.conv5(y)))

        return y

    def forward(self, x):
        # SÃ©lectionne la taille de l'entrÃ©e
        batch_size = x.size()[0]

        # ExÃ©cute la partie convolution
        y = self.conv_forward(x)

        # Fait un average pooling
        y = y.view(batch_size, 64, -1).mean(dim=2)

        return torch.sigmoid(self.lin1(y))


if __name__ == '__main__':
    # DÃ©finit la seed
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    # Des effets stochastiques peuvent survenir
    # avec cudnn, mÃªme si la seed est activÃ©e
    # voir le thread: https://bit.ly/2QDNxRE
    torch.backends.cudnn.deterministic = True

    # DÃ©finit si cuda est utilisÃ© ou non
    # mettre cuda pour utiliser un GPU
    device = 'cpu'

    # DÃ©finit les paramÃ¨tres d'entraÃ®nement
    # Nous vous conseillons ces paramÃ¨tres. 
    # Cependant, vous pouvez les changer
    nb_epoch = 10
    learning_rate = 0.01
    momentum = 0.9
    batch_size = 32

    # Charge les donnÃ©es d'entraÃ®nement et de test
    train_set = VolcanoesDataset('data/Volcanoes_train.pt.gz')
    test_set = VolcanoesDataset('data/Volcanoes_test.pt.gz')

    # CrÃ©e le dataloader d'entraÃ®nement
    train_loader = VolcanoesLoader(train_set, batch_size=batch_size, \
        balanced=True, random_seed=RANDOM_SEED)
    test_loader = VolcanoesLoader(test_set, batch_size=batch_size, 
        balanced=True, random_seed=RANDOM_SEED)

    # TODO Q1C 
    # Instancier un rÃ©seau VolcanoesNet
    # dans une variable nommÃ©e "model"
    model = VolcanoesNet()

    # Tranfert le rÃ©seau au bon endroit
    model.to(device)

    # TODO Q1C
    # Instancier une fonction d'erreur BinaryCrossEntropy
    # et la mettre dans une variable nommÃ©e criterion
    criterion = nn.BCELoss()


    # TODO Q1C 
    # Instancier l'algorithme d'optimisation SGD
    # Ne pas oublier de lui donner les hyperparamÃ¨tres
    # d'entraÃ®nement : learning rate et momentum!
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)


    # TODO Q1C
    # Mettre le rÃ©seau en mode entraÃ®nement
    model.train()


    # TODO Q1C
    # Remplir les TODO dans la boucle d'entraÃ®nement
    for i_epoch in range(nb_epoch):

        start_time, train_losses = time.time(), []
        for i_batch, batch in enumerate(train_loader):
            images, targets = batch

            images = images.to(device)
            targets = targets.to(device)

            # TODO Q1C 
            # Mettre les gradients Ã  zÃ©ro
            optimizer.zero_grad()


            # TODO Q1C
            # Calculer:
            # 1. l'infÃ©rence dans une variable "predictions"
            # 2. l'erreur dans une variable "loss"
            predictions = model(images)
            loss = criterion(predictions, targets)


            # TODO Q1C
            # RÃ©tropropager l'erreur et effectuer
            # une Ã©tape d'optimisation
            loss.backward()
            optimizer.step()


            # Ajoute le loss de la batch
            train_losses.append(loss.item())

        print(' [-] epoch {:4}/{:}, train loss {:.6f} in {:.2f}s'.format(
            i_epoch+1, nb_epoch, np.mean(train_losses), time.time()-start_time))

        # sauvegarde le rÃ©seau
        torch.save(model.state_dict(), 'volcanoes_model.pt')

    # affiche le score Ã  l'Ã©cran
    test_acc = compute_accuracy(model, test_loader, device)
    print(' [-] test acc. {:.6f}%'.format(test_acc * 100))

    # affiche la matrice de confusion Ã  l'Ã©cran
    matrix = compute_confusion_matrix(model, test_loader, device)
    print(' [-] conf. mtx. : \n{}'.format(matrix))



