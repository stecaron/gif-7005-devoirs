#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 21:01:05 2018

@author: stephanecaron
"""

# -*- coding: utf-8 -*-
###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 3, Question 2
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

import itertools
import time
import numpy
import warnings
from io import BytesIO
from http.client import HTTPConnection

from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import minmax_scale
from sklearn.svm import SVC

# Nous ne voulons pas avoir ce type d'avertissement, qui
# n'est pas utile dans le cadre de ce devoir
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Fonctions utilitaires liÃ©es Ã  l'Ã©valuation
_times = []
def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps Ã  s'exÃ©cuter! ".format(question)+
            "Le temps maximum permis est de {0:.4f} secondes, mais votre code a requis {1:.4f} secondes! ".format(maxduration,duration)+
            "Assurez-vous que vous ne faites pas d'appels bloquants (par exemple Ã  show()) dans cette boucle!") 

# DÃ©finition des durÃ©es d'exÃ©cution maximales pour chaque sous-question
TMAX_KNN = 40
TMAX_SVM = 200
TMAX_PERCEPTRON = 400
TMAX_EVAL = 80

def fetchPendigits():
    """
    Cette fonction tÃ©lÃ©charge le jeu de donnÃ©es pendigits et le
    retourne sous forme de deux tableaux numpy. Le premier Ã©lÃ©ment
    retournÃ© par cette fonction est un tableau de 10992x16 qui
    contient les samples; le second Ã©lÃ©ment est un vecteur de 10992
    qui contient les valeurs cible (target).
    """
    host = 'vision.gel.ulaval.ca'
    url = '/~cgagne/enseignement/apprentissage/A2018/travaux/ucipendigits.npy'
    connection = HTTPConnection(host, port=80, timeout=10)
    connection.request('GET', url)

    rep = connection.getresponse()
    if rep.status != 200:
        print("ERREUR : impossible de tÃ©lÃ©charger le jeu de donnÃ©es UCI Pendigits! Code d'erreur {}".format(rep.status))
        print("VÃ©rifiez que votre ordinateur est bien connectÃ© Ã  Internet.")
        return
    stream = BytesIO(rep.read())
    dataPendigits = numpy.load(stream)
    return dataPendigits[:, :-1].astype('float32'), dataPendigits[:, -1]

# Ne modifiez rien avant cette ligne!



if __name__ == "__main__":
    # Question 2B

    # TODO Q2B
    # Chargez le jeu de donnÃ©es Pendigits. Utilisez pour cela la fonction
    # fetchPendigits fournie. N'oubliez pas de normaliser
    # les donnÃ©es d'entrÃ©e entre 0 et 1 pour toutes les dimensions.
    # Notez finalement que fetch_openml retourne les donnÃ©es d'une maniÃ¨re
    # diffÃ©rente des fonctions load_*, assurez-vous que vous utilisez
    # correctement les donnÃ©es et qu'elles sont du bon type.
   
    
    # TODO Q2B
    # SÃ©parez le jeu de donnÃ©es Pendigits en deux sous-jeux: entraÃ®nement (5000) et
    # test (reste des donnÃ©es). Pour la suite du code, rappelez-vous que vous ne
    # pouvez PAS vous servir du jeu de test pour dÃ©terminer la configuration
    # d'hyper-paramÃ¨tres la plus performante. Ce jeu de test ne doit Ãªtre utilisÃ©
    # qu'Ã  la toute fin, pour rapporter les rÃ©sultats finaux en gÃ©nÃ©ralisation.


    # TODO Q2B
    # Pour chaque classifieur :
    # - k plus proches voisins,
    # - SVM Ã  noyau gaussien,
    # - Perceptron multicouche,
    # dÃ©terminez les valeurs optimales des hyper-paramÃ¨tres Ã  utiliser.
    # Suivez les instructions de l'Ã©noncÃ© quant au nombre d'hyper-paramÃ¨tres Ã 
    # optimiser et n'oubliez pas d'expliquer vos choix d'hyper-paramÃ¨tres
    # dans votre rapport.
    # Vous Ãªtes libres d'utiliser la mÃ©thodologie que vous souhaitez, en autant
    # que vous ne touchez pas au jeu de test.
    #
    # Note : optimisez les hyper-paramÃ¨tres des diffÃ©rentes mÃ©thodes dans
    # l'ordre dans lequel ils sont Ã©numÃ©rÃ©s plus haut, en insÃ©rant votre code
    # d'optimisation entre les commentaires le spÃ©cifiant
    

    
    _times.append(time.time())
    # TODO Q2B
    # Optimisez ici la paramÃ©trisation du kPP
   

    
    _times.append(time.time())
    checkTime(TMAX_KNN, "K plus proches voisins")
    # TODO Q2B
    # Optimisez ici la paramÃ©trisation du SVM Ã  noyau gaussien
   


    _times.append(time.time())
    checkTime(TMAX_SVM, "SVM")
    # TODO Q2B
    # Optimisez ici la paramÃ©trisation du perceptron multicouche
    # Note : il se peut que vous obteniez ici des "ConvergenceWarning"
    # Ne vous en souciez pas et laissez le paramÃ¨tre max_iter Ã  sa
    # valeur suggÃ©rÃ©e dans l'Ã©noncÃ© (100)
    



    _times.append(time.time())
    checkTime(TMAX_PERCEPTRON, "SVM")
    

    # TODO Q2B
    # Ã‰valuez les performances des meilleures paramÃ©trisations sur le jeu de test
    # et rapportez ces performances dans le rapport
    
    

    _times.append(time.time())
    checkTime(TMAX_EVAL, "Evaluation des modÃ¨les")
# N'Ã©crivez pas de code Ã  partir de cet endroit