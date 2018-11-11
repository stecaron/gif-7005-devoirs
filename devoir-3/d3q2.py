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
   
    data_pendigits = fetchPendigits()
    X = minmax_scale(data_pendigits[0])
    
    # TODO Q2B
    # SÃ©parez le jeu de donnÃ©es Pendigits en deux sous-jeux: entraÃ®nement (5000) et
    # test (reste des donnÃ©es). Pour la suite du code, rappelez-vous que vous ne
    # pouvez PAS vous servir du jeu de test pour dÃ©terminer la configuration
    # d'hyper-paramÃ¨tres la plus performante. Ce jeu de test ne doit Ãªtre utilisÃ©
    # qu'Ã  la toute fin, pour rapporter les rÃ©sultats finaux en gÃ©nÃ©ralisation.

    X_train, X_test, y_train, y_test = train_test_split(
    X, data_pendigits[1], test_size=5992/(5000+5992), random_state=42)
        

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
    
    # Definion des grilles d'hyperparametres
    n_neighbors_possible = [1, 3, 5, 10, 20, 50, 100]
    weights_possible = ["uniform", "distance"]
    
    for n_neighbors, weights in itertools.product(n_neighbors_possible, weights_possible):
        
        # On initialise le classfieur kPP avec les hyperparametres de la grillle
        classifieur_kpp = KNeighborsClassifier(n_neighbors=n_neighbors, weights = weights)
        
        # On fit avec la CV à 5 plis
        accuracy = []

        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        for train_index, test_index in kf.split(X_train):
            X_train_cv, X_validation = X_train[train_index], X_train[test_index]
            y_train_cv, y_validation = y_train[train_index], y_train[test_index]
            
            # On entraine le modele
            classifieur_kpp.fit(X_train_cv, y_train_cv)
            temp = sum(classifieur_kpp.predict(X_validation) == y_validation) / y_validation.size
            accuracy.append(temp)
            
            
        # On calcule l'erreur moyenne pour ce classifieur
        avgAccuracy = numpy.mean(accuracy)
        print("Number of neighbors: "+str(n_neighbors)+", Weight: "+str(weights)+", Mean accuray :"+str(avgAccuracy))
        
   

    
    _times.append(time.time())
    checkTime(TMAX_KNN, "K plus proches voisins")
    # TODO Q2B
    # Optimisez ici la paramÃ©trisation du SVM Ã  noyau gaussien

    # Definion des grilles d'hyperparametres
    C_possible = [0.01, 0.1, 1, 10, 100, 1000]
    gamma_possible = [0.1, 0.2, 0.5]
    
    for C, gamma in itertools.product(C_possible, gamma_possible):
        
        # On initialise le classfieur kPP avec les hyperparametres de la grille
        classifieur_svm = SVC(C=C, kernel='rbf', gamma=gamma)
        
        # On fit avec la CV à 5 plis
        accuracy = []

        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        for train_index, test_index in kf.split(X_train):
            X_train_cv, X_validation = X_train[train_index], X_train[test_index]
            y_train_cv, y_validation = y_train[train_index], y_train[test_index]
            
            # On entraine le modele
            classifieur_svm.fit(X_train_cv, y_train_cv)
            temp = sum(classifieur_svm.predict(X_validation) == y_validation) / y_validation.size
            accuracy.append(temp)
            
            
        # On calcule l'erreur moyenne pour ce classifieur
        avgAccuracy = numpy.mean(accuracy)
        print("C : "+str(C)+", Gamma: "+str(gamma)+", Mean accuray :"+str(avgAccuracy))
           


        _times.append(time.time())
        checkTime(TMAX_SVM, "SVM")
        # TODO Q2B
        # Optimisez ici la paramÃ©trisation du perceptron multicouche
        # Note : il se peut que vous obteniez ici des "ConvergenceWarning"
        # Ne vous en souciez pas et laissez le paramÃ¨tre max_iter Ã  sa
        # valeur suggÃ©rÃ©e dans l'Ã©noncÃ© (100)
        
        # Definion des grilles d'hyperparametres
        hidden_layer_sizes_possible = [ 10, 100, 500, 1000]
        activation_possible = ['relu', 'identity', 'tanh', 'logistic']
        
        for hidden_layer_sizes, activation in itertools.product(hidden_layer_sizes_possible, activation_possible):
            
            # On initialise le classfieur kPP avec les hyperparametres de la grille
            classifieur_mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                            activation=activation)
            
            # On fit avec la CV à 5 plis
            accuracy = []
    
            kf = KFold(n_splits=5, random_state=42, shuffle=True)
            for train_index, test_index in kf.split(X_train):
                X_train_cv, X_validation = X_train[train_index], X_train[test_index]
                y_train_cv, y_validation = y_train[train_index], y_train[test_index]
                
                # On entraine le modele
                classifieur_mlp.fit(X_train_cv, y_train_cv)
                temp = sum(classifieur_mlp.predict(X_validation) == y_validation) / y_validation.size
                accuracy.append(temp)
                
                
            # On calcule l'erreur moyenne pour ce classifieur
            avgAccuracy = numpy.mean(accuracy)
            print("Hidden layer sizes : "+str(hidden_layer_sizes)+", Activation: "+str(activation)+", Mean accuray :"+str(avgAccuracy))
               
    
    
        _times.append(time.time())
        checkTime(TMAX_PERCEPTRON, "SVM")
    

    # TODO Q2B
    # Ã‰valuez les performances des meilleures paramÃ©trisations sur le jeu de test
    # et rapportez ces performances dans le rapport
    
    model_final_kpp = KNeighborsClassifier(n_neighbors=3, weights='distance')
    model_final_svm = SVC(C=10, gamma=0.5)
    model_final_mlp = MLPClassifier(hidden_layer_sizes=100, activation='relu')
    
    # On reentraine les modeles sur le jeu de donnees entrainement complet
    model_final_kpp.fit(X_train, y_train)
    model_final_svm.fit(X_train, y_train)
    model_final_mlp.fit(X_train, y_train)
    
    print('Accuray for kNN on test: '+str(model_final_kpp.score(X_test, y_test)))
    print('Accuray for SVM on test: '+str(model_final_svm.score(X_test, y_test)))
    print('Accuray for MLP on test: '+str(model_final_mlp.score(X_test, y_test)))
    
    
    

    _times.append(time.time())
    checkTime(TMAX_EVAL, "Evaluation des modÃ¨les")
# N'Ã©crivez pas de code Ã  partir de cet endroit