#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 19:06:28 2018

@author: stephanecaron
"""

# -*- coding: utf-8 -*-
###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 2, Question 2
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
import numpy

from matplotlib import pyplot, patches

from sklearn.datasets import make_classification, load_breast_cancer, load_iris
from sklearn.preprocessing import minmax_scale, normalize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

# Fonctions utilitaires liÃ©es Ã  l'Ã©valuation
_times = []
def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps Ã  s'exÃ©cuter! ".format(question)+
            "Le temps maximum permis est de {0:.4f} secondes, mais votre code a requis {1:.4f} secondes! ".format(maxduration,duration)+
            "Assurez-vous que vous ne faites pas d'appels bloquants (par exemple Ã  show()) dans cette boucle!") 

# DÃ©finition des durÃ©es d'exÃ©cution maximales pour chaque sous-question
TMAX_Q2B = 5.0
TMAX_Q2Bdisp = 10.0
TMAX_Q2C = 20
TMAX_Q2Dbc = 60
TMAX_Q2Diris = 40
TMAX_Q2Ebc = 30
TMAX_Q2Eiris = 15


# Ne modifiez rien avant cette ligne!



# Question 2B
# ImplÃ©mentation du discriminant linÃ©aire
class DiscriminantLineaire:
    def __init__(self, eta=1e-2, epsilon=1e-2, max_iter=1000):
        # Cette fonction est dÃ©jÃ  codÃ©e pour vous, vous n'avez qu'Ã  utiliser
        # les variables membres qu'elle dÃ©finit dans les autres fonctions de
        # cette classe.
        self.eta = eta
        # Epsilon et max_iter servent Ã  stocker les critÃ¨res d'arrÃªt
        # max_iter est un simple critÃ¨re considÃ©rant le nombre de mises Ã  jour
        # effectuÃ©es sur les poids (autrement dit, on cesse l'apprentissage
        # aprÃ¨s max_iter itÃ©ration de la boucle d'entraÃ®nement), alors que
        # epsilon indique la diffÃ©rence minimale qu'il doit y avoir entre
        # les erreurs de deux itÃ©rations successives pour que l'on ne
        # considÃ¨re pas l'algorithme comme ayant convergÃ©. Par exemple,
        # si epsilon=1e-2, alors tant que la diffÃ©rence entre l'erreur
        # obtenue Ã  la prÃ©cÃ©dente itÃ©ration et l'itÃ©ration courante est
        # plus grande que 0.01, on continue, sinon on arrÃªte.
        self.epsilon = epsilon
        self.max_iter = max_iter
    
    def fit(self, X, y):
        # ImplÃ©mentez la fonction d'entraÃ®nement du classifieur, selon
        # les Ã©quations que vous avez dÃ©veloppÃ©es dans votre rapport.

        # On initialise les poids alÃ©atoirement
        w = numpy.random.rand(X.shape[1]+1)

        # TODO Q2B
        # Vous devez ici implÃ©menter l'entraÃ®nement.
        # Celui-ci devrait Ãªtre contenu dans la boucle suivante, qui se rÃ©pÃ¨te
        # self.max_iter fois
        # Vous Ãªtes libres d'utiliser les noms de variable de votre choix, sauf
        # pour les poids qui doivent Ãªtre contenus dans la variable w dÃ©finie plus haut
        for i in range(self.max_iter):
            # ...
     
        # Ã€ ce stade, la variable w devrait contenir les poids entraÃ®nÃ©s
        # On les copie dans une variable membre pour les conserver
        self.w = w
    
    def predict(self, X):
        # TODO Q2B
        # ImplÃ©mentez la fonction de prÃ©diction
        # Vous pouvez supposer que fit() a prÃ©alablement Ã©tÃ© exÃ©cutÃ©
    
    def score(self, X, y):
        # TODO Q2B
        # ImplÃ©mentez la fonction retournant le score (accuracy)
        # du classifieur sur les donnÃ©es reÃ§ues en argument.
        # Vous pouvez supposer que fit() a prÃ©alablement Ã©tÃ© exÃ©cutÃ©
        # Indice : rÃ©utiliser votre implÃ©mentation de predict() rÃ©duit de
        # beaucoup la taille de cette fonction!



# Question 2B
# ImplÃ©mentation du classifieur un contre tous utilisant le discriminant linÃ©aire
# dÃ©fini plus haut
class ClassifieurUnContreTous:
    def __init__(self, n_classes, **kwargs):
        # Cette fonction est dÃ©jÃ  codÃ©e pour vous, vous n'avez qu'Ã  utiliser
        # les variables membres qu'elle dÃ©finit dans les autres fonctions de
        # cette classe.
        self.n_classes = n_classes
        self.estimators = [DiscriminantLineaire(**kwargs) for c in range(n_classes)]
    
    def fit(self, X, y):
        # TODO Q2C
        # ImplÃ©mentez ici une approche un contre tous, oÃ¹ chaque classifieur 
        # (contenu dans self.estimators) est entraÃ®nÃ© Ã  distinguer une seule classe 
        # versus toutes les autres
    
    def predict(self, X):
        # TODO Q2C
        # ImplÃ©mentez ici la prÃ©diction utilisant l'approche un contre tous
        # Vous pouvez supposer que fit() a prÃ©alablement Ã©tÃ© exÃ©cutÃ©
    
    def score(self, X, y):
        # TODO Q2C
        # ImplÃ©mentez ici le calcul du score utilisant l'approche un contre tous
        # Ce score correspond Ã  la prÃ©cision (accuracy) moyenne.
        # Vous pouvez supposer que fit() a prÃ©alablement Ã©tÃ© exÃ©cutÃ©


if __name__ == '__main__':
    # Question 2C

    _times.append(time.time())
    # ProblÃ¨me Ã  2 classes
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=1)

    # TODO Q2C
    # Testez la performance du discriminant linÃ©aire pour le problÃ¨me
    # Ã  deux classes, et tracez les rÃ©gions de dÃ©cision
    




    _times.append(time.time())
    checkTime(TMAX_Q2Bdisp, "2B")
    
    pyplot.show()


    _times.append(time.time())
    # 3 classes
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=1, n_classes=3)

    # TODO Q2C
    # Testez la performance du discriminant linÃ©aire pour le problÃ¨me
    # Ã  trois classes, et tracez les rÃ©gions de dÃ©cision
   




    _times.append(time.time())
    checkTime(TMAX_Q2Bdisp, "2C")
    
    pyplot.show()



    # Question 2D

    _times.append(time.time())

    # TODO Q2D
    # Chargez les donnÃ©es "Breast cancer Wisconsin" et normalisez les de
    # maniÃ¨re Ã  ce que leur minimum et maximum soient de 0 et 1
    
    

    # TODO Q2D
    # Comparez les diverses approches demandÃ©es dans l'Ã©noncÃ© sur Breast Cancer
    # Initialisez votre discriminant linÃ©aire avec les paramÃ¨tres suivants :
    # DiscriminantLineaire(eta=1e-4, epsilon=1e-6, max_iter=10000)
    # Pour les autres approches, conservez les valeurs par dÃ©faut
    # N'oubliez pas que l'Ã©valuation doit Ãªtre faite par une validation
    # croisÃ©e Ã  K=3 plis!
   



    _times.append(time.time())
    checkTime(TMAX_Q2Dbc, "2Dbc")
    
    
    
    _times.append(time.time())
    # TODO Q2D
    # Chargez les donnÃ©es "Iris" et normalisez les de
    # maniÃ¨re Ã  ce que leur minimum et maximum soient de 0 et 1
   
    


    # TODO Q2D
    # Comparez les diverses approches demandÃ©es dans l'Ã©noncÃ© sur Iris
    # Pour utilisez votre discriminant linÃ©aire, utilisez l'approche Un Contre Tous
    # implÃ©mentÃ© au 2C.
    # Initialisez vos discriminants linÃ©aires avec les paramÃ¨tres suivants :
    # DiscriminantLineaire(eta=1e-4, epsilon=1e-6, max_iter=10000)
    # Pour les autres approches, conservez les valeurs par dÃ©faut
    # N'oubliez pas que l'Ã©valuation doit Ãªtre faite par une validation
    # croisÃ©e Ã  K=3 plis!
    

    

    _times.append(time.time())
    checkTime(TMAX_Q2Diris, "2Diris")



    
    
    _times.append(time.time())
    # TODO Q2E
    # Testez un classifeur K plus proches voisins sur Breast Cancer
    # L'Ã©valuation doit Ãªtre faite en utilisant une approche leave-one-out
    # Testez avec k = {1, 3, 5, 7, 11, 13, 15, 25, 35, 45} et avec les valeurs
    # "uniform" et "distance" comme valeur de l'argument "weights".
    # N'oubliez pas de normaliser le jeu de donnÃ©es en utilisant minmax_scale!
    #
    # Stockez les performances obtenues (prÃ©cision moyenne pour chaque valeur de k)
    # dans deux listes, scoresUniformWeights pour weights=uniform et 
    # scoresDistanceWeights pour weights=distance
    # Le premier Ã©lÃ©ment de chacune de ces listes devrait contenir la prÃ©cision
    # pour k=1, le second la prÃ©cision pour k=3, et ainsi de suite.
    scoresUniformWeights = []
    scoresDistanceWeights = []


    _times.append(time.time())
    checkTime(TMAX_Q2Ebc, "2Ebc")

    # TODO Q2E
    # Produisez un graphique contenant deux courbes, l'une pour weights=uniform
    # et l'autre pour weights=distance. L'axe x de la figure doit Ãªtre le nombre
    # de voisins et l'axe y la performance en leave-one-out

    pyplot.show()


    _times.append(time.time())
    # TODO Q2E
    # Testez un classifeur K plus proches voisins sur Iris
    # L'Ã©valuation doit Ãªtre faite en utilisant une approche leave-one-out
    # Testez avec k = {1, 3, 5, 7, 11, 13, 15, 25, 35, 45} et avec les valeurs
    # "uniform" et "distance" comme valeur de l'argument "weights".
    # N'oubliez pas de normaliser le jeu de donnÃ©es en utilisant minmax_scale!
    #
    # Stockez les performances obtenues (prÃ©cision moyenne pour chaque valeur de k)
    # dans deux listes, scoresUniformWeights pour weights=uniform et 
    # scoresDistanceWeights pour weights=distance
    # Le premier Ã©lÃ©ment de chacune de ces listes devrait contenir la prÃ©cision
    # pour k=1, le second la prÃ©cision pour k=3, et ainsi de suite.
    scoresUniformWeights = []
    scoresDistanceWeights = []


    _times.append(time.time())
    checkTime(TMAX_Q2Eiris, "2Eiris")

    
    # TODO Q2E
    # Produisez un graphique contenant deux courbes, l'une pour weights=uniform
    # et l'autre pour weights=distance. L'axe x de la figure doit Ãªtre le nombre
    # de voisins et l'axe y la performance en leave-one-out

    pyplot.show()
    


# N'Ã©crivez pas de code Ã  partir de cet endroit