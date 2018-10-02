###############################################################################
# Introduction Ã  l'apprentissage machine
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 1, Question 2
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

from matplotlib import pyplot

# Jeu de donnÃ©es utilisÃ©s
from sklearn.datasets import load_iris, make_circles

# Classifieurs utilisÃ©s
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid

# MÃ©thodes d'Ã©valuation
from sklearn.model_selection import train_test_split, RepeatedKFold

# Fonctions utilitaires liÃ©es Ã  l'Ã©valuation
_times = []


def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps Ã  s'exÃ©cuter! ".format(question) +
              "Le temps maximum permis est de {0:.4f} secondes, mais votre code a requis {1:.4f} secondes! ".format(
                  maxduration, duration) +
              "Assurez-vous que vous ne faites pas d'appels bloquants (par exemple Ã  show()) dans cette boucle!")


# DÃ©finition des durÃ©es d'exÃ©cution maximales pour chaque sous-question
TMAX_Q2A = 0.5
TMAX_Q2B = 1.5
TMAX_Q2Cii = 0.5
TMAX_Q2Ciii = 0.5
TMAX_Q2D = 1.0

# DÃ©finition des erreurs maximales attendues pour chaque sous-question
ERRMAX_Q2B = 0.22
ERRMAX_Q2Cii = 0.07
ERRMAX_Q2Ciii = 0.07

# Ne changez rien avant cette ligne!
# Seul le code suivant le "if __name__ == '__main__':" comporte des sections Ã  implÃ©menter

if __name__ == '__main__':
    # Question 2A

    # Chargez ici le dataset 'iris' dans une variable nommÃ©e data
    data = load_iris()

    # Cette ligne crÃ©e une liste contenant toutes les paires
    # possibles entre les 4 mesures
    # Par exemple : [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]

    # Utilisons cette liste de paires pour afficher les donnÃ©es,
    # deux mesures Ã  la fois
    # On crÃ©e une figure Ã  plusieurs sous-graphes

    fig, subfigs = pyplot.subplots(2, 3)
    _times.append(time.time())
    for (f1, f2), subfig in zip(pairs, subfigs.reshape(-1)):
        # Affichez les donnÃ©es en utilisant f1 et f2 comme mesures
        subfig.scatter(data.data[data.target == 0, f1], data.data[data.target == 0, f2], c="red", label=data.target_names[0])
        subfig.scatter(data.data[data.target == 1, f1], data.data[data.target == 1, f2], c="green", label=data.target_names[1])
        subfig.scatter(data.data[data.target == 2, f1], data.data[data.target == 2, f2], c="blue", label=data.target_names[2])
        subfig.legend()
        subfig.set_xlabel(data.feature_names[f1])
        subfig.set_ylabel(data.feature_names[f2])
        pass
    _times.append(time.time())
    checkTime(TMAX_Q2A, "2A")

    # On affiche la figure
    pyplot.show()

    # Question 2B

    # Reprenons les paires de mesures, mais entraÃ®nons cette fois
    # diffÃ©rents modÃ¨les demandÃ©s avec chaque paire
    for (f1, f2) in pairs:

        # TODO Q2B
        # CrÃ©ez ici un sous-dataset contenant seulement les
        # mesures dÃ©signÃ©es par f1 et f2
        data_subset = np.array(data.data[:,[f1, f2]])

        # TODO Q2B
        # Initialisez ici les diffÃ©rents classifieurs, dans
        # une liste nommÃ©e "classifieurs"
        classifieurs = [QuadraticDiscriminantAnalysis(), LinearDiscriminantAnalysis(), GaussianNB(), NearestCentroid()]

        # TODO Q2B
        # CrÃ©ez ici une grille permettant d'afficher les rÃ©gions de
        # dÃ©cision pour chaque classifieur
        # Indice : numpy.meshgrid pourrait vous Ãªtre utile ici
        # N'utilisez pas un pas trop petit!
        xvalues = np.arange(min(data_subset[:,0]), max(data_subset[:,0]) + 0.2, 0.1)
        yvalues = np.arange(min(data_subset[:,1]), max(data_subset[:,1]) + 0.2, 0.1)
        xx, yy = np.meshgrid(xvalues, yvalues)

        # On crÃ©e une figure Ã  plusieurs sous-graphes
        fig, subfigs = pyplot.subplots(2, 2, sharex='all', sharey='all')
        _times.append(time.time())
        for clf, subfig in zip(classifieurs, subfigs.reshape(-1)):
            # TODO Q2B
            # EntraÃ®nez le classifieur
            clf.fit(data_subset, data.target)

            # TODO Q2B
            # Obtenez et affichez son erreur (1 - accuracy)
            # Stockez la valeur de cette erreur dans la variable err
            err = 1-sum(clf.predict(data_subset) == data.target)/data.target.size

            # TODO Q2B
            # Utilisez la grille que vous avez crÃ©Ã©e plus haut
            # pour afficher les rÃ©gions de dÃ©cision, de mÃªme
            # que les points colorÃ©s selon leur vraie classe
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            subfig.contourf(xx, yy, Z, alpha = 0.5)
            subfig.scatter(data_subset[:,0], data_subset[:,1], c = data.target)


            # Identification des axes et des mÃ©thodes
            subfig.set_xlabel(data.feature_names[f1])
            subfig.set_ylabel(data.feature_names[f2])
            subfig.set_title(clf.__class__.__name__)

            if err > ERRMAX_Q2B:
                print("[ATTENTION] Votre code pour la question 2B ne produit pas les performances attendues! " +
                      "Le taux d'erreur maximal attendu est de {0:.3f}, mais l'erreur rapportÃ©e dans votre code est de {1:.3f}!".format(
                          ERRMAX_Q2B, err))

        _times.append(time.time())
        checkTime(TMAX_Q2B, "2B")

        # On affiche les graphiques
        pyplot.show()

    # Question 2C
    # Note : Q2C (i) peut Ãªtre rÃ©pondue en utilisant le code prÃ©cÃ©dent
    classifieur_qda = QuadraticDiscriminantAnalysis()
    classifieur_qda.fit(data.data, data.target)
    erreur = 1 - sum(classifieur_qda.predict(data.data) == data.target) / data.target.size


    _times.append(time.time())
    # TODO Q2Cii
    # Ã?crivez ici le code permettant de partitionner les donnÃ©es en jeux
    # d'entraÃ®nement / de validation et de tester la performance du classifieur
    # mentionnÃ© dans l'Ã©noncÃ©
    # Vous devez rÃ©pÃ©ter cette mesure 10 fois avec des partitions diffÃ©rentes
    # Stockez l'erreur moyenne sur ces 10 itÃ©rations dans une variable nommÃ©e avgError

    erreur = np.zeros(10)
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.5, random_state=42
        )
        classifieur_qda = QuadraticDiscriminantAnalysis()
        classifieur_qda.fit(X_train, y_train)
        erreur[i] = 1 - sum(classifieur_qda.predict(X_test) == y_test) / y_test.size

    avgError = erreur.mean()

    _times.append(time.time())
    checkTime(TMAX_Q2Cii, "2Cii")

    if avgError > ERRMAX_Q2Cii:
        print("[ATTENTION] Votre code pour la question 2C ii) ne produit pas les performances attendues! " +
              "Le taux d'erreur maximal attendu est de {0:.3f}, mais l'erreur rapportÃ©e dans votre code est de {1:.3f}!".format(
                  ERRMAX_Q2Cii, avgError))

    _times.append(time.time())
    # TODO Q2Ciii
    # Ã?crivez ici le code permettant de dÃ©terminer la performance du classifieur
    # avec un K-fold avec k=3.
    # Vous devez rÃ©pÃ©ter le K-folding 10 fois
    # Stockez l'erreur moyenne sur ces 10 itÃ©rations dans une variable nommÃ©e avgError
    erreur = np.zeros(10)
    rkf = RepeatedKFold(n_splits=3, n_repeats=10, random_state=42)
    for train_index, test_index in rkf.split(data.data):
        X_train, X_test = data.data[train_index], data.data[test_index]
        y_train, y_test = data.target[train_index], data.target[test_index]

        classifieur_qda = QuadraticDiscriminantAnalysis()
        classifieur_qda.fit(X_train, y_train)
        erreur[i] = 1 - sum(classifieur_qda.predict(X_test) == y_test) / y_test.size

    avgError = erreur.mean()

    _times.append(time.time())
    checkTime(TMAX_Q2Ciii, "2Ciii")

    if avgError > ERRMAX_Q2Ciii:
        print("[ATTENTION] Votre code pour la question 2C iii) ne produit pas les performances attendues! " +
              "Le taux d'erreur maximal attendu est de {0:.3f}, mais l'erreur rapportÃ©e dans votre code est de {1:.3f}!".format(
                  ERRMAX_Q2Ciii, avgError))

    # Question 2D
    # TODO Q2D
    # Initialisez ici les diffÃ©rents classifieurs, dans
    # une liste nommÃ©e "classifieurs"
    classifieurs = [QuadraticDiscriminantAnalysis(), LinearDiscriminantAnalysis(), GaussianNB(), NearestCentroid()]

    # CrÃ©ation du jeu de donnÃ©es
    X, y = make_circles(factor=0.3)

    # TODO Q2D
    # CrÃ©ez ici une grille permettant d'afficher les rÃ©gions de
    # dÃ©cision pour chaque classifieur
    # Indice : numpy.meshgrid pourrait vous Ãªtre utile ici
    # N'utilisez pas un pas trop petit!
    xvalues = np.arange(min(X[:, 0]), max(X[:, 0]) + 0.1, 0.01)
    yvalues = np.arange(min(X[:, 1]), max(X[:, 1]) + 0.1, 0.01)
    xx, yy = np.meshgrid(xvalues, yvalues)

    # On crÃ©e une figure Ã  plusieurs sous-graphes
    fig, subfigs = pyplot.subplots(2, 2, sharex='all', sharey='all')
    _times.append(time.time())

    # Spliter le dataset et toujours reutiliser le meme split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    for clf, subfig in zip(classifieurs, subfigs.reshape(-1)):

        # TODO Q2D
        # Obtenez et affichez son erreur (1 - accuracy)
        # Stockez la valeur de cette erreur dans la variable err
        clf.fit(X_train, y_train)
        err = 1 - sum(clf.predict(X_train) == y_train) / y_train.size
        print(err)


        # TODO Q2D
        # Utilisez la grille que vous avez crÃ©Ã©e plus haut
        # pour afficher les rÃ©gions de dÃ©cision, de mÃªme
        # que les points colorÃ©s selon leur vraie classe
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        subfig.contourf(xx, yy, Z, alpha=0.5)
        subfig.scatter(X[:, 0], X[:, 1], c=y)
        subfig.set_title(clf.__class__.__name__)

    _times.append(time.time())
    checkTime(TMAX_Q2D, "2D")

    pyplot.show()

# N'Ã©crivez pas de code Ã  partir de cet endroit