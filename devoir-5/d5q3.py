# -*- coding: utf-8 -*-
###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 5, Question 3
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

import warnings

# Nous ne voulons pas avoir ce type d'avertissement, qui
# n'est pas utile dans le cadre de ce devoir
warnings.filterwarnings("ignore", category=FutureWarning)

from matplotlib import pyplot

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_gaussian_quantiles, make_moons

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


TMAX_Q3Ai = 60
TMAX_Q3Aii = 80
TMAX_Q3C = 130

# Ne modifiez rien avant cette ligne!


if __name__ == "__main__":
    # Question 3
    # CrÃ©ation du jeu de donnÃ©es
    # Ne modifiez pas ces lignes
    X1, y1 = make_gaussian_quantiles(cov=2.2,
                                     n_samples=600, n_features=2,
                                     n_classes=2, random_state=42)
    X2, y2 = make_moons(n_samples=300, noise=0.25, random_state=42)
    X3, y3 = make_moons(n_samples=300, noise=0.3, random_state=42)
    X2[:, 0] -= 3.5
    X2[:, 1] += 3.5
    X3[:, 0] += 4.0
    X3[:, 1] += 2.0
    X = numpy.concatenate((X1, X2, X3))
    y = numpy.concatenate((y1, y2 + 2, y3 + 1))

    # Division du jeu en entraÃ®nement / test
    # Ne modifiez pas la random seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=42)

    _times.append(time.time())
    # TODO Q3A
    # EntraÃ®nez un classifieur par ensemble de type AdaBoost sur le jeu de donnÃ©es (X_train, y_train)
    # dÃ©fini plus haut, en utilisant des souches de dÃ©cision comme classifieur de base.
    # Rapportez les rÃ©sultats et figures tel que demandÃ© dans l'Ã©noncÃ©, sur
    # les jeux d'entraÃ®nement et de test.

    accuracy_test = numpy.zeros(49)
    accuracy_train = numpy.zeros(49)
    for n_tree in range(49):
        adaboost_souche = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=n_tree+1)
        adaboost_souche.fit(X_train, y_train)
        accuracy_train[n_tree] = adaboost_souche.score(X_train, y_train)
        accuracy_test[n_tree] = adaboost_souche.score(X_test, y_test)

    pyplot.figure()
    pyplot.plot(range(1, 50, 1), accuracy_train, label="Précision sur le train")
    pyplot.plot(range(1, 50, 1), accuracy_test, label="Précision sur le test")
    # pyplot.title("Précision sur le test en fonction du nombre de weak learners.")
    pyplot.legend(loc="lower right")
    # pyplot.savefig("/Users/stephanecaron/Documents/universitee/maitrise-statistique/automne-2018/GIF-7005/devoirs/devoir-5/rapport/images/3a-souche.png")
    pyplot.show()

    # Tracer les zones de décisions pour les souches
    n_tree_to_test = [3, 5, 30]
    fig, subfigs = pyplot.subplots(2, 2, sharex='all', sharey='all')
    for n_tree, subfig in zip(n_tree_to_test, subfigs.reshape(-1)):
        adaboost_souche = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=n_tree + 1)
        adaboost_souche.fit(X_train, y_train)
        x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
        y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
        xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, 0.05),
                                numpy.arange(y_min, y_max, 0.05))

        Z = adaboost_souche.predict(numpy.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        subfig.contourf(xx, yy, Z, alpha=0.5)
        subfig.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
        subfig.set_title("Nombre de souches = " + str(n_tree))

    pyplot.delaxes(subfigs[1][1])

    # pyplot.savefig("/Users/stephanecaron/Documents/universitee/maitrise-statistique/automne-2018/GIF-7005/devoirs/devoir-5/rapport/images/3a-souche-zones.png")
    pyplot.show()

    _times.append(time.time())
    checkTime(TMAX_Q3Ai, "3A avec souches")

    # TODO Q3A
    # EntraÃ®nez un classifieur par ensemble de type AdaBoost sur le jeu de donnÃ©es (X_train, y_train)
    # dÃ©fini plus haut, en utilisant des arbres de dÃ©cision de profonduer 3 comme
    # classifieur de base. Rapportez les rÃ©sultats et figures tel que demandÃ© dans l'Ã©noncÃ©, sur
    # les jeux d'entraÃ®nement et de test.

    accuracy_test = numpy.zeros(49)
    accuracy_train = numpy.zeros(49)
    for n_tree in range(49):
        adaboost_arbres = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=n_tree + 1)
        adaboost_arbres.fit(X_train, y_train)
        accuracy_train[n_tree] = adaboost_arbres.score(X_train, y_train)
        accuracy_test[n_tree] = adaboost_arbres.score(X_test, y_test)

    pyplot.figure()
    pyplot.plot(range(1, 50, 1), accuracy_train, label="Précision sur le train")
    pyplot.plot(range(1, 50, 1), accuracy_test, label="Précision sur le test")
    # pyplot.title("Précision sur le test en fonction du nombre de weak learners.")à
    pyplot.legend(loc="lower right")
    # pyplot.savefig("/Users/stephanecaron/Documents/universitee/maitrise-statistique/automne-2018/GIF-7005/devoirs/devoir-5/rapport/images/3a-arbres.png")
    pyplot.show()

    # Tracer les zones de décisions pour les arbres
    fig, subfigs = pyplot.subplots(2, 2, sharex='all', sharey='all')
    n_tree_to_test = [20, 30, 45]
    for n_tree, subfig in zip(n_tree_to_test, subfigs.reshape(-1)):
        adaboost_arbres = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=n_tree + 1)
        adaboost_arbres.fit(X_train, y_train)
        x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
        y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
        xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, 0.05),
                                numpy.arange(y_min, y_max, 0.05))

        Z = adaboost_arbres.predict(numpy.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        subfig.contourf(xx, yy, Z, alpha=0.5)
        subfig.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
        subfig.set_title("Nombre d'arbres = " + str(n_tree))

    pyplot.delaxes(subfigs[1][1])

    # pyplot.savefig("/Users/stephanecaron/Documents/universitee/maitrise-statistique/automne-2018/GIF-7005/devoirs/devoir-5/rapport/images/3a-arbres-zones.png")
    pyplot.show()

    _times.append(time.time())
    checkTime(TMAX_Q3Aii, "3A avec arbres de profondeur 3")

    # TODO Q3C
    # EntraÃ®nez un classifieur par ensemble de type Random Forest sur le jeu de donnÃ©es (X_train, y_train)
    # dÃ©fini plus haut. Rapportez les rÃ©sultats et figures tel que demandÃ© dans l'Ã©noncÃ©, sur
    # les jeux d'entraÃ®nement et de test.

    train_accuracy = numpy.zeros(12)
    test_accuracy = numpy.zeros(12)
    for n_tree in range(12):
        rf = RandomForestClassifier(n_estimators=50, max_depth=n_tree+1)
        rf.fit(X_train, y_train)
        train_accuracy[n_tree] = rf.score(X_train, y_train)
        test_accuracy[n_tree] = rf.score(X_test, y_test)

    pyplot.figure()
    pyplot.plot(range(1, 13, 1), train_accuracy, label="Précision sur le train")
    pyplot.plot(range(1, 13, 1), test_accuracy, label="Précision sur le test")
    # pyplot.title("Précision sur le train/test en fonction de la profondeur max.")
    pyplot.legend(loc="lower right")
    # pyplot.savefig("/Users/stephanecaron/Documents/universitee/maitrise-statistique/automne-2018/GIF-7005/devoirs/devoir-5/rapport/images/3c-rf.png")
    pyplot.show()

    fig, subfigs = pyplot.subplots(2, 2, sharex='all', sharey='all')
    n_tree_to_test = [1, 4, 8, 12]
    for n_tree, subfig in zip(n_tree_to_test, subfigs.reshape(-1)):
        rf = RandomForestClassifier(n_estimators=50, max_depth=n_tree)
        rf.fit(X_train, y_train)
        x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
        y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
        xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, 0.05),
                                numpy.arange(y_min, y_max, 0.05))

        Z = rf.predict(numpy.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        subfig.contourf(xx, yy, Z, alpha=0.5)
        subfig.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
        subfig.set_title("Profondeur = "+str(n_tree))

    # pyplot.savefig("/Users/stephanecaron/Documents/universitee/maitrise-statistique/automne-2018/GIF-7005/devoirs/devoir-5/rapport/images/3c-rf-zones.png")
    pyplot.show()

    _times.append(time.time())
    checkTime(TMAX_Q3C, "3C")

# N'Ã©crivez pas de code Ã  partir de cet endroit