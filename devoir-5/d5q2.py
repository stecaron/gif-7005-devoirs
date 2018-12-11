# -*- coding: utf-8 -*-
###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 5, Question 2
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

# Nous ne voulons pas avoir ce type d'avertissement, qui
# n'est pas utile dans le cadre de ce devoir
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from matplotlib import pyplot

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, RFE, chi2, mutual_info_classif
from sklearn.model_selection import train_test_split

from sklearn.mixture import GaussianMixture

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


TMAX_Q2Aall = 0.5
TMAX_Q2Achi = 0.5
TMAX_Q2Amut = 60
TMAX_Q2B = 150

# Ne modifiez rien avant cette ligne!


if __name__ == "__main__":
    # Question 2
    # Chargement des donnÃ©es et des noms des caractÃ©ristiques
    # IMPORTANT : ce code assume que vous avez prÃ©alablement tÃ©lÃ©chargÃ©
    # l'archive csdmc-spam-binary.zip Ã  l'adresse
    # http://vision.gel.ulaval.ca/~cgagne/enseignement/apprentissage/A2018/donnees/csdmc-spam-binary.zip
    # et que vous l'avez extrait dans le rÃ©pertoire courant, de telle faÃ§on
    # qu'un dossier nommÃ© "csdmc-spam-binary" soit prÃ©sent.

    X = numpy.loadtxt("csdmc-spam-binary/data", delimiter=",")
    y = numpy.loadtxt("csdmc-spam-binary/target", delimiter=",")
    with open("csdmc-spam-binary/features", "r") as f:
        features = [line[:-1] for line in f]

    # Division du jeu en entraÃ®nement / test
    # Ne modifiez pas la random seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    _times.append(time.time())
    # TODO Q2A
    # EntraÃ®nez un classifieur SVM linÃ©aire sur le jeu de donnÃ©es *complet*
    # et rapportez sa performance en test
    svm_complet = LinearSVC()
    svm_complet.fit(X_train, y_train)
    svm_complet.score(X_test, y_test)

    _times.append(time.time())
    checkTime(TMAX_Q2Aall, "2A (avec toutes les variables)")

    # TODO Q2A
    # EntraÃ®nez un classifieur SVM linÃ©aire sur le jeu de donnÃ©es
    # en rÃ©duisant le nombre de caractÃ©ristiques (features) Ã  10 en
    # utilisant le chiÂ² comme mÃ©trique et rapportez sa performance en test
    chi2_fit = SelectKBest(chi2, k=10).fit(X_train, y_train)
    X_train_chi2 = chi2_fit.transform(X_train)
    X_test_chi2 = chi2_fit.transform(X_test)
    svm_chi2 = LinearSVC()
    svm_chi2.fit(X_train_chi2, y_train)
    svm_chi2.score(X_test_chi2, y_test)

    list(itertools.compress(features, chi2_fit.get_support()))

    _times.append(time.time())
    checkTime(TMAX_Q2Achi, "2A (avec sous-ensemble de variables par chi2)")

    # TODO Q2A
    # EntraÃ®nez un classifieur SVM linÃ©aire sur le jeu de donnÃ©es
    # en rÃ©duisant le nombre de caractÃ©ristiques (features) Ã  10 en utilisant
    # l'information mutuelle comme mÃ©trique et rapportez sa performance en test
    mutual_info_fit = SelectKBest(mutual_info_classif, k=10).fit(X_train, y_train)
    X_train_mutual = mutual_info_fit.transform(X_train)
    X_test_mutual = mutual_info_fit.transform(X_test)
    svm_mutual = LinearSVC()
    svm_mutual.fit(X_train_mutual, y_train)
    svm_mutual.score(X_test_mutual, y_test)

    list(itertools.compress(features, mutual_info_fit.get_support()))

    _times.append(time.time())
    checkTime(TMAX_Q2Amut, "2A (avec sous-ensemble de variables par mutual info)")

    # TODO Q2B
    # EntraÃ®nez un classifieur SVM linÃ©aire sur le jeu de donnÃ©es
    # en rÃ©duisant le nombre de caractÃ©ristiques (features) Ã  10 par
    # sÃ©lection sÃ©quentielle arriÃ¨re et rapportez sa performance en test
    svm_stepwise = LinearSVC()
    ref = RFE(svm_stepwise, n_features_to_select=10)
    ref.fit(X_train, y_train)
    ref.score(X_test, y_test)

    list(itertools.compress(features, ref.get_support()))

    _times.append(time.time())
    checkTime(TMAX_Q2B, "2B")

# N'Ã©crivez pas de code Ã  partir de cet endroit