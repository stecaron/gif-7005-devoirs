# -*- coding: utf-8 -*-
###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 5, Question 1
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

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, v_measure_score
from sklearn.datasets import load_breast_cancer
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


TMAX_Q1A = 15
TMAX_Q1B = 20
TMAX_Q1C = 20

# Ne modifiez rien avant cette ligne!


if __name__ == "__main__":
    # Question 1
    # Chargement des donnÃ©es de Breast Cancer Wisconsin
    # Pas de division entraÃ®nement / test pour cette question
    X, y = load_breast_cancer(return_X_y=True)

    _times.append(time.time())
    # TODO Q1A
    # Ã‰crivez ici une fonction nommÃ©e `evalKmeans(X, y, k)`, qui prend en paramÃ¨tre
    # un jeu de donnÃ©es (`X` et `y`) et le nombre de clusters Ã  utiliser `k`, et qui
    # retourne un tuple de 3 Ã©lÃ©ments, Ã  savoir les scores basÃ©s sur :
    # - L'indice de Rand ajustÃ©
    # - L'information mutuelle
    # - La mesure V
    # Voyez l'Ã©noncÃ© pour plus de dÃ©tails sur ces scores.

    def evalKmeans(X, y, k):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        clusters = kmeans.labels_

        predictions = numpy.zeros(clusters.__len__())
        for cluster in range(k):
            ind = clusters == cluster
            label = y[ind]
            if (sum(label) > label.__len__()/2):
                predictions[ind] = 1
            else:
                predictions[ind] = 0


        return adjusted_rand_score(y, predictions), \
               adjusted_mutual_info_score(y, predictions), \
               v_measure_score(y, predictions)

    pyplot.figure()
    # TODO Q1A
    # Ã‰valuez ici la performance de K-means en utilisant la fonction `evalKmeans`
    # dÃ©finie plus haut, en faisant varier le nombre de clusters entre 2 et 50
    # par incrÃ©ment de 2. Tracez les rÃ©sultats obtenus sous forme de courbe
    adj_rand_score = []
    adj_mutual_info = []
    v_measure_info = []
    for k in range(2, 52, 2):
        scores = evalKmeans(X, y, k)
        adj_rand_score.append(scores[0])
        adj_mutual_info.append(scores[1])
        v_measure_info.append(scores[2])

    pyplot.plot(range(2, 52, 2), adj_rand_score, label="Indice de Rand ajusté")
    pyplot.plot(range(2, 52, 2), adj_mutual_info, label="Score basé sur l'info mutuelle")
    pyplot.plot(range(2, 52, 2), v_measure_info, label="Mesure V")
    pyplot.legend(loc='lower right')

    pyplot.savefig("/Users/stephanecaron/Documents/universitee/maitrise-statistique/automne-2018/GIF-7005/devoirs/devoir-5/rapport/images/1a.png")

    _times.append(time.time())
    checkTime(TMAX_Q1A, "1A")

    # On affiche la courbe obtenue
    pyplot.show()

    _times.append(time.time())
    # TODO Q1B
    # Ã‰crivez ici une fonction nommÃ©e `evalEM(X, y, k, init)`, qui prend en paramÃ¨tre
    # un jeu de donnÃ©es (`X` et `y`), le nombre de clusters Ã  utiliser `k`
    # et l'initialisation demandÃ©e ('random' ou 'kmeans') et qui
    # retourne un tuple de 3 Ã©lÃ©ments, Ã  savoir les scores basÃ©s sur :
    # - L'indice de Rand ajustÃ©
    # - L'information mutuelle
    # - La mesure V
    # Voyez l'Ã©noncÃ© pour plus de dÃ©tails sur ces scores.
    # N'oubliez pas que vous devez d'abord implÃ©menter les Ã©quations fournies Ã
    # la question 1B pour dÃ©terminer les Ã©tiquettes de classe, avant de passer
    # les rÃ©sultats aux diffÃ©rentes mÃ©triques!


    def evalEM(X, y, k, init):

        # Initialisation de la densite melange
        densite_melange = GaussianMixture(n_components=k, init_params=init, random_state=42)

        densite_melange.fit(X)
        predictions_proba = densite_melange.predict_proba(X)

        C0_GJ = numpy.sum((y == 0).reshape(-1, 1) * predictions_proba, axis=0) / \
                numpy.sum((y == 0).reshape(-1, 1) * predictions_proba)
        C1_GJ = numpy.sum((y == 1).reshape(-1, 1) * predictions_proba, axis=0) / \
                numpy.sum((y == 1).reshape(-1, 1) * predictions_proba)

        C0_XT = numpy.sum(C0_GJ * predictions_proba, axis=1)/\
                (numpy.sum(C0_GJ * predictions_proba, axis=1) + numpy.sum(C1_GJ * predictions_proba, axis=1))

        preds_labels = numpy.zeros(y.shape[0])

        for obs in range(preds_labels.shape[0]):
            if C0_XT[obs] > 0.5:
                preds_labels[obs] = 0
            else:
                preds_labels[obs] = 1

        return adjusted_rand_score(y, preds_labels), adjusted_mutual_info_score(y, preds_labels), \
        v_measure_score(y, preds_labels)


    pyplot.figure()
    # TODO Q1B
    # Ã‰valuez ici la performance de EM en utilisant la fonction `evalEM`
    # dÃ©finie plus haut, en faisant varier le nombre de clusters entre 2 et 50
    # par incrÃ©ment de 2 et en utilisant une initialisation alÃ©atoire.
    # Tracez les rÃ©sultats obtenus sous forme de courbe
    # Notez que ce calcul est assez long et devrait requÃ©rir au moins 120 secondes;
    # la limite de temps qui vous est accordÃ©e est beaucoup plus laxiste.
    adj_rand_score = []
    adj_mutual_info = []
    v_measure_info = []
    for k in range(2, 52, 2):
        scores = evalEM(X, y, k, init='random')
        adj_rand_score.append(scores[0])
        adj_mutual_info.append(scores[1])
        v_measure_info.append(scores[2])

    pyplot.plot(range(2, 52, 2), adj_rand_score, label="Indice de Rand ajusté")
    pyplot.plot(range(2, 52, 2), adj_mutual_info, label="Score basé sur l'info mutuelle")
    pyplot.plot(range(2, 52, 2), v_measure_info, label="Mesure V")
    pyplot.legend(loc='upper right')

    pyplot.savefig("/Users/stephanecaron/Documents/universitee/maitrise-statistique/automne-2018/GIF-7005/devoirs/devoir-5/rapport/images/1b.png")

    _times.append(time.time())
    checkTime(TMAX_Q1B, "1B")

    # On affiche la courbe obtenue
    pyplot.show()

    _times.append(time.time())
    pyplot.figure()
    # TODO Q1C
    # Ã‰valuez ici la performance de EM en utilisant la fonction `evalEM`
    # dÃ©finie plus haut, en faisant varier le nombre de clusters entre 2 et 50
    # par incrÃ©ment de 2 et en utilisant une initialisation par K-means.
    # Tracez les rÃ©sultats obtenus sous forme de courbe
    # Notez que ce calcul est assez long et devrait requÃ©rir au moins 120 secondes;
    # la limite de temps qui vous est accordÃ©e est beaucoup plus laxiste.
    adj_rand_score = []
    adj_mutual_info = []
    v_measure_info = []
    for k in range(2, 52, 2):
        scores = evalEM(X, y, k, init='kmeans')
        adj_rand_score.append(scores[0])
        adj_mutual_info.append(scores[1])
        v_measure_info.append(scores[2])

    pyplot.plot(range(2, 52, 2), adj_rand_score, label="Indice de Rand ajusté")
    pyplot.plot(range(2, 52, 2), adj_mutual_info, label="Score basé sur l'info mutuelle")
    pyplot.plot(range(2, 52, 2), v_measure_info, label="Mesure V")
    pyplot.legend(loc='lower right')

    pyplot.savefig("/Users/stephanecaron/Documents/universitee/maitrise-statistique/automne-2018/GIF-7005/devoirs/devoir-5/rapport/images/1c.png")

    _times.append(time.time())
    checkTime(TMAX_Q1C, "1C")

    # On affiche la courbe obtenue
    pyplot.show()

# N'Ã©crivez pas de code Ã  partir de cet endroit