###############################################################################
# Introduction Ã  l'apprentissage machine
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 1, Question 3
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

from matplotlib import pyplot
from sklearn.datasets import make_moons, load_iris

# Fonctions utilitaires liÃ©es Ã  l'Ã©valuation
_times = []


def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps Ã  s'exÃ©cuter! ".format(question) +
              "Le temps maximum permis est de {0:.4f} secondes, mais votre code a requis {1:.4f} secondes! ".format(
                  maxduration, duration) +
              "Assurez-vous que vous ne faites pas d'appels bloquants (par exemple Ã  show()) dans cette boucle!")


TMAX_Q3D = 1.0


# Ne modifiez rien avant cette ligne!


# Question 3C
class ClassifieurAvecRejet:

    def __init__(self, _lambda=1):
        # _lambda est le coÃ»t de rejet
        self._lambda = _lambda

    def fit(self, X, y):
        # TODO Q3C
        # ImplÃ©mentez ici une fonction permettant d'entraÃ®ner votre modÃ¨le
        # Ã  partir des donnÃ©es fournies en argument
        pass

    def predict_proba(self, X):
        # TODO Q3C
        # ImplÃ©mentez une fonction retournant la probabilitÃ© d'appartenance Ã  chaque
        # classe, pour les donnÃ©es passÃ©es en argument. Cette fonction peut supposer
        # que fit() a prÃ©alablement Ã©tÃ© appelÃ©.
        # Indice : calculez les diffÃ©rents termes de l'Ã©quation de Bayes sÃ©parÃ©ment
        pass

    def predict(self, X):
        # TODO Q3C
        # ImplÃ©mentez une fonction retournant les prÃ©dictions pour les donnÃ©es
        # passÃ©es en argument. Cette fonction peut supposer que fit() a prÃ©alablement
        # Ã©tÃ© appelÃ©.
        # Indice : vous pouvez utiliser predict_proba() pour Ã©viter une redondance du code
        pass

    def score(self, X, y):
        # TODO Q3C
        # ImplÃ©mentez une fonction retournant le score (tenant compte des donnÃ©es
        # rejetÃ©es si lambda < 1.0) pour les donnÃ©es passÃ©es en argument.
        # Cette fonction peut supposer que fit() a prÃ©alablement Ã©tÃ© exÃ©cutÃ©.
        pass


# Question 3D
if __name__ == "__main__":

    # TODO Q3D
    # Chargez ici le dataset 'iris' dans une variable nommÃ©e data
    data = load_iris()

    # Cette ligne crÃ©e une liste contenant toutes les paires
    # possibles entre les 4 mesures
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]

    # Utilisons cette liste de paires pour tester le classifieur
    # avec diffÃ©rents lambda
    for (f1, f2) in pairs:
        # TODO Q3D
        # CrÃ©ez ici un sous-dataset contenant seulement les
        # mesures dÃ©signÃ©es par f1 et f2

        # TODO Q3D
        # CrÃ©ez ici une grille permettant d'afficher les rÃ©gions de
        # dÃ©cision pour chaque classifieur
        # Indice : numpy.meshgrid pourrait vous Ãªtre utile ici
        # N'utilisez pas un pas trop petit!

        # On initialise les classifieurs avec diffÃ©rents paramÃ¨tres lambda
        classifieurs = [ClassifieurAvecRejet(0.1),
                        ClassifieurAvecRejet(0.3),
                        ClassifieurAvecRejet(0.5),
                        ClassifieurAvecRejet(1)]

        # On crÃ©e une figure Ã  plusieurs sous-graphes pour pouvoir montrer,
        # pour chaque configuration, les rÃ©gions de dÃ©cisions, incluant
        # la zone de rejet
        fig, subfigs = pyplot.subplots(2, 2, sharex='all', sharey='all')
        _times.append(time.time())
        for clf, subfig in zip(classifieurs, subfigs.reshape(-1)):
            # TODO Q3D
            # EntraÃ®nez le classifieur

            # TODO Q3D
            # Obtenez et affichez son score
            # Stockez la valeur de cette erreur dans la variable err

            # TODO Q3D
            # Utilisez la grille que vous avez crÃ©Ã©e plus haut
            # pour afficher les rÃ©gions de dÃ©cision, INCLUANT LA
            # ZONE DE REJET, de mÃªme que les points colorÃ©s selon
            # leur vraie classe

            # On ajoute un titre et des Ã©tiquettes d'axes
            subfig.set_title("lambda=" + str(clf._lambda))
            subfig.set_xlabel(data.feature_names[f1])
            subfig.set_ylabel(data.feature_names[f2])
        _times.append(time.time())
        checkTime(TMAX_Q3D, "3D")

        # On affiche les graphiques
        pyplot.show()

# N'Ã©crivez pas de code Ã  partir de cet endroit