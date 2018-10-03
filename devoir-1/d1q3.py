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

# Question 3C
class ClassifieurAvecRejet():

    def __init__(self, _lambda=1):
        # _lambda est le coÃ»t de rejet
        self._lambda = _lambda

    def fit(self, X, y):
        # TODO Q3C
        # ImplÃ©mentez ici une fonction permettant d'entraÃ®ner votre modÃ¨le
        # Ã  partir des donnÃ©es fournies en argument
        self._mu = numpy.zeros((numpy.unique(y).__len__(), X.shape[1],))
        self._s =  numpy.zeros((numpy.unique(y).__len__(), X.shape[1], X.shape[1]))
        self._priori = numpy.zeros(numpy.unique(y).__len__())
        self._k = numpy.unique(y)
        for item in self._k:
            self._mu[item,:] = X[y == item,:].mean(axis=0)
            self._priori[item] = sum(y == item)/y.__len__()
            self._mu_stack = numpy.tile(self._mu[item,:], (sum(y == item),1))
            self._s[item,:,:] = numpy.diag(numpy.repeat(numpy.power((X[y == item,:] - self._mu_stack), 2).sum()/(X[y == item,:].shape[0] * X.shape[1]),X.shape[1]))
        pass

    def predict_proba(self, X):
        # TODO Q3C
        # ImplÃ©mentez une fonction retournant la probabilitÃ© d'appartenance Ã  chaque
        # classe, pour les donnÃ©es passÃ©es en argument. Cette fonction peut supposer
        # que fit() a prÃ©alablement Ã©tÃ© appelÃ©.
        # Indice : calculez les diffÃ©rents termes de l'Ã©quation de Bayes sÃ©parÃ©ment
        proba = numpy.zeros((X.shape[0], self._k.__len__()))
        self._pr_x_sachant_c = numpy.zeros((self._k.__len__()))
        for obs in range(X.shape[0]):
            for item in self._k:
                self._pr_x_sachant_c[item] = (1/numpy.power(2*numpy.pi, 0.5 * X.shape[1]) * numpy.power(numpy.linalg.det(self._s[item,:,:]), 0.5))*numpy.exp((-1/2) * numpy.dot(numpy.dot((X[obs,:] - self._mu[item,:]).transpose(), numpy.linalg.inv(self._s[item,:,:])), (X[obs,:] - self._mu[item,:])))
            proba[obs, :] = self._pr_x_sachant_c * self._priori / numpy.dot(self._pr_x_sachant_c, self._priori.transpose())
        return proba
        pass

    def predict(self, X):
        # TODO Q3C
        # ImplÃ©mentez une fonction retournant les prÃ©dictions pour les donnÃ©es
        # passÃ©es en argument. Cette fonction peut supposer que fit() a prÃ©alablement
        # Ã©tÃ© appelÃ©.
        # Indice : vous pouvez utiliser predict_proba() pour Ã©viter une redondance du code
        self._predictions = numpy.zeros((X.shape[0]))
        self._predictions_pourc = self.predict_proba(X)
        self._predicted_max = numpy.amax(self._predictions_pourc , axis=1)
        self._predictions = numpy.argmax(self._predictions_pourc , axis=1)
        numpy.place(self._predictions, self._predicted_max<1-self._lambda, max(self._k) + 1)
        return self._predictions
        pass

    def score(self, X, y):
        # TODO Q3C
        # ImplÃ©mentez une fonction retournant le score (tenant compte des donnÃ©es
        # rejetÃ©es si lambda < 1.0) pour les donnÃ©es passÃ©es en argument.
        # Cette fonction peut supposer que fit() a prÃ©alablement Ã©tÃ© exÃ©cutÃ©.
        err = numpy.zeros(X.shape[0])
        predictions = self.predict(X)
        for obs in range(X.shape[0]):
            if predictions[obs] == max(self._k) + 1:
                err[obs] = self._lambda
            else:
                err[obs] = 1 - (predictions[obs] == y[obs])
        return err.sum(axis=0)/y.__len__()
        pass



# Ne modifiez rien avant cette ligne!


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
    #cmpt=0
    for (f1, f2) in pairs:
        # TODO Q3D
        # CrÃ©ez ici un sous-dataset contenant seulement les
        # mesures dÃ©signÃ©es par f1 et f2
        data_subset = numpy.array(data.data[:, [f1, f2]])

        # TODO Q3D
        # CrÃ©ez ici une grille permettant d'afficher les rÃ©gions de
        # dÃ©cision pour chaque classifieur
        # Indice : numpy.meshgrid pourrait vous Ãªtre utile ici
        # N'utilisez pas un pas trop petit!
        xvalues = numpy.arange(min(data_subset[:, 0]), max(data_subset[:, 0]) + 0.2, 0.15)
        yvalues = numpy.arange(min(data_subset[:, 1]), max(data_subset[:, 1]) + 0.2, 0.15)
        xx, yy = numpy.meshgrid(xvalues, yvalues)

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
            clf.fit(data_subset, data.target)

            # TODO Q3D
            # Obtenez et affichez son score
            # Stockez la valeur de cette erreur dans la variable err

            err = clf.score(data_subset, data.target)
            print(err)

            # TODO Q3D
            # Utilisez la grille que vous avez crÃ©Ã©e plus haut
            # pour afficher les rÃ©gions de dÃ©cision, INCLUANT LA
            # ZONE DE REJET, de mÃªme que les points colorÃ©s selon
            # leur vraie classe
            subfig.scatter(data_subset[:, 0], data_subset[:, 1], c=data.target)
            z = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
            z = z.reshape(xx.shape)
            subfig.contourf(xx, yy, z, alpha=.25)
            # subfig.contourf.set_under('w')
            # subfig.set_clim(-1)

            # On ajoute un titre et des Ã©tiquettes d'axes
            subfig.set_title("lambda=" + str(clf._lambda))
            subfig.set_xlabel(data.feature_names[f1])
            subfig.set_ylabel(data.feature_names[f2])
        _times.append(time.time())
        checkTime(TMAX_Q3D, "3D")

        # On affiche les graphiques
        pyplot.show()
        #fig.savefig(
            #"/Users/stephanecaron/Documents/universitee/maitrise-statistique/automne-2018/GIF-7005/devoirs/devoir-1/images/3d-"+str(cmpt)+".png")
        #cmpt=cmpt+1
# N'Ã©crivez pas de code Ã  partir de cet endroit