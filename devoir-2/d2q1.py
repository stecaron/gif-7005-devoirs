#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 19:05:39 2018

@author: stephanecaron
"""

# -*- coding: utf-8 -*-
###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 2, Question 1
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

from scipy.stats import norm

from sklearn.neighbors import KernelDensity

# Fonctions utilitaires liÃ©es Ã  l'Ã©valuation
_times = []
def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps Ã  s'exÃ©cuter! ".format(question)+
            "Le temps maximum permis est de {0:.4f} secondes, mais votre code a requis {1:.4f} secondes! ".format(maxduration,duration)+
            "Assurez-vous que vous ne faites pas d'appels bloquants (par exemple Ã  show()) dans cette boucle!") 

# DÃ©finition des durÃ©es d'exÃ©cution maximales pour chaque sous-question
TMAX_Q1A = 1.0
TMAX_Q1B = 2.5


# Ne changez rien avant cette ligne!


# DÃ©finition de la PDF de la densitÃ©-mÃ©lange
def pdf(X):
    return 0.4 * norm(0, 1).pdf(X[:, 0]) + 0.6 * norm(5, 1).pdf(X[:, 0])


# Question 1A

# TODO Q1A
# ComplÃ©tez la fonction sample(n), qui gÃ©nÃ¨re n
# donnÃ©es suivant la distribution mentionnÃ©e dans l'Ã©noncÃ©
def sample(n):
    X = np.concatenate((np.random.normal(0, 1, int(0.4 * n)),
                        np.random.normal(5, 1, int(0.6 * n))))[:, np.newaxis]
    
    return X


if __name__ == '__main__':
    # Question 1A

    _times.append(time.time())
    # TODO Q1A
    # Ã‰chantillonnez 50 et 10 000 donnÃ©es en utilisant la fonction
    # sample(n) que vous avez dÃ©finie plus haut et tracez l'histogramme
    # de cette distribution Ã©chantillonÃ©e, en utilisant 25 bins,
    # dans le domaine [-5, 10].
    # Sur les mÃªmes graphiques, tracez Ã©galement la fonction de densitÃ© rÃ©elle.
    X1 = sample(50)
    X2 = sample(10000)
    
    X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
    bins = np.linspace(-5, 10, 25)
    
    fig, ax = pyplot.subplots(2, 1, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.4, wspace=0.05)
    
    ax[0].hist(x=X1, bins=bins, range=(-5,10), facecolor = 'royalblue', normed=True)
    ax[0].set_title("Histograme de la densité mélange avec n = 50")
    ax[0].plot(X_plot, pdf(X_plot))
    
    ax[1].hist(x=X2, bins=bins, range=(-5,10), facecolor = 'royalblue', normed=True)
    ax[1].set_title("Histograme de la densité mélange avec n = 10 000")
    ax[1].plot(X_plot, pdf(X_plot))

    # Affichage du graphique
    _times.append(time.time())
    checkTime(TMAX_Q1A, "1A")
    pyplot.show()


    # Question 1B
    _times.append(time.time())
    
    # TODO Q1B
    # Ã‰chantillonnez 50 et 10 000 donnÃ©es, mais utilisez cette fois une
    # estimation avec noyau boxcar pour prÃ©senter les donnÃ©es. Pour chaque
    # nombre de donnÃ©es (50 et 10 000), vous devez prÃ©senter les distributions
    # estimÃ©es avec des tailles de noyau (bandwidth) de {0.3, 1, 2, 5}, dans
    # la mÃªme figure, mais tracÃ©es avec des couleurs diffÃ©rentes.
    
#    def is_within_the_hypercube(u):
#        vec = np.array(u)
#        np.place(vec, abs(vec)<1, 1)
#        np.place(vec, abs(vec)>1, 0)
#        return vec
#        
#    def densite_estime(X, data, h):
#        N = data.shape[0]
#        return((1/2*N*h) * sum(is_within_the_hypercube((X-data)/h)))
#    
#    def hist_noyaux_boxcar(vec_X, data, bandwidth):
#        Y_densite = []
#        for i in range(vec_X.shape[0]):
#            Y_densite.append(densite_estime(vec_X[i,:], data, bandwidth))
#        return Y_densite/sum(Y_densite)
    
    X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
    kde_1 = KernelDensity(kernel='tophat', bandwidth=0.3).fit(X1)
    kde_2 = KernelDensity(kernel='tophat', bandwidth=1).fit(X1)
    kde_3 = KernelDensity(kernel='tophat', bandwidth=2).fit(X1)
    kde_4 = KernelDensity(kernel='tophat', bandwidth=5).fit(X1)
    
    kde_1_2 = KernelDensity(kernel='tophat', bandwidth=0.3).fit(X2)
    kde_2_2 = KernelDensity(kernel='tophat', bandwidth=1).fit(X2)
    kde_3_2 = KernelDensity(kernel='tophat', bandwidth=2).fit(X2)
    kde_4_2 = KernelDensity(kernel='tophat', bandwidth=5).fit(X2)
    
    fig, ax = pyplot.subplots(2, 1, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.4, wspace=0.05)
    
    ax[0].plot(X_plot, np.exp(kde_1.score_samples(X_plot)), label="bandwidth=0.3")
    ax[0].plot(X_plot, np.exp(kde_2.score_samples(X_plot)), label="bandwidth=1")
    ax[0].plot(X_plot, np.exp(kde_3.score_samples(X_plot)), label="bandwidth=2")
    ax[0].plot(X_plot, np.exp(kde_4.score_samples(X_plot)), label="bandwidth=5")
    ax[0].legend(loc="upper left", fontsize="x-small")
    ax[0].set_xlim((-5,10))
    ax[0].set_title("Histograme de la densité mélange avec n = 50")
    
    ax[1].plot(X_plot, np.exp(kde_1_2.score_samples(X_plot)), label="bandwidth=0.3")
    ax[1].plot(X_plot, np.exp(kde_2_2.score_samples(X_plot)), label="bandwidth=1")
    ax[1].plot(X_plot, np.exp(kde_3_2.score_samples(X_plot)), label="bandwidth=2")
    ax[1].plot(X_plot, np.exp(kde_4_2.score_samples(X_plot)), label="bandwidth=5")
    ax[1].legend(loc="upper left", fontsize="x-small")
    ax[1].set_xlim((-5,10))
    ax[1].set_title("Histograme de la densité mélange avec n = 10 000")
    

    # Affichage du graphique
    _times.append(time.time())
    checkTime(TMAX_Q1B, "1B")
    pyplot.show()
    
    fig.savefig("/Users/stephanecaron/Documents/universitee/maitrise-statistique/automne-2018/GIF-7005/devoirs/devoir-2/images/q1-b.png", quality=95)



# N'Ã©crivez pas de code Ã  partir de cet endroit