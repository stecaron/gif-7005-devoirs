#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 21:02:06 2018

@author: stephanecaron
"""

# -*- coding: utf-8 -*-
###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 3, Question 3
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
import itertools

from scipy.optimize import fmin_l_bfgs_b
from scipy.spatial.distance import cdist

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from matplotlib import pyplot


# Fonctions utilitaires liÃ©es Ã  l'Ã©valuation
_times = []
def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps Ã  s'exÃ©cuter! ".format(question)+
            "Le temps maximum permis est de {0:.4f} secondes, mais votre code a requis {1:.4f} secondes! ".format(maxduration,duration)+
            "Assurez-vous que vous ne faites pas d'appels bloquants (par exemple Ã  show()) dans cette boucle!") 

# DÃ©finition des durÃ©es d'exÃ©cution maximales pour chaque sous-question
TMAX_FIT = 2.0
TMAX_EVAL = 3.0


# Ne modifiez rien avant cette ligne!


# Question 3B
# ImplÃ©mentation du discriminant Ã  noyau
class DiscriminantANoyau:

    def __init__(self, lambda_, sigma):
        # Cette fonction est dÃ©jÃ  codÃ©e pour vous, vous n'avez qu'Ã  utiliser
        # les variables membres qu'elle dÃ©finit dans les autres fonctions de
        # cette classe.
        # Lambda et sigma sont dÃ©finis dans l'Ã©noncÃ©.
        self.lambda_ = lambda_
        self.sigma = sigma
        
    def get_h(self, X, x_t, y, params):
        h = numpy.empty(x_t.shape[0])
        for row in range(x_t.shape[0]):
            h[row] = numpy.sum(params[1:] * y * numpy.exp(-(numpy.linalg.norm((X - x_t[row]), axis=1, ord=2)**2)/self.sigma**2)) + params[0]
        return h
        
    
    def fit(self, X, y):
        # ImplÃ©mentez la fonction d'entraÃ®nement du classifieur, selon
        # les Ã©quations que vous avez dÃ©veloppÃ©es dans votre rapport.
        
        # On remplace la classe negative par -1 (au lieu de 0)
        y_new = numpy.copy(y)
        numpy.place(y_new, y_new==0, -1)

        # TODO Q3B
        # Vous devez Ã©crire une fonction nommÃ©e evaluateFunc,
        # qui reÃ§oit un seul argument en paramÃ¨tre, qui correspond aux
        # valeurs des paramÃ¨tres pour lesquels on souhaite connaÃ®tre
        # l'erreur et le gradient d'erreur pour chaque paramÃ¨tre.
        # Cette fonction sera appelÃ©e Ã  rÃ©pÃ©tition par l'optimiseur
        # de scipy, qui l'utilisera pour minimiser l'erreur et obtenir
        # un jeu de paramÃ¨tres optimal.
       
        def evaluateFunc(hypers):
            
            # On trouve nos obs mal classées avec le w initial
            obs_mal_classees = numpy.where(self.get_h(X, X, y_new, hypers) * y_new < 1)[0]
            x_mal_classe = X[obs_mal_classees]
            y_mal_classe = y_new[obs_mal_classees]
            
            
            # On trouve l'erreur
            err = numpy.sum((1 - y_mal_classe * self.get_h(X, x_mal_classe, y, hypers))) + self.lambda_ * numpy.sum(hypers[1:])
            
            # On definit les gradients
            grad = numpy.zeros(hypers.shape[0])
            grad[0] = -numpy.sum(y_mal_classe)
            for row in range(grad.shape[0]-1):
#                grad[row+1] = -numpy.sum(y[row] * y_mal_classe * numpy.exp(-cdist(x_mal_classe, X[row].reshape(1,-1), 'euclidean')**2/self.sigma**2)) + self.lambda_
                grad[row+1] = -numpy.sum(y[row] * y_mal_classe * numpy.exp(-(numpy.linalg.norm((x_mal_classe - X[row]), axis=1, ord=2)**2)/self.sigma**2)) + self.lambda_
 
            return err, grad
        
        # TODO Q3B
        # Initialisez alÃ©atoirement les paramÃ¨tres alpha et omega0
        # (l'optimiseur requiert un "initial guess", et nous ne pouvons pas
        # simplement n'utiliser que des zÃ©ros pour diffÃ©rentes raisons).
        # Stochez ces valeurs initiales alÃ©atoires dans un array numpy nommÃ©
        # "params"
        # DÃ©terminez Ã©galement les bornes Ã  utiliser sur ces paramÃ¨tres
        # et stockez les dans une variable nommÃ©e "bounds".
        # Indice : les paramÃ¨tres peuvent-ils avoir une valeur maximale (au-
        # dessus de laquelle ils ne veulent plus rien dire)? Une valeur
        # minimale? RÃ©fÃ©rez-vous Ã  la documentation de fmin_l_bfgs_b
        # pour savoir comment indiquer l'absence de bornes.
        
        # On set les parametres initiaux
        params = numpy.random.rand(X.shape[0] + 1)
        
        # On definit les bounds
        bounds = [(None, None)] + [(0, numpy.inf)]*X.shape[0]
       

        # Ã€ ce stade, trois choses devraient Ãªtre dÃ©finies :
        # - Une fonction d'Ã©valuation nommÃ©e evaluateFunc, capable de retourner
        #   l'erreur et le gradient d'erreur pour chaque paramÃ¨tre pour une
        #   configuration de paramÃ¨tres alpha et omega_0 donnÃ©e.
        # - Un tableau numpy nommÃ© params de mÃªme taille que le nombre de
        #   paramÃ¨tres Ã  entraÃ®ner.
        # - Une liste nommÃ©e bounds contenant les bornes que l'optimiseur doit 
        #   respecter pour chaque paramÃ¨tre
        # On appelle maintenant l'optimiseur avec ces informations et on stocke
        # les valeurs optimisÃ©es dans params
        _times.append(time.time())
        params, minval, infos = fmin_l_bfgs_b(evaluateFunc, params, bounds=bounds)
        _times.append(time.time())
        checkTime(TMAX_FIT, "Entrainement")

        # On affiche quelques statistiques
        print("EntraÃ®nement terminÃ© aprÃ¨s {it} itÃ©rations et "
                "{calls} appels Ã  evaluateFunc".format(it=infos['nit'], calls=infos['funcalls']))
        print("\tErreur minimale : {:.5f}".format(minval))
        print("\tL'algorithme a convergÃ©" if infos['warnflag'] == 0 else "\tL'algorithme n'a PAS convergÃ©")
        print("\tGradients des paramÃ¨tres Ã  la convergence (ou Ã  l'Ã©puisement des ressources) :")
        print(infos['grad'])

        # TODO Q3B
        # Stockez les paramÃ¨tres optimisÃ©s de la faÃ§on suivante
        # - Le vecteur alpha dans self.alphas
        # - Le biais omega0 dans self.w0
        
        self.alphas = params[1:]
        self.w0 = params[0]



        # On retient Ã©galement le jeu d'entraÃ®nement, qui pourra
        # vous Ãªtre utile pour les autres fonctions Ã  implÃ©menter
        self.X, self.y = X, y_new
    
    def predict(self, X):
        # TODO Q3B
        # ImplÃ©mentez la fonction de prÃ©diction
        # Vous pouvez supposer que fit() a prÃ©alablement Ã©tÃ© exÃ©cutÃ©
        # et que les variables membres alphas, w0, X et y existent.
        # N'oubliez pas que ce classifieur doit retourner -1 ou 1
        
        hypers = numpy.append(self.w0, self.alphas)
        h_values = self.get_h(self.X, X, self.y, hypers)
        preds = numpy.copy(h_values)
        numpy.place(preds, preds<0, -1)
        numpy.place(preds, preds>0, 1)
        
        return preds
        
    
    def score(self, X, y):
        # TODO Q3B
        # ImplÃ©mentez la fonction retournant le score (accuracy)
        # du classifieur sur les donnÃ©es reÃ§ues en argument.
        # Vous pouvez supposer que fit() a prÃ©alablement Ã©tÃ© exÃ©cutÃ©
        # Indice : rÃ©utiliser votre implÃ©mentation de predict() rÃ©duit de
        # beaucoup la taille de cette fonction!
        
        y_new = numpy.copy(y)
        numpy.place(y_new, y_new==0, -1)
        
        preds_classes = self.predict(X)
        accuracy = sum(preds_classes == y_new)/y_new.shape[0]
        return(accuracy)

    



if __name__ == "__main__":
    # Question 3B

    # TODO Q3B
    # CrÃ©ez le jeu de donnÃ©es Ã  partir de la fonction make_moons, tel que
    # demandÃ© dans l'Ã©noncÃ©
    # N'oubliez pas de vous assurer que les valeurs possibles de y sont
    # bel et bien -1 et 1, et non 0 et 1!
    
    data = make_moons(n_samples=1000, noise=0.3)
    X = data[0]
    y = data[1]

    
    # TODO Q3B
    # SÃ©parez le jeu de donnÃ©es en deux parts Ã©gales, l'une pour l'entraÃ®nement
    # et l'autre pour le test
    
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42)
    
    _times.append(time.time())
    # TODO Q3B
    # Une fois les paramÃ¨tres lambda et sigma de votre classifieur optimisÃ©s,
    # crÃ©ez une instance de ce classifieur en utilisant ces paramÃ¨tres optimaux,
    # et calculez sa performance sur le jeu de test.
    
    sigma_possible = [0.1, 0.5, 1]
    lambda_possible = [0.5, 1, 2]
    
    for sigma in sigma_possible:
        for lambda_ in lambda_possible:
        
            clf = DiscriminantANoyau(lambda_, sigma)
            clf.fit(X_train, y_train)
            
            print("sigma : "+str(sigma)+", lambda: "+str(lambda_)+", Score train:"+str(clf.score(X_train, y_train))+", Score test:"+str(clf.score(X_test, y_test)))
        
        
    # Les meilleurs résultats sur le train sont:
    # lambda_ = et sigma = 
    
    # Tester les performances sur le test
    clf_optimal = DiscriminantANoyau(0.5, 0.5)
    clf_optimal.fit(X_train, y_train)
    clf_optimal.score(X_test, y_test)


    
    # TODO Q3B
    # CrÃ©ez ici une grille permettant d'afficher les rÃ©gions de
    # dÃ©cision pour chaque classifieur
    # Indice : numpy.meshgrid pourrait vous Ãªtre utile ici
    # Par la suite, affichez les rÃ©gions de dÃ©cision dans la mÃªme figure
    # que les donnÃ©es de test.
    # Note : utilisez un pas de 0.02 pour le meshgrid
   
    # Tracer le graph avec les zones de decisions
    xvalues = numpy.arange(min(X[:, 0]), max(X[:, 0]) + 0.2, 0.02)
    yvalues = numpy.arange(min(X[:, 1]), max(X[:, 1]) + 0.2, 0.02)
    xx, yy = numpy.meshgrid(xvalues, yvalues)
    
    fig = pyplot
    fig.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    z = clf_optimal.predict(numpy.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    fig.contourf(xx, yy, z, alpha=.25)
    
#    fig.savefig("/Users/stephanecaron/Documents/universitee/maitrise-statistique/automne-2018/GIF-7005/devoirs/devoir-3/images/q3-c.png", quality=95)
    


    # On affiche la figure
    # _times.append(time.time())
    checkTime(TMAX_FIT, "Evaluation")
    pyplot.show()
# N'Ã©crivez pas de code Ã  partir de cet endroit