# -*- coding: utf-8 -*-
###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 5, Question 4
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

from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.datasets import load_digits

from scipy.spatial.distance import cdist


def plot_clustering(X_red, labels, title, savepath):
    # TirÃ© de https://scikit-learn.org/stable/auto_examples/cluster/plot_digits_linkage.html
    # Auteur : Gael Varoquaux
    # DistribuÃ© sous license BSD
    #
    # X_red doit Ãªtre un array numpy contenant les caractÃ©ristiques (features)
    #   des donnÃ©es d'entrÃ©e, rÃ©duit Ã  2 dimensions
    #
    # labels doit Ãªtre un array numpy contenant les Ã©tiquettes de chacun des
    #   Ã©lÃ©ments de X_red, dans le mÃªme ordre
    #
    # title est le titre que vous souhaitez donner Ã  la figure
    #
    # savepath est le nom du fichier oÃ¹ la figure doit Ãªtre sauvegardÃ©e
    #
    x_min, x_max = numpy.min(X_red, axis=0), numpy.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    pyplot.figure(figsize=(9, 6), dpi=160)
    for i in range(X_red.shape[0]):
        pyplot.text(X_red[i, 0], X_red[i, 1], str(y[i]),
                    color=pyplot.cm.nipy_spectral(labels[i] / 10.),
                    fontdict={'weight': 'bold', 'size': 9})

    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.title(title, size=17)
    pyplot.axis('off')
    pyplot.tight_layout(rect=[0, 0.03, 1, 0.95])
    pyplot.savefig(savepath)
    pyplot.close()


# Ne modifiez rien avant cette ligne!


if __name__ == "__main__":
    # On charge le jeu de donnÃ©es "digits"
    X, y = load_digits(return_X_y=True)

    # TODO Q4
    # Ã‰crivez le code permettant de projeter le jeu de donnÃ©es en 2 dimensions
    # avec les classes scikit-learn suivantes : PCA, MDS et TSNE

    # Reduction avec PCA
    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)

    # Reduction avec MDS
    mds = MDS(n_components=2, n_init=1)
    mds.fit(X)
    X_mds = mds.transform(X)

    # Reduction avec TSNE
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)

    # TODO Q4
    # Calculez le ratio entre la distance moyenne intra-classe et la distance moyenne
    # inter-classe, pour chacune des classes, pour chacune des mÃ©thodes, y compris
    # le jeu de donnÃ©es original. Utilisez une distance euclidienne.
    # La fonction cdist importÃ©e plus haut pourrait vous Ãªtre utile

    def ratio_distance_intra_inter(X, y):
        ratio = numpy.zeros(10)

        for classe in range(10):
            X_intra = X[numpy.where(classe == y)]
            X_inter = X[numpy.where(classe != y)]
            intra_dist = numpy.mean(cdist(X_intra, X_intra))
            inter_dist = numpy.mean(cdist(X_intra, X_inter))
            ratio[classe] = intra_dist/inter_dist

        return ratio

    ratio_pca = ratio_distance_intra_inter(X_pca, y)
    ratio_mds = ratio_distance_intra_inter(X_mds, y)
    ratio_tsne = ratio_distance_intra_inter(X_tsne, y)
    ratio_original = ratio_distance_intra_inter(X, y)

    # Trouver des images prochent dans l'espace a 2D, mais de classes différentes
    image_to_look = [(3, 9), (1, 8)]
    digits_images = load_digits().images
    for combinaison in image_to_look:
        X_same = X_tsne[numpy.where(combinaison[0] == y)]
        X_diff = X_tsne[numpy.where(combinaison[1] == y)]
        dist_inter = cdist(X_same, X_diff)
        dist_min = numpy.where(dist_inter == numpy.min(dist_inter))
        dist_min = tuple([i.item() for i in dist_min])
        fig = pyplot.figure(figsize=(10,10))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
        digits = load_digits()
        j = 0
        for index, valeur in zip(dist_min, combinaison):
            image_look = digits_images[numpy.where(valeur == y)]
            y_look = y[numpy.where(valeur == y)]
            ax = fig.add_subplot(1, 2, j + 1, xticks=[], yticks=[])
            ax.imshow(image_look[index], cmap=pyplot.cm.binary, interpolation='nearest')
            ax.text(0, 7, str(y_look[index]))
            j += 1
        pyplot.show()

    # TODO Q4
    # Utilisez la fonction plot_clustering pour afficher les rÃ©sultats des
    # diffÃ©rentes mÃ©thodes de rÃ©duction de dimensionnalitÃ©
    # Produisez Ã©galement un graphique montrant les diffÃ©rents ratios de
    # distance intra/inter classe pour toutes les mÃ©thodes

    plot_clustering(X_pca, y, "Réduction dimensionnalité avec PCA (2 dimensions)", "/Users/stephanecaron/Documents/universitee/maitrise-statistique/automne-2018/GIF-7005/devoirs/devoir-5/rapport/images/4pca.png")
    plot_clustering(X_mds, y, "Réduction dimensionnalité avec MDS (2 dimensions)", "/Users/stephanecaron/Documents/universitee/maitrise-statistique/automne-2018/GIF-7005/devoirs/devoir-5/rapport/images/4mds.png")
    plot_clustering(X_tsne, y, "Réduction dimensionnalité avec TSNE (2 dimensions)", "/Users/stephanecaron/Documents/universitee/maitrise-statistique/automne-2018/GIF-7005/devoirs/devoir-5/rapport/images/4tsne.png")

# N'Ã©crivez pas de code Ã  partir de cet endroit