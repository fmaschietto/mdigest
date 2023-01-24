#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: fmaschietto, bcallen95

from mdigest.core.imports import *
from mdigest.core import auxiliary


def plot_eigenvalues_distribution(**kwargs):
    """
    Plot the eigenvalues of a given correlation matrix.

    Parameters
    ----------
    kwargs: dict, possible options are:
            - matrix,      np.ndarray correlation matrix
            - eigenvalues, np.ndarray of eigenvalues
            - dim,         int specifying number of eigenvalues to display
            - color,       sns.palette or array of colors
            - ax,          mpl.ax object
            - matrix_type,   str, title for the legend


    if 'matrix' is provided the latter will be diagonalized and the first n eigenvalues shown as a barplot;
    alternatively, use 'eigenvalues'.
    """

    e_vals = 0
    if kwargs.__contains__('matrix'):
        M = kwargs['matrix']
        e_vals, _ = auxiliary.sorted_eig(M)

    elif kwargs.__contains__('eigenvalues'):
        e_vals = kwargs['eigenvalues']
    else:
        print('@> provide eigenvalues or matrix to diagonalize')

    if kwargs.__contains__('dim'):
        dim = kwargs['dim']
    else:
        dim = 10
    if kwargs.__contains__('color'):
        color = kwargs['color']
    else:
        color = sns.color_palette('bright', dim)[0]
    if kwargs.__contains__('label'):
        label = kwargs['label']
    else:
        label = sns.color_palette('')

    if kwargs.__contains__('ax'):
        ax = kwargs['ax']
        ax.bar(np.arange(len(e_vals[0:dim])), e_vals[0:dim], ec='k', alpha=.8, color=color, label=label)
        ax.set_xticks(np.arange(0, dim, int(np.round(dim / 3, 0))), np.arange(0, dim, int(np.round(dim / 3, 0))) + 1)
        ax.set_xlabel('# eigenvalue')
        ax.set_ylabel('magnitude')


    else:
        plt.figure(figsize=(10,5))
        plt.bar(np.arange(len(e_vals[0:dim])), e_vals[0:dim], ec='k', alpha=.8, label=label)
        plt.xticks(np.arange(0, dim, int(np.round(dim/3, 0))), np.arange(0, dim, int(np.round(dim/3, 0)))+1)
        plt.xlabel('# eigenvalue')
        plt.ylabel('magnitude')

    if kwargs.__contains__('matrix_type'):
        plt.title('Eigenvalues of the {} matrix'.format(kwargs['matrix_type']))




