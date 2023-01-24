#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: fmaschietto, bcallen95

from ..imports import *


def plot_pca_space(pcfit, labelsarr, names_list, out, **kwargs):
    """
    Plot pca space

    Parameters
    ----------
    pcfit: np.ndarray,
        array of fitted transformed data from [pca_call = sklearn.PCA() --> pcfit = pca_call.fit_transform(data)]
    labelsarr: np.ndarray,
        array of labels
            example: if the data is the concatenation of two simulations then
            ``data = np.concatenate([sim_1, sim_2])`` and ``labelsarr = np.asarray(names_list[0]*len(sim_2) + names_list[1]*len(sim_2))``;
            the order has to match the data.
    names_list: list of strings
        list containing the title name of each simulation
    out: str,
        plot name (``'/path/to/plot/plotname.pdf'``)
    """

    title  = ''
    levels = 20
    colors = sns.color_palette("bright", len(names_list))
    print(len(colors))
    if kwargs:
        if kwargs.__contains__('title'):
            title=kwargs['title']
        else:
            title = 'Simulation'
        if kwargs.__contains__('levels'):
            levels = kwargs['levels']
        else:
            levels = 20

        if kwargs.__contains__('colors'):
            colors= kwargs['colors']
        else:
            colors = sns.color_palette("bright", len(names_list))
    plt.figure(figsize=(20,20))

    s0 = pcfit[:,0]
    s1 = pcfit[:,1]
    df_pca = pd.DataFrame({'PC1': s0, 'PC2': s1, title: labelsarr})

    sns.displot(df_pca, x="PC1", y="PC2", hue=title, kind="kde", palette=colors, levels=levels)

    for label, name, color in zip(pd.unique(labelsarr), names_list, colors):
        ix = np.where(labelsarr == label)[0]
        plt.gca().scatter(s0[ix[0]], s1[ix[0]],  color=color,  marker='>',  s=200, zorder=100, edgecolor='k')
        plt.gca().scatter(s0[ix[-1]],s1[ix[-1]], color=color,  marker='8',  s=200, zorder=100, edgecolor='k')

    m1 = mlines.Line2D([], [], color='w', marker='>', markeredgewidth=.08, markeredgecolor='k',
                              markersize=130, label='Start of simulation')
    m2 = mlines.Line2D([], [], color='w', marker='8',  markeredgewidth=.08, markeredgecolor='k',
                              markersize=130, label='End of simulation')
    plt.savefig(out, format='pdf', bbox_inches='tight')

