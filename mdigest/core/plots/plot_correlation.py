#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: fmaschietto, bcallen95

from   mdigest.core.imports import  *
import mdigest.core.auxiliary as aux
from collections import OrderedDict
plt.rcParams.update({'figure.autolayout': True})
CB_color_cycle = sns.color_palette('bright', 10)


class Plots:
    def __init__(self, MDSIMA, MDSIMB, matrix_type='gcc_lmi', compute_centrality=True, **kwargs):
        """
        Description
        -----------
        General class to compare (plot) attributes from different simulations such as delta correlation heatmaps, eigenvalues
        Load desired attribute to plot using MDSIMA, MDSIMB object

        Parameters
        ----------
        MDSIMA: obj,
            correlation object for simulation of interest, could be derived from DynCorr, dihDynCorr, KS_Energy
        MDSIMB: obj,
            correlation object for simulation of interest (other than MDSIMA), could be derived from DynCorr, dihDynCorr, KS_Energy
        matrix_type: str,
            possible options are:
                - gcc_mi
                - gcc_lmi
                - exclusion
                - pcc
                - dcc
        compute_centrality: bool,
            whether to compute centrality
        kwargs: dict,
            possible options are:\
                - save, output filename in format /path/figurenoextension

        TODO  add other matrix types for dcorrelation and kscorrelation modules
        """

        if kwargs.__contains__('save'):
            self.save = kwargs['save']
        else:
            print('@>: provide output filename in format /path/figurenoextension')
            self.save=False

        self.params    = None
        self.num_replicas   = MDSIMA.num_replicas

        self.nresidues      = MDSIMA.nresidues

        if matrix_type == 'gcc_mi':
            self.matrix_A = np.asarray(
                [d['gcc_mi'] for d in list(OrderedDict(sorted(MDSIMA.gcc_allreplicas.items())).values())])
            self.matrix_B = np.asarray(
                [d['gcc_mi'] for d in list(OrderedDict(sorted(MDSIMB.gcc_allreplicas.items())).values())])

            if compute_centrality:
                self.centrality_A = np.asarray(
                    [aux.compute_eigenvector_centrality(mat, weight='weight')[1] for mat in self.matrix_A])
                self.centrality_B = np.asarray(
                    [aux.compute_eigenvector_centrality(mat, weight='weight')[1] for mat in self.matrix_B])
            else:
                self.centrality_A = np.asarray([d['gcc_mi'] for d in list(
                    OrderedDict(sorted(MDSIMA.eigenvector_centrality_allreplicas.items())).values())])
                self.centrality_B = np.asarray([d['gcc_mi'] for d in list(
                    OrderedDict(sorted(MDSIMB.eigenvector_centrality_allreplicas.items())).values())])

        if matrix_type == 'gcc_lmi':
            self.matrix_A  = np.asarray([d['gcc_lmi'] for d in list(OrderedDict(sorted(MDSIMA.gcc_allreplicas.items())).values())])
            self.matrix_B  = np.asarray([d['gcc_lmi'] for d in list(OrderedDict(sorted(MDSIMB.gcc_allreplicas.items())).values())])
            
            if compute_centrality:
                self.centrality_A = np.asarray(
                    [aux.compute_eigenvector_centrality(mat, weight='weight')[1] for mat in self.matrix_A])
                self.centrality_B = np.asarray(
                    [aux.compute_eigenvector_centrality(mat, weight='weight')[1] for mat in self.matrix_B])
            else:
                self.centrality_A = np.asarray([d['gcc_lmi'] for d in list(OrderedDict(sorted(MDSIMA.eigenvector_centrality_allreplicas.items())).values())])
                self.centrality_B = np.asarray([d['gcc_lmi'] for d in list(OrderedDict(sorted(MDSIMB.eigenvector_centrality_allreplicas.items())).values())])
        
        if matrix_type == 'exclusion':
            self.matrix_A         = np.asarray(list(OrderedDict(sorted(MDSIMA.exclusion_matrix_allreplicas.items())).values()))
            self.matrix_B         = np.asarray(list(OrderedDict(sorted(MDSIMB.exclusion_matrix_allreplicas.items())).values()))
            if compute_centrality:
                self.centrality_A    = np.asarray([aux.compute_eigenvector_centrality(mat, weight='weight')[1] for mat in self.matrix_A])
                self.centrality_B    = np.asarray([aux.compute_eigenvector_centrality(mat, weight='weight')[1] for mat in self.matrix_B])
            
        if matrix_type == 'pcc':
            self.matrix_A         = np.asarray(list(OrderedDict(sorted(MDSIMA.pcc_allreplicas.items())).values()))
            self.matrix_B         = np.asarray(list(OrderedDict(sorted(MDSIMB.pcc_allreplicas.items())).values()))
            if compute_centrality:
                self.centrality_A = np.asarray([aux.compute_eigenvector_centrality(mat, weight='weight')[1] for mat in self.matrix_A])
                self.centrality_B = np.asarray([aux.compute_eigenvector_centrality(mat, weight='weight')[1] for mat in self.matrix_B])
            
        if matrix_type == 'dcc':
            self.matrix_A         = np.asarray(list(OrderedDict(sorted(MDSIMA.dcc_allreplicas.items())).values()))
            self.matrix_B         = np.asarray(list(OrderedDict(sorted(MDSIMB.dcc_allreplicas.items())).values()))
            if compute_centrality:
                self.centrality_A = np.asarray([aux.compute_eigenvector_centrality(mat, weight='weight')[1] for mat in self.matrix_A])
                self.centrality_B = np.asarray([aux.compute_eigenvector_centrality(mat, weight='weight')[1] for mat in self.matrix_B])
            

        # sanity checks
        if MDSIMA.num_replicas != MDSIMB.num_replicas:
            print('the simualtions provider have been partitionned differently')
        if MDSIMA.nresidues != MDSIMB.nresidues:
            print('the simulations have different number of residues')


    def load_parameters(self, params):
        """
        Load parameters

        Parameters
        ----------
        params: dict,
            dictionary containing parameters for plotting\
                example: ``params={'labels':['APO', 'HOLO'], 'fig': plt.subplots(self.num_replicas, 3, figsize=(3*self.num_replicas+10,1*self.num_replicas+3.5))}``
        """

        self.params = params


    def _normalize_centrality(self, cent):
        """
        Normalize centrality array

        Parameters
        ----------
        cent: np.ndarray,
            array of centralities coefficients

        Returns
        -------
        :return np.ndarray,
            normalized centrality array
        """

        max_value = np.max(cent)
        min_value = np.min(cent)
        norm_centralities = np.zeros( cent.shape)
        for i in range(len(cent)):
            norm_centralities[i] = 2 * (( cent[i] - min_value) / (max_value - min_value)) - 1.0
        return norm_centralities


    def plot_gcc_per_replica(self):
        """
        Plot difference heatmaps of matrices for each replica
        """
        if self.params is not None:
            params = self.params
        else:
            params={'labels':['APO', 'HOLO'],
                    'fig': plt.subplots(self.num_replicas, 3, figsize=(3*self.num_replicas+10,1*self.num_replicas+3.5))}

        num_replicas = self.num_replicas

        fig, axs = params['fig']

        if  num_replicas > 1:
            a_mini = []
            a_maxi = []
            h_mini = []
            h_maxi = []
            d_maxi = []
            d_mini = []
            for m in range(num_replicas):
                pos_A = axs[0, m].imshow(self.matrix_A[m], cmap='nipy_spectral', origin='lower', interpolation='none', aspect='auto')
                pos_B = axs[1, m].imshow(self.matrix_B[m], cmap='nipy_spectral', origin='lower', interpolation='none', aspect='auto')
                pos_D = axs[2, m].imshow(self.matrix_B[m] - self.matrix_A[m], cmap='coolwarm', origin='lower', interpolation='none',aspect='auto')

                a_mini.append(np.min(self.matrix_A[m]))
                a_maxi.append(np.max(self.matrix_A[m]))
                h_mini.append(np.min(self.matrix_A[m]))
                h_maxi.append(np.max(self.matrix_B[m]))
                d_mini.append(np.min(self.matrix_B[m]-self.matrix_A[m]))
                d_maxi.append(np.max(self.matrix_B[m]-self.matrix_B[m]))
            for m in range(num_replicas):
                a_mini = np.min(np.asarray(a_mini))
                a_maxi = np.min(np.asarray(a_maxi))
                h_mini = np.min(np.asarray(h_mini))
                h_maxi = np.min(np.asarray(h_maxi))
                d_maxi = np.min(np.asarray(d_maxi))
                d_mini = np.min(np.asarray(d_mini))

                fig.colorbar(pos_A, ax=axs[0, m])
                fig.colorbar(pos_B, ax=axs[1, m])
                fig.colorbar(pos_D, ax=axs[2, m])
                axs[0, m].set_ylim(a_mini, a_maxi)
                axs[1, m].set_ylim(h_mini, h_maxi)
                axs[2, m].set_ylim(d_mini, d_maxi)
                axs[0, m].set_title('A rep. %d' %m)
                axs[1, m].set_title('B rep. %d' % m)
                axs[2, m].set_title(r'$\Delta_{B-A}$ rep. %d' % m)
        else:
            pos_A = axs[0].imshow(self.matrix_A[0], cmap='nipy_spectral', origin='lower', interpolation='none', aspect='auto')
            pos_H = axs[1].imshow(self.matrix_B[0], cmap='nipy_spectral', origin='lower', interpolation='none', aspect='auto')
            pos_D = axs[2].imshow(self.matrix_B[0] - self.matrix_A[0], cmap='coolwarm', origin='lower', interpolation='none',aspect='auto')
            fig.colorbar(pos_A, ax=axs[0])
            fig.colorbar(pos_H, ax=axs[1])
            fig.colorbar(pos_D, ax=axs[2])
            axs[0].set_title(params['labels'][0])
            axs[1].set_title(params['labels'][1])
            axs[2].set_title(r'$\Delta_{%s-%s}$' %(params['labels'][0],params['labels'][1]))

        for ax in axs.ravel():
            l = np.arange(0, self.nresidues, int(np.floor(self.nresidues/5)))
            ax.set_xticks(l)
            ax.set_xticklabels(l, rotation=60)
            ax.set_yticks(l)
            ax.set_yticklabels(l)
        if self.save:
          plt.savefig(str(self.save) + '_corr.pdf', format='pdf', bbox_inches='tight')
        plt.show()


    def plot_eigcent_per_replica(self):
        """
        Plot eigenvector centrality for each replica
        """

        plt.rcParams.update({'font.size': 30})
        def _outliers_percentile(centr, thr):
            mask_upper = centr > np.percentile(centr, thr)
            mask_lower = centr < np.percentile(centr, 100 - thr)
            return mask_upper, mask_lower

        num_replicas = self.num_replicas
        if 'fig' in self.params.keys():
            fig, axs = self.params['fig']
        else:
            fig, axs = plt.subplots(num_replicas+1,1, figsize=(3*num_replicas+10,1*num_replicas+6))

        x = np.arange(self.nresidues)
        for r in range(num_replicas):
            print('rep. %d' %r)
            c = self.centrality_B[r] - self.centrality_A[r]
            c = self._normalize_centrality(c)

            uppers, lowers = _outliers_percentile(c, 95)
            red = c[uppers]

            xr = np.where(uppers == True)[0]
            blue = c[lowers]
            xb = np.where(lowers == True)[0]

            print("rep. = %d,  RED %s" %(r, str(xr+1)))
            print("rep. = %d,  BLU %s" %(r, str(xb+1)))

            axs[r+1].plot(c, linestyle='-', color='tab:gray', linewidth=3)
            axs[r+1].set_ylabel('$\Delta$ centrality')
            axs[r+1].set_xlabel('# residues')
            axs[r+1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            axs[r+1].fill_between(x, np.min(c), c,
                                alpha=.2, color=CB_color_cycle[r], label='rep. %d' % r)
            axs[r+1].scatter(xr, red, s=150, lw=1,  color='tab:red', alpha=1., ec='k')
            axs[r+1].scatter(xb, blue, s=150, lw=1, color='tab:blue', alpha=1., ec='k')
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]

        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        plt.legend(lines, labels, ncol=5, loc='best', framealpha=False,
                   bbox_transform=plt.gcf().transFigure)
        fig.delaxes(ax=axs[0])
        if self.save:
          plt.savefig(str(self.save) + '_eig.pdf', format='pdf', bbox_inches='tight')
        plt.show()



