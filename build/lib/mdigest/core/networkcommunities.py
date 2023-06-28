"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: fmaschietto, bcallen95"""

from mdigest.core.imports import *
import mdigest.core.savedata as sd
import mdigest.core.toolkit as tk
import mdigest.core.auxiliary as aux

import itertools
import copy
from operator import itemgetter

import community
import networkx as nx
# from networkx.algorithms.community.louvain         import louvain_communities as community_louvain
from networkx.algorithms.shortest_paths.dense import reconstruct_path as reconstruct
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community.quality import modularity
from networkx.algorithms.shortest_paths.dense import floyd_warshall_predecessor_and_distance as nxFWPD

import networkx.algorithms.community.quality as nxquality
from networkx import eigenvector_centrality_numpy as nxeigencentrality
from networkx import edge_betweenness_centrality as nxbetweenness


class CMTY:


    def __init__(self):
        """
        Description
        -----------
        Class to handle computation of communities from correlated data

        Attributes
        ----------
        self.cmty_data: obj,
            class object,
        self.verbosity: bool,
            set verbosity
        self.threshold: int,
            impose convergence for Girvan-Newman (GN) algorhitm when number of cycles exceeds  threshold
        self.exclusion_matrix: np.ndarray or dict of np.ndarrays,
            exclusion matrix/ces
        self.distance_matrix: np.ndarray or dict of np.ndarrays,
            distance_matrix,
        self.matrix_dict: nested dict,
            data for community andalysis,
                - example: ``matrix_dictionary = OD({ 'entry_1': dih_gcc_1, 'entry_2':gcc_2})``
        self.pruned_dict: nested dict,
            data for community andalysis where the correlation matrices are filtered
                - ``filters_dictionary = OD({ 'entry_1':excl_mat_1, 'entry_2': excl_mat_1})
                - ``mdigest.CMTY.populate_filters(filters_dictionary=filters_dictionary)``
                - ``filters = {'exclusion': True}``
                - ``mdigest.CMTY.assign_filters(filters=filters)``
        self.filter_dict: nested dict,
            each entry in self.matrix_dict is assigned a filter_dict, which decides which kind of filter will applied to that entry
            ``filter_dict[key] = {'exclusion': True, 'filter_by': False}``
        self.nxGraphs_dict: dict,
            ordered dictionary with the same keys as self.matrix_dict and corresponding nx.Graph as values
        self.nxGraphs: list of nx.Graph object,
            list of graphs where the Graphs appear in the same order as in nxGraphs_dict
        self.num_instances: int,
            number of instances, corresponding to the lenght of self.matrix_dict
        self.list2dict: dict,
            dict with integers (range(num_instances)) as keys and self.matrix_dict keys as values
        self.number_of_nodes: int,
            number or nodes
        self.bestQ_collect: dict,
            best modularity
        self.Qlist_collect: list,
            modularity values list
        self.communities_list_collect: list,
            list of communities from GN
        self.reference_rep_comm_list: list,
            reference iteration in the Louvain procedure (the iteration that affords the largest modularity)
        self.reference_idx: int,
            index of reference iteration
        self.num_iterations: int,
            iteration for Louvain algorithm
        self.nodes_communities_collect: dict,
            nodes as keys, assigned community as values
        self.max_betw_dict_collect: dict,
            maximum betweennes edges at each GN iteration are stored as keys, corresponding betweennesses as values
        self.betweenness: dict,
            betweeness in entire system, for each matrix instance (entry in graph list). 
            Number of shortest paths passign through a given edge computed for each
        self.predecessors: dict, 
            predecessors dictionary
        self.partitions_collect: dict,
            dict with community index as nodes and list of nodes belonging to that community as values
        self.distances_collect: nested  dict,
            dicts, keyed by source and target, of predecessors and distances in the shortest path for each entry in graph list,
            stored in a dictionary with entry name as keys.
        self.max_distance: float
            maximum travelled distance between any two nodes
        self.max_direct_distance: float
            maximum direct distance between two nodes
        """

        self.cmty_data = None
        self.verbosity = None
        self.threshold = None
        self.exclusion_matrix = None
        self.distance_matrix = None
        self.matrix_dict = {}
        self.pruned_dict = OrderedDict()
        self.filter_dict = OrderedDict()
        self.nxGraphs_dict = OrderedDict()
        self.nxGraphs = []
        self.num_instances = 0
        self.list2dict = {}

        # Girvan-Newman / Louvain attributes
        self.number_of_nodes = None
        self.bestQ_collect = None
        self.Qlist_collect = None
        self.communities_list_collect = None
        self.reference_rep_comm_list = None
        self.reference_idx = None
        self.num_iterations = None

        self.nodes_communities_collect = {}
        self.max_betw_dict_collect = {}
        self.betweenness = {}
        #
        self.predecessors = {}
        self.partitions_collect = None
        self.distances_collect = None
        self.max_distance = None
        self.max_direct_distance = None

    def save_class(self, file_name_root='../output/cache/community', save_space=True):
        """
        General function to save instances of the CMTY classs to file

        Parameters
        ----------
        file_name_root: str,
            file rootname

        """

        self.cmty_data = sd.MDSdata()
        self.cmty_data.cmty_data = self.cmty_data
        self.cmty_data.nxGraphs_dict = self.nxGraphs_dict
        self.cmty_data.nodes_communities_collect = self.nodes_communities_collect
        self.cmty_data.max_betw_dict_collect = self.max_betw_dict_collect
        # self.cmty_data.betweenness               = self.betweenness
        self.cmty_data.partitions_collect = self.partitions_collect
        self.cmty_data.distances_collect = self.distances_collect
        self.cmty_data.max_distance = self.max_distance
        self.cmty_data.max_direct_distance = self.max_direct_distance
        if not save_space:
            self.cmty_data.predecessors = self.predecessors
        self.cmty_data.save_to_file(file_name_root)

    # ----------------- general functions to load graphs and parameters --------------------------------------#

    def create_matrix_dict(self, matrix_dictionary):
        """
        Populate matrix_dictionary attribute with kwargs

        Parameters
        ----------
        matrix_dictionary: dict,
            dictionary with format of ``{'matrix_label': np.array or class_object.matrix_attribute}`` containing
            matrices to feed into the community pipeline
        """

        self.matrix_dict = OrderedDict(matrix_dictionary)

    def populate_filters(self, filters_dictionary):
        """
        Populate exclusion_matrix and distance_matrix attributes with filter_dictionary.

        Parameters
        ----------
        filters_dictionary: dict,
            dictionary with format ``filters_dictionary={'exclusion_matrix': mat}``;  mat can be either None, np.array or class_object.matrix_attribute or dict
            containing an exclusion matrix for each matrix in self.matrix_dict.
            The list of keys must match those in self.matrix_dict.
            To include filtering by distance, use ``filters_dictionary={'distance_matrix': mat}``, where mat is as np.array or class_object.matrix_attribute or dict
            containing a distance matrix for each entry in self.matrix_dict. The list of keys must match those in self.matrix_dict.


        Examples
        ---------

         - ``filters_dictionary = { 'exclusion_matrix': np.array}`` or  ``filters_dictionary ='distance_matrix': np.array }`` or
         - ``filters_dictionary = { 'exclusion_matrix': {'ca_lmi_rep_0': np.array,'ca_lmi_rep_1': np.array},
                                  'distance_matrix': {'ca_lmi_rep_0': np.array,'ca_lmi_rep_1': np.array }}``.
        keys in exclusion matrix and distance matrix have to match
        """

        if filters_dictionary.__contains__('exclusion_matrix'):
            self.exclusion_matrix = filters_dictionary['exclusion_matrix']

        if filters_dictionary.__contains__('distance_matrix'):
            self.distance_matrix = filters_dictionary['distance_matrix']

    def assign_filters(self, filters=None):
        """

        Parameters
        ----------
        filters: dict,
            dictionary with format of {'exclusion': bool, 'filter_by': bool}, specifying whether to apply
            specified filter (Default is False for all keys.)
        """

        if filters is None:
            self.filter_dict = OrderedDict()

            for key in self.matrix_dict.keys():
                self.filter_dict[key] = {'exclusion': False, 'filter_by': False}
                warnings.warn("Setting default exclusion/filter criteria. (No filter will be applied.)")
        else:
            # instantiate self.filters
            filters = OrderedDict(filters)
            for key in self.matrix_dict.keys():
                self.filter_dict[key] = {'exclusion': False, 'filter_by': False}

                if filters.__contains__('exclusion'):
                    self.filter_dict[key] = {'exclusion': True, 'filter_by': False}

                if filters.__contains__('filter_by'):
                    self.filter_dict[key] = {'exclusion': False, 'filter_by': True}

            if self.exclusion_matrix is None:
                warnings.warn(
                    "No exclusion matrix has been loaded, call self.populate_filters({'exclusion_matrix': exc_matrix}) "
                    "to suppress this warning.")
            if self.distance_matrix is None:
                warnings.warn(
                    "No distance matrix has been loaded, call self.populate_filters({'distance_matrix': dist_matrix}) "
                    "to suppress this warning.")

    def set_parameters(self, parameters):
        """
        Set parameters
        """
        if parameters.__contains__("VERBOSE"):
            self.verbosity = parameters['VERBOSE']

        else:
            self.verbosity = False
        print("@>: verbosity = %s" % self.verbosity)

        if parameters.__contains__("THRESHOLD"):
            self.threshold = parameters['THRESHOLD']
        else:
            self.threshold = None

        if parameters.__contains__("LOUVAIN_ITERATIONS"):
            self.num_iterations = parameters['LOUVAIN_ITERATIONS']
            print("@>: number of iterations is set to %d" % self.num_iterations)
        else:
            self.num_iterations = None
            print("@>: Louvain set to None")

    def load_graph(self, distance_threshold=5.0):
        """
        General function to load graph

        Parameters
        ----------
        distance_threshold: threshold applied in prune_adjacency

        """

        for key, mat_instance in self.matrix_dict.items():
            print('processing %s ...' % key)

            # We substitute zeros for a non-zero value to avoid "zero division" warnings
            #   from the np.log transformation below.
            mat_instance[np.where(mat_instance == 0)] = 10 ** -11

            # Use log transformation for network distance calculations.

            mat_instance = -1.0 * np.emath.log(mat_instance)

            # Now we guarantee that the previous transformation does not
            #   create "negative infitite" distances. We set those to zero.
            mat_instance[np.where(np.isinf(mat_instance))] = 0.

            if isinstance(mat_instance[0, 0], complex):
                dimensions = mat_instance.shape
                cast2real = np.zeros((dimensions[0] ** 2))
                mat_instance = mat_instance.flatten()
                for i in range(len(mat_instance)):
                    cast2real[i] = mat_instance[i].real
                mat_instance = cast2real.reshape(dimensions)

            if self.filter_dict.__contains__(key):
                filter_by = self.filter_dict[key]['filter_by']
                exclusion = self.filter_dict[key]['exclusion']
            else:
                filter_by = self.filter_dict['filter_by']
                exclusion = self.filter_dict['exclusion']

            if filter_by:
                if isinstance(self.distance_matrix, dict):
                    mat_instance = aux.prune_adjacency(mat_instance, self.distance_matrix[key],
                                                       loc_factor=distance_threshold)
            elif exclusion:
                if isinstance(self.exclusion_matrix, dict):
                    mat_instance[self.exclusion_matrix[key] == 1] = 0.
                else:
                    print('@>: loaded {} matrix as nxGraph'.format(key))

                    mat_instance[np.asarray(self.exclusion_matrix, dtype=int) == 1] = 0.
                self.pruned_dict.update({key: mat_instance})
            else:
                pass

            print(f"@>: there are {np.count_nonzero(mat_instance):d} nonzero elements in this matrix")
            self.nxGraphs_dict[key] = aux.create_graph(mat_instance)
            for pair in self.nxGraphs_dict[key].edges.keys():
                self.nxGraphs_dict[key].edges[(pair[0], pair[1])]["dist"] = mat_instance[pair[0], pair[1]]
            print('@>: graph has %d edges' % len(self.nxGraphs_dict[key].edges))

            # Sets the degree of each node.
            degree_dict = dict(self.nxGraphs_dict[key].degree(self.nxGraphs_dict[key].nodes()))
            nx.set_node_attributes(self.nxGraphs_dict[key], degree_dict, 'degree')
            self.number_of_nodes = len(nx.nodes(self.nxGraphs_dict[key]))
            self.nxGraphs.append(self.nxGraphs_dict[key])
            self.num_instances = len(self.nxGraphs)

            self.list2dict = dict(zip(np.arange(len(self.nxGraphs_dict.keys())), list(self.nxGraphs_dict.keys())))

    # ----------------- community partitioning --------------------------------------#

    def _betweenness(self, predecessors, n):
        """
        Calculate betweenness
        """
        betweenmat = np.zeros((n, n))
        for i in predecessors.keys():
            for j in predecessors[i].keys():
                path = reconstruct(i, j, predecessors)
                for k in range(len(path) - 1):
                    betweenmat[path[k], path[k + 1]] += 1
                    betweenmat[path[k + 1], path[k]] += 1
        return betweenmat

    def most_valuable_edge_fw(self, G):
        """
        Returns most vauable edge, using Floyd-Warshall algoritm
        """

        pred, distances = nx.floyd_warshall_predecessor_and_distance(G)
        n = G.number_of_nodes()
        betweenness_mat = self._betweenness(pred, n)
        edge = np.unravel_index(np.argmax(betweenness_mat), np.array(betweenness_mat).shape)
        if self.verbosity:
            print("@>: most valuable edge:", list(edge))
        return (edge[0], edge[1]), np.amax(betweenness_mat)

    def most_valuable_edge(self, G, count_entries=False, normalized=False, weight='weight'):
        """ Returns most valuable edge according to edge_betweenness_centrality criterion

        Parameters
        ----------
        count_entries: bool,
            use true to print betweenness values calculated without averaging over all shortest paths
            this is how betweenness values are calculated in the original floyd_warshall.c code.
        normalized: bool,
            decides whether betweennesses are normalized or not

        weight: str,
            default 'weigth', uses graph weights

        Returns
        --------
        (edge_key[0], edge_key[1]): tuple,
        maxbet: float
            (edge tuple), maximum_betweeenness
        """

        bw = nx.edge_betweenness_centrality(G, normalized=normalized, weight=weight)
        maxbet = max(bw.values())
        edge_key = tk.keywithmaxval(bw)
        if count_entries:
            A = nx.to_numpy_matrix(G, nodelist=None, weight=None)
            np.fill_diagonal(A, 0.)
            modA = np.linalg.norm(A) / 2
            maxbet = max(bw.values()) * modA
        if (self.verbosity == True) and count_entries:
            print('@>: most valuable edge - count entries:', edge_key, maxbet, modA)

        elif (self.verbosity == True) and not count_entries:
            print('@>: most valuable edge:', edge_key, maxbet)
        return (edge_key[0], edge_key[1]), maxbet

    def _communities_to_partitions(self, communities):
        """
        Convert communities to partitions
        """
        partitions = {}
        for cind, comm in enumerate(communities):
            for n, node in enumerate(comm):
                partitions.update({comm[n]: cind})

        partitions = {key: value for key, value in sorted(partitions.items(), key=lambda item: item[0])}
        return partitions

    def community_data(self, G, partition):
        """
        Compact all information on the communities into a dictionary

        Parameters
        ----------

        G: nx.Graph(),
            a networkx protein graph
        partition: dict,
            dictionary containing the different partitions

        Returns
        ---------
        nodes_communities: dict, n_communities: int
            a dictionary containing data relative to each community:
             - community labels,
             - community index ordered by modularity,
             - community index ordered by eigenvector centrality,
             - community nodes (list of nodes in each partition)
        """

        nodes_communities = {}
        communities_labels = np.unique(np.asarray(list(partition.values()), dtype=int))

        n_communities = len(np.unique(list(partition.values())))
        nodes_communities["comm_labels"] = copy.deepcopy(communities_labels)
        nx.set_node_attributes(G, partition, 'modularity')

        nodes_communities["comm_nodes"] = {}

        for comm_ind, comm in enumerate(communities_labels):
            nodes_in_comm = [n for n in G.nodes() if G.nodes[n]['modularity'] == comm]

            # Then create a dictionary of the eigenvector centralities of those nodes
            self._calc_eigen_central(G)
            nodes_in_comm_eigvs = {n: G.nodes[n]['eigenvector'] for n in nodes_in_comm}

            # Then sort that dictionary
            nodes_in_comm_eigvs_ord = sorted(nodes_in_comm_eigvs.items(), key=itemgetter(1), reverse=True)
            nodes_in_comm_eigvs_ordlist = [x[0] for x in nodes_in_comm_eigvs_ord]

            nodes_communities["comm_nodes"][comm] = copy.deepcopy(nodes_in_comm_eigvs_ordlist)

        return nodes_communities, n_communities

    def sort_cmty(self, cycles, setgraph=-1):
        """
        Sort communities and store sorted indices according to different metrics to a dictionary

        Parameters
        ----------
        cycles: int,
            assign the number of cycles to match the number of replicas (number of graphs on which
            to iterate)
        setgraph: int,
            which graph to use;  if -1 use graph corresponding to louvain cycle
        """

        modularity_collect = []
        self.reference_rep_comm_list = []

        for cycle in range(cycles):

            if setgraph == -1:
                G = self.nxGraphs[cycle]
            else:
                G = self.nxGraphs[setgraph]

            # Orders communities based on size.

            communities_by_size = list(sorted(self.nodes_communities_collect[cycle]["comm_nodes"].keys(),
                                              key=lambda k: len(self.nodes_communities_collect[cycle]["comm_nodes"][k]),
                                              reverse=True))

            self.nodes_communities_collect[cycle]["comm_by_size"] = copy.deepcopy(communities_by_size)

            # Orders communities based on highest eigenvector centrality of all its nodes.
            communities_by_eigen_centr = list(sorted(self.nodes_communities_collect[cycle]["comm_nodes"].keys(),
                                                     key=lambda k:
                                                     G.nodes[self.nodes_communities_collect[cycle]["comm_nodes"][k][0]][
                                                         'eigenvector'], reverse=True))

            self.nodes_communities_collect[cycle]["comm_by_eigen_centr"] = copy.deepcopy(communities_by_eigen_centr)

            # Creates a list of repetitions and order them according to graph modularity.

            modul = nxquality.modularity(G, [set(nodes_list) for nodes_list in
                                             self.nodes_communities_collect[cycle]["comm_nodes"].values()])

            modularity_collect.append((cycle, modul))

            modularity_collect.sort(key=lambda x: x[1], reverse=True)

            # Keep the instance with the highest modularity as a reference for community matching
            reference_idx = modularity_collect[0][0]

        self.reference_idx = reference_idx

        print("@>: reference instance", self.reference_idx, "with modularity = ", modularity_collect[0][1])

        for cycle in range(cycles):
            color_index = np.zeros(len(self.nodes_communities_collect[cycle]["comm_nodes"])).astype(int)

            # Assingn color index accoding to max common element
            for c in range(self.communities_list_collect[cycle]):
                comm = [k for k in self.partitions_collect[cycle].keys() if self.partitions_collect[cycle][k] == c]
                color_index[c] = self._max_common_elements(comm, list(
                    self.nodes_communities_collect[self.reference_idx]["comm_nodes"].values()))
            self.nodes_communities_collect[cycle].update({'color_index': color_index})
        self.reference_rep_comm_list = list(self.nodes_communities_collect[self.reference_idx]["comm_nodes"].values())

    def girvannewman(self, MVE=None):
        """
        Computes Girvan Newman algorithm, slightly faster than Run_Girvan_Newman(), uses builtin functions from nx

        Parameters
        ----------
        MVE: function
            function to calculate most valuable edge. Default is None, which is equivalent to calling
            most_valuable_edge_nx().

        """

        Qlist_collect = {}
        max_btw_dict_collect = {}
        bestQ_collect = {}
        nodes_communities_collect = {}
        communities_list_collect = []
        partitions_collect = []

        for g_idx, current_G in enumerate(self.nxGraphs):

            print('@>: run Girvan Newman for graph %d' % g_idx, self.list2dict[g_idx])
            G = current_G
            G_o = current_G.copy()

            bestQ = 0.0
            Q_list = []

            counter = 1
            if self.threshold is None:
                self.threshold = 10
                print("@>: threshold is set to %d" % self.threshold)
            k = self.threshold  # keep this reasonably large, so that the maximum modularity split corresponds to a number of
            # communities < 50
            undivided = [set(G.nodes()), set()]
            Q = modularity(G, undivided)  # weight is default
            Q_list.append(Q)

            comp = girvan_newman(G)

            limited = itertools.takewhile(lambda commu: len(commu) <= k, comp)

            for communities_ in limited:
                counter += 1

                Q = modularity(G, communities_)

                if Q > bestQ:
                    bestQ = Q
                    communities = list(communities_)

                Q_list.append(Q)

                if self.verbosity:
                    if bestQ > 0.0:
                        print("@>: max modularity (Q):", bestQ)
                        print("@>: graph communities:", communities)
                    else:
                        print("@>: max modularity (Q):", bestQ)

            Qlist_collect.update({g_idx: Q_list})
            bestQ_collect.update({g_idx: bestQ})
            communities = [list(comm) for comm in communities]

            nodes_communities_collect[g_idx] = {}
            partition = self._communities_to_partitions(communities)

            nodes_communities, n_communities = self.community_data(G, partition)
            nodes_communities_collect[g_idx] = nodes_communities

            communities_list_collect.append(n_communities)
            partitions_collect.append(partition)

        self.bestQ_collect = bestQ_collect
        self.Qlist_collect = Qlist_collect
        self.nodes_communities_collect = nodes_communities_collect
        self.partitions_collect = partitions_collect
        self.communities_list_collect = communities_list_collect
        self.max_betw_dict_collect = max_btw_dict_collect

        self.sort_cmty(self.num_instances, setgraph=-1)


    def best_iteration_louvain(self, G, nodes_comm_alliter, comm_list_alliter, partitions_alliter):
        """
        Select the best iteration from Louvain heuristic algorithm by selecting
        the instance with the highest modularity

        Parameters
        ----------
        G: nx.Graph(),
            a networkx protein graph
        nodes_comm_alliter: dict,
            dictionary of Dictionary containing the list of nodes for each community for at every iteration of Louvain
            heuristic algorithm
        comm_list_alliter: list,
            contains the list of nodes for each community at every iteration of Louvain heuristic algorithm
        partitions_alliter: list
            partitions at every iteration of Louvain heuristic algorithm

        Returns
        --------
         best_iteration_comm_nodes: dict,
            communities nodes for the best iteration
        best_comm_list: list,
            list containing the index of the communities in the `best run` best_partitions
        best_partitions: dict,
            best partitions
        """

        modularity_alliter = []
        reference_idx = 0
        for itera in range(self.num_iterations):
            # Creates a list of repetitions and order them according to graph modularity.
            modul = nxquality.modularity(G, [set(nodes_list) for nodes_list in
                                             nodes_comm_alliter[itera]["comm_nodes"].values()])

            modularity_alliter.append((itera, modul))
            modularity_alliter.sort(key=lambda x: x[1], reverse=True)

            # Keep the instance with the highest modularity as a reference for community matching
            reference_idx = modularity_alliter[0][0]

        best_iteration_comm_nodes = nodes_comm_alliter[reference_idx]
        best_comm_list = comm_list_alliter[reference_idx]
        best_partitions = partitions_alliter[reference_idx]
        return best_iteration_comm_nodes, best_comm_list, best_partitions

    def _max_common_elements(self, list1, lists_ref):
        """
        Assign community according to maximum overlap with reference communities.
        """
        common = []
        for lr in lists_ref:
            common.append(np.sum([l in lr for l in list1]))
        return np.argmax(common)

    def _calc_eigen_central(self, G):
        """
        Compute eigenvector centrality to rank the nodes in the network.
        """

        cent = nxeigencentrality(G, weight='weight')
        nx.set_node_attributes(G, cent, 'eigenvector')

    def run_cmty_louvain(self, setgraph=0):
        # TODO check if this function is actually necessary (other than run_cmtys_louvain())
        # TODO if keep add aggregate option

        """
        Runs community louvain (one graph) and returns a dictionary with communities at each iteration
        of the louvain procedure, can be useful if one wants to check consistency across different louvain iterations for a
        single replica.

        Parameters
        ----------
        setgraph: int,
            default 0, assing to integer correponding to replica on which to apply Louvain algorithm
        """

        np.random.seed(1)

        G = self.nxGraphs[setgraph]
        print("@>: partitioning graph n. {} in graphs list".format(setgraph))
        print("@>: louvain algorithm will be iterated {} times".format(self.num_iterations))

        self.nodes_communities_collect = {}
        self.partitions_collect = []
        self.communities_list_collect = []
        for itera in range(self.num_iterations + 1):
            self.nodes_communities_collect[itera] = {}

            partition = community.best_partition(G, weight='degree')
            # partition = community_louvain(G, weight="weight", resolution=0, threshold=0.0000001, seed=None)
            # tmp_ = {}
            # partition = [list(p) for p in partition]
            # for i, p in enumerate(partition):
            #     for node in p:
            #         tmp_.update({node:i})
            # partition = tmp_

            nodes_communities, n_communities = self.community_data(G, partition)
            self.nodes_communities_collect[itera] = nodes_communities
            self.communities_list_collect.append(n_communities)
            self.partitions_collect.append(partition)
        self.sort_cmty(self.num_iterations, setgraph=setgraph)

    def run_cmtys_louvain(self, aggregate=False, **kwargs):
        """
        Community generation using the louvain protocol, iterate over multiple replicas, for each save
        the partition with higher modularity to a dictionary

        Parameters
        ----------
        aggregate: bool,
            whether to group communities by redistributing nodes of communities smaller than given threshold
            over the other communities. Aggregation assign each node to the partition that has yields the larges
            modularity.
        kwargs: dict,
            use threshold = int to set threshold for regrouping communities,
            default is 5 (communities <= 5 elements are redistributed)
        """

        def _reassign_and_modularity(graph, partitions, lone_partition_list):
            """Reassign node to the community which affords the largest network modularity
            """
            original = partitions.copy()
            tmp_partitions = original.copy()

            for lp, p in enumerate(lone_partition_list):
                discard = tmp_partitions.pop(p)

                print('@>: reassign node {}'.format(discard))

                modularity_arr = np.zeros(len(original))

                for pidx, (key, val) in enumerate(tmp_partitions.items()):
                    tmp_partitions[key] += discard

                    modul = nxquality.modularity(graph, [list(nodes_list) for nodes_list in tmp_partitions.values()])
                    modularity_arr[key] = modul

                    tmp_partitions[key] = tmp_partitions[key][:-len(discard)]
                    best_modularity = np.argmax(modularity_arr)

                tmp_partitions[best_modularity] += discard

            final_partitions = {}
            for i, (key, val) in enumerate(tmp_partitions.items()):
                final_partitions.update({i: val})

            return final_partitions

        np.random.seed(1)
        self.nodes_communities_collect = {}
        self.communities_list_collect = []
        self.partitions_collect = []

        for instance in range(self.num_instances):
            np.random.seed(1)
            G = self.nxGraphs[instance]

            communities_list_alliter = []
            partitions_alliter = []

            nodes_communities_alliter = {}
            for itera in range(self.num_iterations):
                partition = community.best_partition(G, weight='degree')

                ###############
                # uncomment to use community louvain package
                res = {}
                for k, v in partition.items():
                    res[v] = [k] if v not in res.keys() else res[v] + [k]
                partition = tk.dict2list(res)

                ###############
                # uncomment to use nx implementation of community_louvain
                # partition = community_louvain(G, weight="weight", resolution=0, threshold=0.0000001, seed=None)
                # partition = [list(p) for p in partition]
                ###################################

                if aggregate:
                    if kwargs:
                        thr = kwargs['threshold']
                    else:
                        thr = 5
                    ref_partitions = tk.list2dict(partition)
                    keys = []
                    for k, v in ref_partitions.items():
                        if 0 < len(v) <= thr:
                            keys.append(k)

                    partition = _reassign_and_modularity(G, ref_partitions, keys)
                else:
                    partition = tk.list2dict(partition)
                tmp_ = {}
                for k, part in partition.items():
                    for node in part:
                        tmp_.update({node: k})
                partition = tmp_

                nodes_communities, n_communities = self.community_data(G, partition)
                nodes_communities_alliter.update({itera: nodes_communities})
                communities_list_alliter.append(n_communities)
                partitions_alliter.append(partition)

            best_iter_comm_nodes, best_comm_list, best_partitions = self.best_iteration_louvain(G,
                                                nodes_communities_alliter, communities_list_alliter, partitions_alliter)

            self.nodes_communities_collect.update({instance: best_iter_comm_nodes})
            self.communities_list_collect.append(best_comm_list)
            self.partitions_collect.append(best_partitions)

        self.sort_cmty(self.num_instances, setgraph=-1)
        if self.verbosity:
            print('@>: partitions_collect', len(self.partitions_collect))
            print('@>: communities_list_collect', len(self.communities_list_collect))
            print('@>: communities keys', self.nodes_communities_collect.keys())

    # -------------------- community analysis --------------------------------------

    def _inter_community_betweeness(self, partition, G):
        """
        Calculate inter community betweenness

        Parameters
        ----------
        partition: dict,
        G: nx.Graph

        Returns
        -------
        :returns intercomm: dict,
            inter community betweenness
        """
        B = nx.edge_betweenness_centrality(G)
        n_communities = len(np.unique(list(partition.values())))
        intercomm = np.zeros((n_communities, n_communities))
        for key in B.keys():
            c1 = partition[key[0]]
            c2 = partition[key[1]]
            intercomm[c1, c2] += B[key]
            intercomm[c2, c1] += B[key]
        np.fill_diagonal(intercomm, 0)
        return intercomm

    def calculate_betweenness(self):
        """
        Calculate betweenness in entire system, for each matrix instance (entry in graph list).
        Store each in an ordered dictionary
        """
        betweenness = {}

        for instance in tk.log_progress(range(self.num_instances), every=1, size=self.num_instances, name="Instance"):
            # Compute betweeness in entire system, for each matrix instance (entry in graph list).

            # neglect weights - the betweenness estimates the number of shortest paths passign through a given edge
            betweenness[instance] = nxbetweenness(self.nxGraphs[instance], weight=None)

            # Creates an ordered dict of pairs with betweenness > than zero.
            betweenness[instance] = {k: self.betweenness[instance][k] for k in self.betweenness[instance].keys() if
                                          self.betweenness[instance][k] > 0}
            self.betweenness[instance] = OrderedDict(
                sorted(betweenness[instance].items(), key=lambda t: t[1], reverse=True))

    def get_degree(self, instance=0):
        return dict(self.nxGraphs[instance].degree(self.nxGraphs[instance].nodes()))

    def compute_optimal_paths(self):
        """
        Compute optimal source-target paths using the Floyd-Warshall algorithm
        """

        # get the network distance arrays
        self.distances_collect = np.zeros([self.num_instances, self.number_of_nodes, self.number_of_nodes],
                                          dtype=np.float)
        self.predecessors = {}
        for i in range(self.num_instances):
            self.predecessors[i] = 0

        for instance in tk.log_progress(range(self.num_instances), name="Instance"):
            # use the "distance" as weight, (i.e. the log-transformation of the correlations, rather than
            # the correlation itself.
            fw_predecessors, fw_distances = nx.floyd_warshall_predecessor_and_distance(self.nxGraphs[instance],
                                                                                       weight='dist')
            # turns dictionary of distances into 2D np.array per instance
            self.distances_collect[instance, :, :] = np.array(
                [[fw_distances[i][j] for i in sorted(fw_distances[j])] for j in sorted(fw_distances)])

            # combines predecessor dictionaries from all replicas and save in class attribute
            self.predecessors[instance] = copy.deepcopy(fw_predecessors)

        # get maximum network distance
        pruned = np.asarray(list(self.pruned_dict.values()))
        self.max_distance = np.max(pruned[self.distances_collect != np.inf])

        # set -1 as distance of nodes with no connecting path (rather than np.inf)
        self.distances_collect[np.where(np.isinf(self.distances_collect))] = -1

        # check max network distance between directly connected nodes
        self.max_direct_distance = max(
            [self.distances_collect[instance, pruned[instance, :, :] > 0].max() for instance in
             range(self.num_instances)])

    def retrieve_path(self, node_A, node_B, instance):
        """
        Reconstruct path
        """
        return nx.reconstruct_path(node_A, node_B, self.predecessors[instance])





def display_shortes_path(nvView, path, dists, max_direct_dist, selected_atomnodes, opacity=0.75,
                         color="green", side="both", segments=5, disable_impostor=True, use_cylinder=True):
    """
    Display shortest paths
    """
    for source, target in zip(path, path[1:]):

        source_sel = tk.get_NGLselection_from_node(source, selected_atomnodes)
        target_sel = tk.get_NGLselection_from_node(target, selected_atomnodes)

        if dists[source, target] == 0:
            continue

        radius = 0.05 + (0.5 * (dists[source, target] / max_direct_dist))

        nvView.add_representation("distance", atom_pair=[[source_sel, target_sel]], color=color, label_visible=False,
                                  side=side, name="link", use_cylinder=use_cylinder, radial_sements=segments,
                                  radius=radius, disable_impostor=disable_impostor, opacity=opacity, lazy=True)

