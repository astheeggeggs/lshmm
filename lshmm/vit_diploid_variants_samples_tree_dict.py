import copy
from math import isclose

import numpy as np
import tskit


def mirror_coordinates(ts):
    """
    Returns a copy of the specified tree sequence in which all
    coordinates x are transformed into L - x.
    """
    L = ts.sequence_length
    tables = ts.dump_tables()
    left = tables.edges.left
    right = tables.edges.right
    tables.edges.left = L - right
    tables.edges.right = L - left
    tables.sites.position = L - tables.sites.position  # + 1
    # TODO migrations.
    tables.sort()
    return tables.tree_sequence()


class ValueTransition:
    """Simple struct holding value transition values."""

    def __init__(self, tree_node=-1, inner_summation=-1, value_list=-1, value_index=-1):
        self.tree_node = tree_node
        self.value_list = value_list
        self.inner_summation = inner_summation
        self.value_index = value_index

    def copy(self):
        return ValueTransition(
            self.tree_node,
            self.inner_summation,
            self.value_list,
            self.value_index,
        )

    def __repr__(self):
        return repr(self.__dict__)

    def __str__(self):
        return repr(self)


class InternalValueTransition:
    """Simple struct holding the internal value transition values."""

    def __init__(self, tree_node=-1, value=-1, inner_summation=-1):
        self.tree_node = tree_node
        self.value = value
        self.inner_summation = inner_summation

    def __repr__(self):
        return repr(self.__dict__)

    def __str__(self):
        return repr(self)


class LsHmmAlgorithm:
    """
    Abstract superclass of Li and Stephens HMM algorithm.
    """

    def __init__(self, ts, rho, mu, precision=30):
        self.ts = ts
        self.mu = mu
        self.rho = rho
        self.precision = precision
        # The array of ValueTransitions.
        self.T = []
        # indexes in to the T array for each node.
        self.T_index = np.zeros(ts.num_nodes, dtype=int) - 1
        # The number of nodes underneath each element in the T array.
        self.N = np.zeros(ts.num_nodes, dtype=int)
        # Efficiently compute the allelic state at a site
        self.allelic_state = np.zeros(ts.num_nodes, dtype=int) - 1
        # Diffs so we can can update T and T_index between trees.
        self.edge_diffs = self.ts.edge_diffs()
        self.parent = np.zeros(self.ts.num_nodes, dtype=int) - 1
        self.tree = tskit.Tree(self.ts)
        self.output = None

    def decode_site_dict(self):
        """
        Decodes the tree encoding of the values into an explicit
        matrix.
        """
        A = np.zeros((self.ts.num_samples, self.ts.num_samples))
        # To look at the inner summations too.
        B = np.zeros((self.ts.num_samples, self.ts.num_samples))
        f = {st.tree_node: st for st in self.T}

        for j1, u1 in enumerate(self.ts.samples()):
            while u1 not in f:
                u1 = self.tree.parent(u1)
            f1 = {st.tree_node: st for st in f[u1].value_list}
            for j2, u2 in enumerate(self.ts.samples()):
                while u2 not in f1:
                    u2 = self.tree.parent(u2)
                A[j1, j2] = f1[u2].value
                B[j1, j2] = f1[u2].inner_summation
        return A, B

    def check_integrity(self):
        M = [st.tree_node for st in self.T if st.tree_node != -1]
        assert np.all(self.T_index[M] >= 0)
        index = np.ones_like(self.T_index, dtype=bool)
        index[M] = 0
        assert np.all(self.T_index[index] == -1)
        for j, st in enumerate(self.T):
            if st.tree_node != -1:
                assert j == self.T_index[st.tree_node]

    def stupid_compress_dict(self):
        # Duncan to create a stupid compression that just runs parsimony so is guaranteed to work.
        tree = self.tree
        T = self.T
        alleles = np.zeros((tree.num_samples(), tree.num_samples()))
        alleles_string_vec = np.zeros(tree.num_samples()).astype("object")
        genotypes = np.zeros(tree.num_samples(), dtype=int)
        genotype_index = 0
        mapping_back = {}

        node_map = {st.tree_node: st for st in self.T}

        for st1 in T:
            if st1.tree_node != -1:
                alleles_string_tmp = [
                    f"{st2.tree_node}:{st2.value:.16f}" for st2 in st1.value_list
                ]
                alleles_string = ",".join(alleles_string_tmp)
                # Add an extra element that tells me the alleles_string there.
                st1.alleles_string = alleles_string
                st1.genotype_index = genotype_index
                # assert alleles_string not in mapping_back
                if alleles_string not in mapping_back:
                    mapping_back[alleles_string] = {
                        "tree_node": st1.tree_node,
                        "value_list": st1.value_list,
                        "inner_summation": st1.inner_summation,
                    }
                genotype_index += 1

        for leaf in tree.samples():
            u = leaf
            while u not in node_map:
                u = tree.parent(u)
            genotypes[leaf] = node_map[u].genotype_index

        alleles_string_vec = []
        for st in T:
            if st.tree_node != -1:
                alleles_string_vec.append(st.alleles_string)

        ancestral_allele, mutations = tree.map_mutations(genotypes, alleles_string_vec)

        # Retain the old T_index, because the internal T that's passed up the tree will retain this ordering.
        old_T_index = copy.deepcopy(self.T_index)
        self.T_index = np.zeros(tree.num_nodes, dtype=int) - 1
        self.N = np.zeros(tree.num_nodes, dtype=int)
        self.T.clear()

        # First, create T root.
        self.T_index[tree.root] = 0
        self.T.append(
            ValueTransition(
                tree_node=tree.root,
                value_list=[
                    InternalValueTransition(
                        tree_node=tree.root,
                        value=mapping_back[ancestral_allele]["value_list"][
                            old_T_index[mapping_back[ancestral_allele]["tree_node"]]
                        ].value,
                    )
                ],
            )
        )

        # Then create the rest of T, adding the root each time to value_list
        for i, mut in enumerate(mutations):
            self.T_index[mut.node] = i + 1
            self.T.append(
                ValueTransition(
                    tree_node=mut.node,
                    value_list=[
                        InternalValueTransition(
                            tree_node=tree.root,
                            value=mapping_back[mut.derived_state]["value_list"][
                                old_T_index[mapping_back[ancestral_allele]["tree_node"]]
                            ].value,
                        )
                    ],
                )
            )

        # First add to the root
        for mut in mutations:
            self.T[self.T_index[tree.root]].value_list.append(
                InternalValueTransition(
                    tree_node=mut.node,
                    value=mapping_back[ancestral_allele]["value_list"][
                        old_T_index[mapping_back[mut.derived_state]["tree_node"]]
                    ].value,
                )
            )

        # Then add the rest of T_internal to each internal T.
        for mut1 in mutations:
            for mut2 in mutations:
                self.T[self.T_index[mut1.node]].value_list.append(
                    InternalValueTransition(
                        tree_node=mut2.node,
                        value=mapping_back[mut1.derived_state]["value_list"][
                            old_T_index[mapping_back[mut2.derived_state]["tree_node"]]
                        ].value,
                    )
                )

        # General approach here is to use mapping_back[mut.derived_state]['value_list'][old_T_index[mapping_back[mut2.derived_state]["tree_node"]]
        # and append this to the T_inner.

        node_map = {st.tree_node: st for st in self.T}

        for u in tree.samples():
            while u not in node_map:
                u = tree.parent(u)
            self.N[self.T_index[u]] += 1

    def update_tree(self):
        """
        Update the internal data structures to move on to the next tree.
        """
        parent = self.parent
        T_index = self.T_index
        T = self.T
        _, edges_out, edges_in = next(self.edge_diffs)

        for edge in edges_out:
            u = edge.child
            if T_index[u] == -1:
                # Make sure the subtree we're detaching has an T_index-value at the root.
                while T_index[u] == -1:
                    u = parent[u]
                    assert u != -1
                T_index[edge.child] = len(T)
                T.append(
                    ValueTransition(
                        tree_node=edge.child,
                        value_list=copy.deepcopy(T[T_index[u]].value_list),
                    )
                )
                # Add on this extra node to each of the internal lists
                for st in T:
                    if not (st.value_list == tskit.NULL):
                        st.value_list.append(
                            InternalValueTransition(
                                tree_node=edge.child,
                                value=st.value_list.copy()[T_index[u]].value,
                            )
                        )
            parent[edge.child] = -1

        for edge in edges_in:
            parent[edge.child] = edge.parent
            u = edge.parent
            if parent[edge.parent] == -1:
                # Grafting onto a new root.
                if T_index[edge.parent] == -1:
                    T_index[edge.parent] = len(T)
                    T.append(
                        ValueTransition(
                            tree_node=edge.parent,
                            value_list=copy.deepcopy(T[T_index[edge.child]].value_list),
                        )
                    )
                    # Add on this extra node to each of the internal lists
                    for st in T:
                        if not (st.value_list == tskit.NULL):
                            st.value_list.append(
                                InternalValueTransition(
                                    tree_node=edge.parent,
                                    value=st.value_list.copy()[
                                        T_index[edge.child]
                                    ].value,
                                )
                            )
            else:
                # Grafting into an existing subtree.
                while T_index[u] == -1:
                    u = parent[u]
                    assert u != -1
            assert T_index[u] != -1 and T_index[edge.child] != -1
            if (
                T[T_index[u]].value_list == T[T_index[edge.child]].value_list
            ):  # DEV: is this fine?
                st = T[T_index[edge.child]]
                # Mark the lower ValueTransition as unused.
                st.value_list = -1
                # Also need to mark the corresponding InternalValueTransition as unused for the remaining states
                for st2 in T:
                    if not (st2.value_list == tskit.NULL):
                        st2.value_list[T_index[edge.child]].value = -1
                        st2.value_list[T_index[edge.child]].tree_node = -1

                st.tree_node = -1
                T_index[edge.child] = -1

        # We can have values left over still pointing to old roots. Remove
        for root in self.tree.roots:
            if T_index[root] != -1:
                # Use a special marker here to designate the real roots.
                T[T_index[root]].value_index = -2

        for vt in T:
            if vt.tree_node != -1:
                if parent[vt.tree_node] == -1 and vt.value_index != -2:
                    # Also need to mark the corresponding InternalValueTransition as unused for the remaining states
                    for st2 in T:
                        if not (st2.value_list == tskit.NULL):
                            st2.value_list[T_index[vt.tree_node]].value = -1
                            st2.value_list[T_index[vt.tree_node]].tree_node = -1
                    T_index[vt.tree_node] = -1
                    vt.tree_node = -1
                vt.value_index = -1

        self.N = np.zeros(self.tree.num_nodes, dtype=int)
        node_map = {st.tree_node: st for st in self.T}

        for u in self.tree.samples():
            while u not in node_map:
                u = self.tree.parent(u)
            self.N[self.T_index[u]] += 1

    def update_probabilities(self, site, genotype_state):
        tree = self.tree
        T_index = self.T_index
        T = self.T
        alleles = ["0", "1"]
        allelic_state = self.allelic_state
        # Set the allelic_state for this site.
        allelic_state[self.tree.root] = alleles.index(site.ancestral_state)
        normalisation_factor_inner = {}

        for st1 in T:
            if st1.tree_node != -1:
                normalisation_factor_inner[
                    st1.tree_node
                ] = self.compute_normalisation_factor_inner_dict(st1.tree_node)

        for st1 in T:
            if st1.tree_node != -1:
                for st2 in st1.value_list:
                    if st2.tree_node != -1:
                        self.T[self.T_index[st1.tree_node]].value_list[
                            self.T_index[st2.tree_node]
                        ].inner_summation = max(
                            normalisation_factor_inner[st1.tree_node],
                            normalisation_factor_inner[st2.tree_node],
                        )

        for mutation in site.mutations:
            u = mutation.node
            allelic_state[u] = alleles.index(mutation.derived_state)
            if T_index[u] == -1:
                while T_index[u] == tskit.NULL:
                    u = tree.parent(u)
                T_index[mutation.node] = len(T)
                T.append(
                    ValueTransition(
                        tree_node=mutation.node,
                        value_list=copy.deepcopy(T[T_index[u]].value_list),
                    )  # DEV: is it possible to not use deepcopies?
                )
                for st in T:
                    if not (st.value_list == tskit.NULL):
                        st.value_list.append(
                            InternalValueTransition(
                                tree_node=mutation.node,
                                value=st.value_list.copy()[T_index[u]].value,
                                inner_summation=st.value_list.copy()[
                                    T_index[u]
                                ].inner_summation,
                            )
                        )

        # Get the allelic state at the leaves.
        allelic_state[: tree.num_samples()] = tree.tree_sequence.genotype_matrix()[
            site.id, :
        ]

        query_is_het = genotype_state == 1

        for st1 in T:
            u1 = st1.tree_node

            if u1 != -1:
                # Get the allelic_state at u. TODO we can cache these states to
                # avoid some upward traversals.
                v1 = u1
                while allelic_state[v1] == -1:
                    v1 = tree.parent(v1)
                    assert v1 != -1

                for st2 in st1.value_list:
                    u2 = st2.tree_node
                    if u2 != -1:
                        # Get the allelic_state at u. TODO we can cache these states to
                        # avoid some upward traversals.
                        v2 = u2
                        while allelic_state[v2] == -1:
                            v2 = tree.parent(v2)
                            assert v2 != -1

                        genotype_template_state = allelic_state[v1] + allelic_state[v2]
                        match = genotype_state == genotype_template_state
                        template_is_het = genotype_template_state == 1
                        # Fill in the value at the combination of states: (s1, s2)
                        st2.value = self.compute_next_probability_dict(
                            site.id,
                            st2.value,
                            st2.inner_summation,
                            match,
                            template_is_het,
                            query_is_het,
                            u2,
                        )

                # This will ensure that allelic_state[:n] is filled
                genotype_template_state = (
                    allelic_state[v1] + allelic_state[: tree.num_samples()]
                )
                # These are vectors of length n (at internal nodes).
                match = genotype_state == genotype_template_state
                template_is_het = genotype_template_state == 1

        # Unset the states
        allelic_state[tree.root] = -1
        for mutation in site.mutations:
            allelic_state[mutation.node] = -1

    def process_site(self, site, genotype_state):
        self.update_probabilities(site, genotype_state)
        self.stupid_compress_dict()
        s1 = self.compute_normalisation_factor_dict()
        T = self.T

        for st in T:
            if st.tree_node != tskit.NULL:
                # Need to loop through value copy, and normalise
                for st2 in st.value_list:
                    st2.value /= s1
                    st2.value = np.round(st2.value, self.precision)

        self.output.store_site(
            site.id, s1, [(st.tree_node, st.value_list) for st in self.T]
        )

    def run_viterbi(self, g):
        n = self.ts.num_samples
        self.tree.clear()
        for u in self.ts.samples():
            self.T_index[u] = len(self.T)
            self.T.append(ValueTransition(tree_node=u, value_list=[]))
            for v in self.ts.samples():
                self.T[self.T_index[u]].value_list.append(
                    InternalValueTransition(tree_node=v, value=(1 / n) ** 2)
                )

        while self.tree.next():
            self.update_tree()
            for site in self.tree.sites():
                self.process_site(site, g[site.id])

        return self.output

    def compute_normalisation_factor_dict(self):
        raise NotImplementedError()

    def compute_next_probability_dict(
        self,
        site_id,
        p_last,
        inner_summation,
        is_match,
        template_is_het,
        query_is_het,
        node,
    ):
        raise NotImplementedError()


class CompressedMatrix:
    """
    Class representing a num_samples x num_sites matrix compressed by a
    tree sequence. Each site is represented by a set of (node, value)
    pairs, which act as "mutations", i.e., any sample that descends
    from a particular node will inherit that value (unless any other
    values are on the path).
    """

    def __init__(self, ts, normalisation_factor=None):
        self.ts = ts
        self.num_sites = ts.num_sites
        self.num_samples = ts.num_samples
        self.value_transitions = [None for _ in range(self.num_sites)]
        if normalisation_factor is None:
            self.normalisation_factor = np.zeros(self.num_sites)
        else:
            self.normalisation_factor = normalisation_factor
            assert len(self.normalisation_factor) == self.num_sites

    def store_site(self, site, normalisation_factor, value_transitions):
        self.normalisation_factor[site] = normalisation_factor
        self.value_transitions[site] = copy.deepcopy(value_transitions)

    # Expose the same API as the low-level classes

    @property
    def num_transitions(self):
        a = [len(self.value_transitions[j]) for j in range(self.num_sites)]
        return np.array(a, dtype=np.int32)

    def get_site(self, site):
        return self.value_transitions[site]

    def decode_site_dict(self, tree, site_id):
        """
        Decodes the tree encoding of the values into an explicit
        matrix.
        """
        A = np.zeros((self.num_samples, self.num_samples))
        f = dict(self.value_transitions[site_id])

        for j1, u1 in enumerate(self.ts.samples()):
            while u1 not in f:
                u1 = tree.parent(u1)
            f1 = {st.tree_node: st for st in f[u1]}
            for j2, u2 in enumerate(self.ts.samples()):
                while u2 not in f1:
                    u2 = tree.parent(u2)
                A[j1, j2] = f1[u2].value
        return A

    def decode(self):
        """
        Decodes the tree encoding of the values into an explicit
        matrix.
        """
        A = np.zeros((self.num_sites, self.num_samples, self.num_samples))
        for tree in self.ts.trees():
            for site in tree.sites():
                A[site.id] = self.decode_site_dict(tree, site.id)
        return A


class ViterbiMatrix(CompressedMatrix):
    """Class representing the compressed Viterbi matrix."""


class ViterbiAlgorithm(LsHmmAlgorithm):
    """
    Runs the Li and Stephens Viterbi algorithm.
    """

    def __init__(self, ts, rho, mu, precision=10):
        super().__init__(ts, rho, mu, precision)
        self.output = ViterbiMatrix(ts)

    def compute_normalisation_factor_dict(self):
        s = 0
        for st in self.T:
            assert st.tree_node != tskit.NULL
            max_st = self.compute_normalisation_factor_inner_dict(st.tree_node)
            if max_st > s:
                s = max_st
        if s == 0:
            raise ValueError(
                "Trying to match non-existent allele with zero mutation rate"
            )
        return s

    def compute_normalisation_factor_inner_dict(self, node):
        s_inner = 0
        V_previous = self.T[self.T_index[node]].value_list
        for st in V_previous:
            j = st.tree_node
            if j != -1:
                max_st = st.value
                if max_st > s_inner:
                    s_inner = max_st

        return s_inner

    def compute_next_probability_dict(
        self,
        site_id,
        p_last,
        inner_normalisation_factor,
        is_match,
        template_is_het,
        query_is_het,
        node,
    ):
        r = self.rho[site_id]
        mu = self.mu[site_id]
        n = self.ts.num_samples
        r_n = r / n

        template_is_hom = np.logical_not(template_is_het)
        query_is_hom = np.logical_not(query_is_het)

        EQUAL_BOTH_HOM = np.logical_and(
            np.logical_and(is_match, template_is_hom), query_is_hom
        )
        UNEQUAL_BOTH_HOM = np.logical_and(
            np.logical_and(np.logical_not(is_match), template_is_hom), query_is_hom
        )
        BOTH_HET = np.logical_and(template_is_het, query_is_het)
        REF_HOM_OBS_HET = np.logical_and(template_is_hom, query_is_het)
        REF_HET_OBS_HOM = np.logical_and(template_is_het, query_is_hom)

        p_e = (
            EQUAL_BOTH_HOM * (1 - mu) ** 2
            + UNEQUAL_BOTH_HOM * (mu ** 2)
            + REF_HOM_OBS_HET * (2 * mu * (1 - mu))
            + REF_HET_OBS_HOM * (mu * (1 - mu))
            + BOTH_HET * ((1 - mu) ** 2 + mu ** 2)
        )

        no_switch = (1 - r) ** 2 + 2 * (r_n * (1 - r)) + r_n ** 2
        single_switch = r_n * (1 - r) + r_n ** 2
        double_switch = r_n ** 2

        V_single_switch = inner_normalisation_factor
        p_t = p_last * no_switch
        single_switch_tmp = single_switch * V_single_switch

        if single_switch_tmp > double_switch:
            # Then single switch is the alternative
            if p_t < single_switch * V_single_switch:
                p_t = single_switch * V_single_switch
        else:
            # Double switch is the alternative
            if p_t < double_switch:
                p_t = double_switch

        return p_t * p_e


def ls_viterbi_tree(g, ts, rho, mu, precision=30):
    """
    Viterbi path computation based on a tree sequence.
    """
    va = ViterbiAlgorithm(ts, rho, mu, precision=precision)
    return va.run_viterbi(g)