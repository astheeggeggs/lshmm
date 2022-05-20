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

    def __init__(self, tree_node=-1, inner_summation=-1, value=-1, value_index=-1):
        self.tree_node = tree_node
        self.value = value
        self.inner_summation = inner_summation
        self.value_index = value_index

    def copy(self):
        return ValueTransition(
            self.tree_node, self.inner_summation, self.value, self.value_index
        )

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

    def decode_site(self):
        """
        Decodes the tree encoding of the values into an explicit
        matrix.
        """
        A = np.zeros((self.ts.num_samples, self.ts.num_samples))
        # To look at the inner summations too.
        B = np.zeros((self.ts.num_samples, self.ts.num_samples))
        f = {st.tree_node: st for st in self.T}
        for j, u in enumerate(self.ts.samples()):
            while u not in f:
                u = self.tree.parent(u)
            A[j, :] = f[u].value
            B[j, :] = f[u].inner_summation
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

    def stupid_compress(self):
        # Duncan to create a stupid compression that just runs parsimony so is guaranteed to work.
        tree = self.tree
        T = self.T
        alleles = np.zeros((tree.num_samples(), tree.num_samples()))
        alleles_string_vec = np.zeros(tree.num_samples()).astype("object")
        genotypes = np.zeros(tree.num_samples(), dtype=int)
        genotype_index = 0
        mapping_back = {}

        node_map = {st.tree_node: st for st in self.T}

        for st in T:
            if st.tree_node != -1:
                alleles_string_tmp = [f"{x:.16f}" for x in st.value]
                alleles_string = ",".join(alleles_string_tmp)
                # Add an extra element that tells me the alleles_string there.
                st.alleles_string = alleles_string
                st.genotype_index = genotype_index
                # if alleles_string in mapping_back:
                # Making sure that if you do overwrite, it's with the same thing (we can just change this to an if statement)
                # if not (np.allclose(mapping_back[alleles_string].get("value"), st.value, rtol=1e-9, atol=0.0) and np.allclose(mapping_back[alleles_string].get("inner_summation"), st.inner_summation, rtol=1e-9, atol=0.0)):
                #     print(mapping_back[alleles_string].get("value"))
                #     print(st.value)
                #     print(mapping_back[alleles_string].get("inner_summation"))
                #     print(st.inner_summation)
                # assert np.allclose(mapping_back[alleles_string].get("value"), st.value, rtol=1e-9, atol=0.0)
                # assert np.allclose(mapping_back[alleles_string].get("inner_summation"), st.inner_summation, rtol=1e-9, atol=0.0)

                # assert alleles_string not in mapping_back
                if alleles_string not in mapping_back:
                    mapping_back[alleles_string] = {
                        "value": st.value,
                        "inner_summation": st.inner_summation,
                    }
                genotype_index += 1

        for leaf in tree.samples():
            u = leaf
            while u not in node_map:
                u = tree.parent(u)
            alleles[leaf, :] = node_map[u].value
            genotypes[leaf] = node_map[u].genotype_index

        alleles_string_vec = []
        for st in T:
            if st.tree_node != -1:
                alleles_string_vec.append(st.alleles_string)

        # print(genotypes)
        # print(alleles_string_vec)
        ancestral_allele, mutations = tree.map_mutations(genotypes, alleles_string_vec)

        self.T_index = np.zeros(tree.num_nodes, dtype=int) - 1
        self.N = np.zeros(tree.num_nodes, dtype=int)
        self.T.clear()
        self.T.append(
            ValueTransition(
                tree_node=tree.root,
                value=mapping_back[ancestral_allele]["value"],
                inner_summation=mapping_back[ancestral_allele]["inner_summation"],
            )
        )
        self.T_index[tree.root] = 0

        for i, mut in enumerate(mutations):
            self.T.append(
                ValueTransition(
                    tree_node=mut.node,
                    value=mapping_back[mut.derived_state]["value"].copy(),
                    inner_summation=mapping_back[mut.derived_state]["inner_summation"],
                )
            )
            self.T_index[mut.node] = i + 1

        node_map = {st.tree_node: st for st in self.T}

        for u in tree.samples():
            while u not in node_map:
                u = tree.parent(u)
            self.N[self.T_index[u]] += 1

    def compress(self):
        tree = self.tree
        T = self.T
        T_index = self.T_index

        values = np.unique(list(st.value if st.tree_node != -1 else 1e200 for st in T))
        for st in T:
            if st.tree_node != -1:
                st.value_index = np.searchsorted(values, st.value)

        child = np.zeros(len(values), dtype=int)
        num_values = len(values)
        value_count = np.zeros(num_values, dtype=int)

        def compute(u, parent_state):
            value_count[:] = 0
            for v in tree.children(u):
                child[:] = optimal_set[v]
                # If the set for a given child is empty, then we know it inherits
                # directly from the parent state and must be a singleton set.
                if np.sum(child) == 0:
                    child[parent_state] = 1
                for j in range(num_values):
                    value_count[j] += child[j]
            max_value_count = np.max(value_count)
            optimal_set[u, :] = 0
            optimal_set[u, value_count == max_value_count] = 1

        optimal_set = np.zeros((tree.tree_sequence.num_nodes, len(values)), dtype=int)
        t_node_time = [
            -1 if st.tree_node == -1 else tree.time(st.tree_node) for st in T
        ]
        order = np.argsort(t_node_time)
        for j in order:
            st = T[j]
            u = st.tree_node
            if u != -1:
                # Compute the value at this node
                state = st.value_index
                if tree.is_internal(u):
                    compute(u, state)
                else:
                    # A[u, state] = 1
                    optimal_set[u, state] = 1
                # Find parent state
                v = tree.parent(u)
                if v != -1:
                    while T_index[v] == -1:
                        v = tree.parent(v)
                    parent_state = T[T_index[v]].value_index
                    v = tree.parent(u)
                    while T_index[v] == -1:
                        compute(v, parent_state)
                        v = tree.parent(v)

        T_old = [st.copy() for st in T]
        T.clear()
        T_parent = []

        old_state = T_old[T_index[tree.root]].value_index
        new_state = np.argmax(optimal_set[tree.root])

        T.append(ValueTransition(tree_node=tree.root, value=values[new_state]))
        T_parent.append(-1)
        stack = [(tree.root, old_state, new_state, 0)]
        while len(stack) > 0:
            u, old_state, new_state, t_parent = stack.pop()
            for v in tree.children(u):
                old_child_state = old_state
                if T_index[v] != -1:
                    old_child_state = T_old[T_index[v]].value_index
                if np.sum(optimal_set[v]) > 0:
                    new_child_state = new_state
                    child_t_parent = t_parent

                    if optimal_set[v, new_state] == 0:
                        new_child_state = np.argmax(optimal_set[v])
                        child_t_parent = len(T)
                        T_parent.append(t_parent)
                        T.append(
                            ValueTransition(tree_node=v, value=values[new_child_state])
                        )
                    stack.append((v, old_child_state, new_child_state, child_t_parent))
                else:
                    if old_child_state != new_state:
                        T_parent.append(t_parent)
                        T.append(
                            ValueTransition(tree_node=v, value=values[old_child_state])
                        )

        for st in T_old:
            if st.tree_node != -1:
                T_index[st.tree_node] = -1
        for j, st in enumerate(T):
            T_index[st.tree_node] = j
            self.N[j] = tree.num_samples(st.tree_node)
        for j in range(len(T)):
            if T_parent[j] != -1:
                self.N[T_parent[j]] -= self.N[j]

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
                    ValueTransition(tree_node=edge.child, value=T[T_index[u]].value)
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
                            tree_node=edge.parent, value=T[T_index[edge.child]].value
                        )
                    )
            else:
                # Grafting into an existing subtree.
                while T_index[u] == -1:
                    u = parent[u]
                    assert u != -1
            assert T_index[u] != -1 and T_index[edge.child] != -1
            if (T[T_index[u]].value == T[T_index[edge.child]].value).all():
                st = T[T_index[edge.child]]
                # Mark the lower ValueTransition as unused.
                st.value = -1
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
                    T_index[vt.tree_node] = -1
                    vt.tree_node = -1
                vt.value_index = -1

    def update_probabilities(self, site, genotype_state):
        tree = self.tree
        T_index = self.T_index
        T = self.T
        alleles = ["0", "1"]
        allelic_state = self.allelic_state
        # Set the allelic_state for this site.
        allelic_state[tree.root] = alleles.index(site.ancestral_state)
        # print("root state")
        # print(allelic_state[tree.root])

        # if site.id == 13:
        #     print(site.mutations)
        #     print(self.T)
        #     print(tree.draw_text())
        #     print(self.ts.draw_text())
        print(site.id)
        if site.id == 5:
            assert 0 == 1
        print(tree.draw_text())

        # blah = self.decode_site()
        for mutation in site.mutations:
            u = mutation.node
            allelic_state[u] = alleles.index(mutation.derived_state)
            if T_index[u] == -1:
                while T_index[u] == tskit.NULL:
                    u = tree.parent(u)
                T_index[mutation.node] = len(T)
                T.append(
                    ValueTransition(
                        tree_node=mutation.node, value=T[T_index[u]].value.copy()
                    )
                )
            # print("difference: ")
            # print(blah - self.decode_site())
            # Does A decode correctly here? (as mutations are added)

        # Extra bit that should be removed eventually - this is to get the allelic state at the leaves.
        allelic_state[: tree.num_samples()] = tree.tree_sequence.genotype_matrix()[
            site.id, :
        ]
        # print("allelic state")
        # print(allelic_state)
        node_map = {st.tree_node: st for st in self.T}
        to_compute = np.zeros(
            tree.num_samples(), dtype=int
        )  # Because the node ordering in the leaves is not 0 -> n_samples - 1.

        # for v in tree.samples():
        for v in self.ts.samples():
            v_tmp = v
            while v not in node_map:
                v = tree.parent(v)
            # print(f"leaf {v}->{v_tmp}")
            to_compute[v_tmp] = v

        # print(to_compute)
        print(self.N)
        print(self.T_index)
        normalisation_factor_inner_transpose = [
            self.compute_normalisation_factor_inner(i) for i in to_compute
        ]
        # if site.id == 13:
        #     print("normalisation_factor_inner_transpose")
        #     print(normalisation_factor_inner_transpose)
        #     print(to_compute)
        #     print(self.T)
        # print(self.mu)
        # print(f'genotype_state: {genotype_state}')
        # print(f'EQUAL_BOTH_HOM: {(1 - self.mu[site.id])**2}')
        # print(f'UNEQUAL_BOTH_HOM: {self.mu[site.id]**2}')
        # print(f'BOTH_HET: {((1-self.mu[site.id])**2 + self.mu[site.id]**2)}')
        # print(f'REF_HOM_OBS_HET: {(2*self.mu[site.id]*(1 - self.mu[site.id]))}')
        # print(f'REF_HET_OBS_HOM: {(self.mu[site.id]*(1-self.mu[site.id]))}')

        genotype_template_state = np.add.outer(
            allelic_state[: tree.num_samples()], allelic_state[: tree.num_samples()]
        )
        # These are vectors of length n (at internal nodes).
        match = genotype_state == genotype_template_state
        # template_is_hom = np.logical_or(genotype_template_state == 0, genotype_template_state == 2)
        template_is_het = genotype_template_state == 1
        # query_is_hom = np.logical_or(genotype_state == 0, genotype_state == 2)
        query_is_het = genotype_state == 1

        index = (
            4 * match.astype(np.int64)
            + 2 * template_is_het.astype(np.int64)
            + int(query_is_het)
        )
        # print(index)
        # print(match)
        # print(template_is_het)
        # print(query_is_het)

        for j, st in enumerate(T):
            u = st.tree_node
            # print(f"tree_node: {st.tree_node}")
            st.inner_summation = (
                self.compute_normalisation_factor_inner(u)
                + normalisation_factor_inner_transpose
            )
            # if site.id == 13 and st.tree_node == 14:
            print("inner_summation")
            print(st.inner_summation)
            # if site.id != 0:
            # assert u != -1
            if u != -1:
                # Get the allelic_state at u. TODO we can cache these states to
                # avoid some upward traversals.
                v = u
                while allelic_state[v] == -1:
                    v = tree.parent(v)
                    assert v != -1

                # This will ensure that allelic_state[:n] is filled
                genotype_template_state = (
                    allelic_state[v] + allelic_state[: tree.num_samples()]
                )
                # These are vectors of length n (at internal nodes).
                match = genotype_state == genotype_template_state
                # template_is_hom = np.logical_or(genotype_template_state == 0, genotype_template_state == 2)
                template_is_het = genotype_template_state == 1
                # query_is_hom = np.logical_or(genotype_state == 0, genotype_state == 2)
                query_is_hom = genotype_state == 1
                st.value = self.compute_next_probability(
                    site.id,
                    st.value,
                    st.inner_summation,
                    match,
                    template_is_het,
                    query_is_het,
                )  # template_is_hom, query_is_hom)
                # print("vector version")
                # print(st.value)
        # Unset the states
        allelic_state[tree.root] = -1
        for mutation in site.mutations:
            allelic_state[mutation.node] = -1

    # Define function that maps leaves to node inherited from.

    def process_site(self, site, genotype_state, forwards=True):

        self.update_probabilities(site, genotype_state)
        A = self.decode_site()
        # if site.id == 13:
        #     print("before")
        #     for st in self.T:
        #         print(f"\t{st}")

        self.stupid_compress()
        # if site.id == 13:
        #     # print("after")
        #     for st in self.T:
        #         print(f"\t{st}")
        #     for j, st in enumerate(self.T):
        #         u = st.tree_node
        #         if u != -1 and st.tree_node == 14:
        #             print(f'node: {u}')
        #             print(f'descendent leaves: {self.N[j]}')
        #             print(st.value)
        #             print(st.inner_summation)

        A_compress = self.decode_site()
        # assert np.allclose(A, A_compress, rtol=1e-9, atol=0.0)
        s = self.compute_normalisation_factor()
        # if not np.allclose(self.A[site.id, :, :] * self.c[site.id], A):
        #     print(site.id)
        #     print(self.A[site.id, :, :] * self.c[site.id] - A)
        #     print(f"normalisation_factor: {s}")
        #     print(f"normalisation_factor matrix: {self.c[site.id]}")
        #     print("N below")
        #     print(self.N)
        # print("normalise")
        # print(np.around(s, 10))
        # assert isclose(self.c[site.id], s)
        # assert np.allclose(self.A[site.id, :, :] * self.c[site.id], A)

        T = self.T
        # print("before normalisation")
        # print(self.T)
        for st in T:
            if st.tree_node != tskit.NULL:
                # print(f"tree node: {st.tree_node}")
                # print(f"value: {st.value}")
                # st.old_value = copy.deepcopy(st.value)
                # st.value = copy.deepcopy(st.value)
                st.value /= s
                st.value = np.round(st.value, self.precision)

        # print(self.T)
        self.output.store_site(site.id, s, [(st.tree_node, st.value) for st in self.T])

    def run_forward(self, g):
        n = self.ts.num_samples
        self.tree.clear()
        for u in self.ts.samples():
            self.T_index[u] = len(self.T)
            self.T.append(
                ValueTransition(tree_node=u, value=((1 / n) ** 2) * np.ones(n))
            )
        while self.tree.next():
            self.update_tree()
            for site in self.tree.sites():
                # print(f"site {site.id}")
                self.process_site(site, g[site.id])
                F_check, inner_summation_check = self.decode_site()
                # if site.id != 0:
                #     print("inner summation")
                #     print(np.around(inner_summation_check, 10))
                # print("F unnormalised")
                # print(np.around(F_check * self.output.normalisation_factor[site.id], 10))
                # print("F")
                # print(np.around(F_check, 10))
        return self.output

    def compute_normalisation_factor(self):
        raise NotImplementedError()

    def compute_next_probability(
        self, site_id, p_last, inner_summation, is_match, template_is_het, query_is_het
    ):  # template_is_hom, query_is_hom):
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
        self.value_transitions[site] = value_transitions

    # Expose the same API as the low-level classes

    @property
    def num_transitions(self):
        a = [len(self.value_transitions[j]) for j in range(self.num_sites)]
        return np.array(a, dtype=np.int32)

    def get_site(self, site):
        return self.value_transitions[site]

    def decode_site(self, tree, site_id):
        """
        Decodes the tree encoding of the values into an explicit
        matrix.
        """
        A = np.zeros((self.num_samples, self.num_samples))
        f = dict(self.value_transitions[site_id])
        for j, u in enumerate(self.ts.samples()):
            while u not in f:
                u = tree.parent(u)
            A[j, :] = f[u]
        return A

    def decode(self):
        """
        Decodes the tree encoding of the values into an explicit
        matrix.
        """
        A = np.zeros((self.num_sites, self.num_samples, self.num_samples))
        for tree in self.ts.trees():
            for site in tree.sites():
                A[site.id] = self.decode_site(tree, site.id)
        return A

    # def decode(self):
    #     """
    #     Decodes the tree encoding of the values into an explicit
    #     matrix.
    #     """
    #     A = np.zeros((self.num_sites, self.num_samples, self.num_samples))
    #     for tree in self.ts.trees():
    #         for site in tree.sites():
    #             f = dict(self.value_transitions[site.id])
    #             for j, u in enumerate(self.ts.samples()):
    #                 while u not in f:
    #                     u = tree.parent(u)
    #                 A[site.id, j, :] = f[u]
    #     return A


class ForwardMatrix(CompressedMatrix):
    """Class representing a compressed forward matrix."""


class ForwardAlgorithm(LsHmmAlgorithm):
    """Runs the Li and Stephens forward algorithm."""

    def __init__(self, ts, rho, mu, A, c, precision=30):
        super().__init__(ts, rho, mu, precision)
        self.output = ForwardMatrix(ts)
        # Debugging
        self.A = A
        self.c = c

    def compute_normalisation_factor(self):
        s = 0
        for j, st in enumerate(self.T):
            assert st.tree_node != tskit.NULL
            assert self.N[j] > 0
            s += self.N[j] * self.compute_normalisation_factor_inner(
                st.tree_node
            )  # st.inner_summation
        return s

    def compute_normalisation_factor_inner(self, node):
        return np.sum(self.T[self.T_index[node]].value)

    def compute_next_probability(
        self,
        site_id,
        p_last,
        inner_normalisation_factor,
        is_match,
        template_is_het,
        query_is_het,
    ):  # template_is_hom, query_is_hom):
        rho = self.rho[site_id]
        mu = self.mu[site_id]
        n = self.ts.num_samples

        # if site_id == 13:
        #     print(rho)
        #     print(mu)
        #     print(n)

        # EQUAL_BOTH_HOM = np.logical_and(np.logical_and(is_match, template_is_hom), query_is_hom)
        # UNEQUAL_BOTH_HOM = np.logical_and(np.logical_and(np.logical_not(is_match), template_is_hom), query_is_hom)
        # BOTH_HET = np.logical_and(np.logical_not(template_is_hom), np.logical_not(query_is_hom))
        # REF_HOM_OBS_HET = np.logical_and(template_is_hom, np.logical_not(query_is_hom))
        # REF_HET_OBS_HOM =  np.logical_and(np.logical_not(template_is_hom), query_is_hom)

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

        # if site_id == 13:
        #     if np.any(EQUAL_BOTH_HOM):
        # print(f'EQUAL_BOTH_HOM: {(1 - mu)**2}')
        # #     if np.any(UNEQUAL_BOTH_HOM):
        # print(f'UNEQUAL_BOTH_HOM: {mu**2}')
        # #     if np.any(BOTH_HET):
        # print(f'BOTH_HET: {((1-mu)**2 + mu**2)}')
        # #     if np.any(REF_HOM_OBS_HET):
        # print(f'REF_HOM_OBS_HET: {(2*mu*(1 - mu))}')
        # #     if np.any(REF_HET_OBS_HOM):
        # print(f'REF_HET_OBS_HOM: {(mu*(1-mu))}')

        p_t = (
            (rho / n) ** 2
            + ((1 - rho) * (rho / n)) * inner_normalisation_factor
            + (1 - rho) ** 2 * p_last
        )
        # if site_id == 13:
        #     print("before multiply by emission")
        #     print(p_t)
        p_e = (
            EQUAL_BOTH_HOM * (1 - mu) ** 2
            + UNEQUAL_BOTH_HOM * (mu ** 2)
            + REF_HOM_OBS_HET * (2 * mu * (1 - mu))
            + REF_HET_OBS_HOM * (mu * (1 - mu))
            + BOTH_HET * ((1 - mu) ** 2 + mu ** 2)
        )
        # if site_id == 13:
        #     print("emission")
        #     print(p_e)
        #     print("after multiply by emission")
        #     print(p_t * p_e)
        return p_t * p_e


def ls_forward_tree(g, ts, rho, mu, A, c, precision=30):
    """Forward matrix computation based on a tree sequence."""
    fa = ForwardAlgorithm(ts, rho, mu, A, c, precision=precision)
    return fa.run_forward(g)
