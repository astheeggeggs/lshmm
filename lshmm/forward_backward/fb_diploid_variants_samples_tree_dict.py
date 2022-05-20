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

    def __init__(
        self, tree_node=-1, inner_summation=-1, value=-1, value_copy=-1, value_index=-1
    ):
        self.tree_node = tree_node
        self.value = value
        self.value_copy = value_copy
        self.inner_summation = inner_summation
        self.value_index = value_index

    def copy(self):
        return ValueTransition(
            self.tree_node,
            self.inner_summation,
            self.value,
            self.value_copy,
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

    def decode_site(self):
        """
        Decodes the tree encoding of the values into an explicit
        matrix.
        """
        A = np.zeros((self.ts.num_samples, self.ts.num_samples))
        # To look at the inner summations too.
        B = np.zeros((self.ts.num_samples, self.ts.num_samples))
        f = {st.tree_node: st for st in self.T}
        # print(f.keys())
        # print(f)
        for j, u in enumerate(self.ts.samples()):
            while u not in f:
                u = self.tree.parent(u)
            # print(u)
            A[j, :] = f[u].value
            B[j, :] = f[u].inner_summation
        return A, B

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
            f1 = {st.tree_node: st for st in f[u1].value_copy}
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
                    f"{st2.tree_node}:{st2.value:.16f}" for st2 in st1.value_copy
                ]
                alleles_string = ",".join(alleles_string_tmp)
                # Add an extra element that tells me the alleles_string there.
                st1.alleles_string = alleles_string
                st1.genotype_index = genotype_index
                # print(f'alleles_string: {alleles_string}')
                # if alleles_string in mapping_back:
                # Making sure that if you do overwrite, it's with the same thing (we can just change this to an if statement)
                # if not (np.allclose(mapping_back[alleles_string].get("value"), st1.value, rtol=1e-9, atol=0.0) and np.allclose(mapping_back[alleles_string].get("inner_summation"), st1.inner_summation, rtol=1e-9, atol=0.0)):
                # print("something doesn't match!")
                # print(mapping_back[alleles_string].get("value"))
                # print(st1.value)
                # print(mapping_back[alleles_string].get("inner_summation"))
                # print(st1.inner_summation)
                # assert np.allclose(mapping_back[alleles_string].get("value"), st1.value, rtol=1e-9, atol=0.0)
                # assert np.allclose(mapping_back[alleles_string].get("inner_summation"), st1.inner_summation, rtol=1e-9, atol=0.0)

                # assert alleles_string not in mapping_back
                if alleles_string not in mapping_back:
                    mapping_back[alleles_string] = {
                        "tree_node": st1.tree_node,
                        "value": st1.value,
                        "value_copy": st1.value_copy,
                        "inner_summation": st1.inner_summation,
                    }
                genotype_index += 1
                # print(mapping_back)

        for leaf in tree.samples():
            u = leaf
            while u not in node_map:
                u = tree.parent(u)
            genotypes[leaf] = node_map[u].genotype_index

        alleles_string_vec = []
        for st in T:
            if st.tree_node != -1:
                alleles_string_vec.append(st.alleles_string)

        # print(genotypes)
        # print(alleles_string_vec)
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
                value=mapping_back[ancestral_allele]["value"],
                # DEV:
                # inner_summation=mapping_back[ancestral_allele]['inner_summation'],
                value_copy=[
                    InternalValueTransition(
                        tree_node=tree.root,
                        # value=mapping_back[ancestral_allele]['value_copy'][self.T_index[tree.root]].value,
                        value=mapping_back[ancestral_allele]["value_copy"][
                            old_T_index[mapping_back[ancestral_allele]["tree_node"]]
                        ].value,
                        # inner_summation=mapping_back[ancestral_allele]['value_copy'][self.T_index[tree.root]].inner_summation,
                        # DEV:
                        # inner_summation=mapping_back[ancestral_allele]['value_copy'][old_T_index[mapping_back[ancestral_allele]['tree_node']]].inner_summation
                    )
                ],
            )
        )

        # Then create the rest of T, adding the root each time to value_copy
        for i, mut in enumerate(mutations):
            self.T_index[mut.node] = i + 1
            self.T.append(
                ValueTransition(
                    tree_node=mut.node,
                    value=mapping_back[mut.derived_state]["value"].copy(),
                    # DEV:
                    # inner_summation = mapping_back[mut.derived_state]['inner_summation'],
                    value_copy=[
                        InternalValueTransition(
                            tree_node=tree.root,
                            # value=mapping_back[mut.derived_state]['value_copy'][self.T_index[tree.root]].value,
                            value=mapping_back[mut.derived_state]["value_copy"][
                                old_T_index[mapping_back[ancestral_allele]["tree_node"]]
                            ].value,
                            # inner_summation=mapping_back[mut.derived_state]['value_copy'][self.T_index[tree.root]].inner_summation,
                            # DEV:
                            # inner_summation=mapping_back[mut.derived_state]['value_copy'][old_T_index[mapping_back[ancestral_allele]['tree_node']]].inner_summation
                        )
                    ],
                )
            )

        # First add to the root
        for mut in mutations:
            # print(self.T_index[mapping_back[mut.derived_state]['tree_node']])
            # print(old_T_index[mapping_back[mut.derived_state]['tree_node']])
            self.T[self.T_index[tree.root]].value_copy.append(
                InternalValueTransition(
                    tree_node=mut.node,
                    # value=mapping_back[ancestral_allele]['value_copy'][self.T_index[mut.node]].value,
                    value=mapping_back[ancestral_allele]["value_copy"][
                        old_T_index[mapping_back[mut.derived_state]["tree_node"]]
                    ].value,
                    # inner_summation=mapping_back[ancestral_allele]['value_copy'][self.T_index[mut.node]].inner_summation,
                    # DEV:
                    # inner_summation=mapping_back[ancestral_allele]['value_copy'][old_T_index[mapping_back[mut.derived_state]['tree_node']]].inner_summation
                )
            )

        # Then add the rest of T_internal to each internal T.
        for mut1 in mutations:
            for mut2 in mutations:
                self.T[self.T_index[mut1.node]].value_copy.append(
                    InternalValueTransition(
                        tree_node=mut2.node,
                        # value=mapping_back[mut1.derived_state]['value_copy'][self.T_index[mut2.node]].value
                        value=mapping_back[mut1.derived_state]["value_copy"][
                            old_T_index[mapping_back[mut2.derived_state]["tree_node"]]
                        ].value,
                        # inner_summation=mapping_back[mut1.derived_state]['value_copy'][self.T_index[mut2.node]].inner_summation,
                        # DEV: - Include these below all the DEVs if you want to pass the inner summations around.
                        # inner_summation=mapping_back[mut1.derived_state]['value_copy'][old_T_index[mapping_back[mut2.derived_state]['tree_node']]].inner_summation
                    )
                )

        # General approach here is to use mapping_back[mut.derived_state]['value_copy'][self.T_index[mut.node].value]
        # (and similarly for inner summation), and append this to the T_inner.

        node_map = {st.tree_node: st for st in self.T}

        for u in tree.samples():
            while u not in node_map:
                u = tree.parent(u)
            self.N[self.T_index[u]] += 1

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
                # print(mapping_back[alleles_string].get("value"))
                # print(st.value)
                # print(mapping_back[alleles_string].get("inner_summation"))
                # print(st.inner_summation)
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
                # DEV:
                # inner_summation=mapping_back[ancestral_allele]['inner_summation']
            )
        )
        self.T_index[tree.root] = 0

        for i, mut in enumerate(mutations):
            self.T.append(
                ValueTransition(
                    tree_node=mut.node,
                    value=mapping_back[mut.derived_state]["value"].copy(),
                    # DEV:
                    # inner_summation = mapping_back[mut.derived_state]['inner_summation']
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
                    ValueTransition(
                        tree_node=edge.child,
                        value=T[T_index[u]].value.copy(),
                        value_copy=copy.deepcopy(T[T_index[u]].value_copy),
                    )
                )
                # Add on this extra node to each of the internal lists
                for st in T:
                    if not isinstance(st.value_copy, int):
                        st.value_copy.append(
                            InternalValueTransition(
                                tree_node=edge.child,
                                value=st.value_copy.copy()[T_index[u]].value,
                                # DEV:
                                # inner_summation=st.value_copy.copy()[T_index[u]].inner_summation
                            )
                        )
                # print(f'node: {edge.child} appended')
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
                            value=T[T_index[edge.child]].value.copy(),
                            value_copy=copy.deepcopy(T[T_index[edge.child]].value_copy),
                        )
                    )
                    # Add on this extra node to each of the internal lists
                    for st in T:
                        if not isinstance(st.value_copy, int):
                            st.value_copy.append(
                                InternalValueTransition(
                                    tree_node=edge.parent,
                                    value=st.value_copy.copy()[
                                        T_index[edge.child]
                                    ].value,
                                    # DEV:
                                    # inner_summation=st.value_copy.copy()[T_index[edge.child]].inner_summation
                                )
                            )
                    # print(f'node: {edge.parent} appended')
            else:
                # Grafting into an existing subtree.
                while T_index[u] == -1:
                    u = parent[u]
                    assert u != -1
            assert T_index[u] != -1 and T_index[edge.child] != -1
            if (
                T[T_index[u]].value == T[T_index[edge.child]].value
            ).all():  # DEV: will need to change this condition
                st = T[T_index[edge.child]]
                # Mark the lower ValueTransition as unused.
                st.value = -1
                st.value_copy = -1
                # print(f'node {st.tree_node} removed')
                # Also need to mark the corresponding InternalValueTransition as unused for the remaining states
                for st2 in T:
                    if not isinstance(st2.value_copy, int):
                        st2.value_copy[T_index[edge.child]].value = -1
                        st2.value_copy[T_index[edge.child]].tree_node = -1

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
                        if not isinstance(st2.value_copy, int):
                            st2.value_copy[T_index[vt.tree_node]].value = -1
                            st2.value_copy[T_index[vt.tree_node]].tree_node = -1
                    T_index[vt.tree_node] = -1
                    vt.tree_node = -1
                vt.value_index = -1

        print(self.N)
        self.N = np.zeros(self.tree.num_nodes, dtype=int)
        node_map = {st.tree_node: st for st in self.T}

        for u in self.tree.samples():
            while u not in node_map:
                u = self.tree.parent(u)
            self.N[self.T_index[u]] += 1

        print(self.N)
        # for s1 in T:
        #     print(f's1: {s1.tree_node}')
        #     print(s1.value)
        #     print(s1.value_copy)
        #     if not isinstance(s1.value_copy, int):
        #         for s2 in s1.value_copy:
        #             print(f's1: {s1.tree_node}, s2: {s2.tree_node}, {s2.value}')

    def update_probabilities(self, site, genotype_state):
        tree = self.tree
        # T_index = self.T_index
        # T = self.T
        alleles = ["0", "1"]
        allelic_state = self.allelic_state
        # Set the allelic_state for this site.
        allelic_state[self.tree.root] = alleles.index(site.ancestral_state)
        # print("root state")
        # print(allelic_state[tree.root])

        if (
            site.id == 0
        ):  # Do this for the first site, as self.N isn't evaluated until compress (but we need it for the internal normalisation)
            # DEV: This is a hack to get the initial T etc correct
            n = self.ts.num_samples
            self.T = []
            self.T_index = np.zeros(self.ts.num_nodes, dtype=int) - 1
            self.T_index[tree.root] = 0
            self.T.append(
                ValueTransition(
                    tree_node=tree.root,
                    value=((1 / n) ** 2) * np.ones(n),
                    value_copy=[
                        InternalValueTransition(tree_node=tree.root, value=(1 / n) ** 2)
                    ],
                )
            )
            node_map = {st.tree_node: st for st in self.T}
            for u in tree.samples():
                while u not in node_map:
                    u = tree.parent(u)
                self.N[self.T_index[u]] += 1

        print(site.id)
        print(tree.draw_text())
        print(self.ts.draw_text())
        print(self.T_index)
        print(self.N)

        T_index = self.T_index
        T = self.T
        normalisation_factor_inner = {}

        for st1 in T:
            if st1.tree_node != -1:
                F_previous = self.T[self.T_index[st1.tree_node]].value_copy
                normalisation_factor_inner[
                    st1.tree_node
                ] = self.compute_normalisation_factor_inner_dict(st1.tree_node)
                print(normalisation_factor_inner[st1.tree_node])

        for st1 in T:
            if st1.tree_node != -1:
                for st2 in st1.value_copy:
                    if st2.tree_node != -1:
                        self.T[self.T_index[st1.tree_node]].value_copy[
                            self.T_index[st2.tree_node]
                        ].inner_summation = (
                            normalisation_factor_inner[st1.tree_node]
                            + normalisation_factor_inner[st2.tree_node]
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
                        value=T[T_index[u]].value.copy(),
                        value_copy=copy.deepcopy(T[T_index[u]].value_copy),
                    )  # DEV: is it possible to not use deepcopies?
                )
                for st in T:
                    if not isinstance(st.value_copy, int):
                        st.value_copy.append(
                            InternalValueTransition(
                                tree_node=mutation.node,
                                value=st.value_copy.copy()[T_index[u]].value,
                                inner_summation=st.value_copy.copy()[
                                    T_index[u]
                                ].inner_summation,
                            )
                        )

        # Extra bit that should be removed eventually - this is to get the allelic state at the leaves.
        allelic_state[: tree.num_samples()] = tree.tree_sequence.genotype_matrix()[
            site.id, :
        ]
        node_map = {st.tree_node: st for st in self.T}
        to_compute = np.zeros(
            tree.num_samples(), dtype=int
        )  # Because the node ordering in the leaves is not 0 -> n_samples - 1.

        # for v in tree.samples():
        for v in self.ts.samples():
            v_tmp = v
            while v not in node_map:
                v = tree.parent(v)
            to_compute[v_tmp] = v

        normalisation_factor_inner_transpose = [
            self.compute_normalisation_factor_inner(i) for i in to_compute
        ]
        print(normalisation_factor_inner_transpose)
        # print(f'genotype_state: {genotype_state}')
        # print(f'EQUAL_BOTH_HOM: {(1 - self.mu[site.id])**2}')
        # print(f'UNEQUAL_BOTH_HOM: {self.mu[site.id]**2}')
        # print(f'BOTH_HET: {((1-self.mu[site.id])**2 + self.mu[site.id]**2)}')
        # print(f'REF_HOM_OBS_HET: {(2*self.mu[site.id]*(1 - self.mu[site.id]))}')
        # print(f'REF_HET_OBS_HOM: {(self.mu[site.id]*(1-self.mu[site.id]))}')

        query_is_het = genotype_state == 1

        for st1 in T:
            # Before
            # if st1.tree_node != tskit.NULL:
            # print("BEFORE")
            # print(T_index)
            # print('vector version')
            # print(st1.value)
            # print('list version')
            # for s in st1.value_copy:
            #     if s.tree_node != tskit.NULL:
            #         print(f'{s.tree_node} : {s.value}')

            u1 = st1.tree_node
            st1.inner_summation = (
                self.compute_normalisation_factor_inner(u1)
                + normalisation_factor_inner_transpose
            )
            print("inner summation")
            print(st1.inner_summation)

            if u1 != -1:
                # Get the allelic_state at u. TODO we can cache these states to
                # avoid some upward traversals.
                v1 = u1
                while allelic_state[v1] == -1:
                    v1 = tree.parent(v1)
                    assert v1 != -1

                for st2 in st1.value_copy:
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
                        # print("")
                        # print(st2)
                        # print(f'node 1: {st1.tree_node}, node 2: {st2.tree_node}')
                        # print(f'old st2.value: {st2.value}')
                        # print(f'st.inner_summation: {st2.inner_summation}')
                        # print(f'match {match}')
                        # print(f'template_is_het {template_is_het}')
                        # print(f'genotype_template_state {genotype_template_state}')
                        print(st2.inner_summation)
                        st2.value = self.compute_next_probability_dict(
                            site.id,
                            st2.value,
                            st2.inner_summation,
                            match,
                            template_is_het,
                            query_is_het,
                        )
                        # print(f'new st2.value: {st2.value}')

                # This will ensure that allelic_state[:n] is filled
                genotype_template_state = (
                    allelic_state[v1] + allelic_state[: tree.num_samples()]
                )
                # These are vectors of length n (at internal nodes).
                match = genotype_state == genotype_template_state
                template_is_het = genotype_template_state == 1

                # print("")
                # print(f'node 1: {st1.tree_node}')
                # print('old st1.value:')
                # print(st1.value)
                # print('st.inner_summation:')
                # print(st1.inner_summation)
                # print('match')
                # print(match)
                # print('template_is_het')
                # print(template_is_het)
                # print('genotype_template_state')
                # print(genotype_template_state)

                st1.value = self.compute_next_probability(
                    site.id,
                    st1.value,
                    st1.inner_summation,
                    match,
                    template_is_het,
                    query_is_het,
                )

                # print("AFTER")
                # print(T_index)
                # print('vector version')
                # print(st1.value)
                # print('list version')
                # for s in st1.value_copy:
                # if s.tree_node != tskit.NULL:
                # print(f'{s.tree_node} : {s.value}')

        # Unset the states
        allelic_state[tree.root] = -1
        for mutation in site.mutations:
            allelic_state[mutation.node] = -1

    # Define function that maps leaves to node inherited from.

    def process_site(self, site, genotype_state, forwards=True):
        print(site)

        A1, B1 = self.decode_site()
        A2, B2 = self.decode_site_dict()

        if not (
            np.allclose(A1, A2, rtol=1e-5, atol=1e-8)
            and np.allclose(B1, B2, rtol=1e-5, atol=1e-8)
        ):
            print("before: difference between vector decoding and dict decoding")
            print(A1 - A2)
            print(B1 - B2)
            print(A1)
            print(A2)
            print(B1)
            print(B2)

        self.update_probabilities(site, genotype_state)
        A1, B1 = self.decode_site()
        A2, B2 = self.decode_site_dict()

        if not (
            np.allclose(A1, A2, rtol=1e-5, atol=1e-8)
            and np.allclose(B1, B2, rtol=1e-5, atol=1e-8)
        ):
            print("after: difference between vector decoding and dict decoding")
            print(A1 - A2)
            print(B1 - B2)
            print(A1)
            print(A2)
            print(B1)
            print(B2)

        assert np.allclose(A1, A2, rtol=1e-5, atol=1e-8)
        assert np.allclose(B1, B2, rtol=1e-5, atol=1e-8)

        # print("before compression")
        # for st in self.T:
        #     print(st.tree_node)
        #     print(st.value)
        #     for st2 in st.value_copy:
        #         print(f'{st2.tree_node}:{st2.value}')

        self.stupid_compress_dict()
        # print("")
        # print("after compression")
        # for st in self.T:
        #     print(st.tree_node)
        #     print(st.value)
        #     for st2 in st.value_copy:
        #         print(f'{st2.tree_node}:{st2.value}')

        A_compress1, B_compress1 = self.decode_site()
        A_compress2, B_compress2 = self.decode_site_dict()

        if not np.allclose(A1, A_compress1, rtol=1e-5, atol=1e-8):
            print(A1 - A_compress1)
        assert np.allclose(A1, A_compress1, rtol=1e-5, atol=1e-8)

        if not np.allclose(A2, A_compress2, rtol=1e-5, atol=1e-8):
            print(A2 - A_compress2)
        assert np.allclose(A2, A_compress2, rtol=1e-5, atol=1e-8)

        s = self.compute_normalisation_factor()
        s1 = self.compute_normalisation_factor_dict()

        assert isclose(s, s1)

        T = self.T
        for st in T:
            if st.tree_node != tskit.NULL:
                # print(f"tree node: {st.tree_node}")
                # print(f"value: {st.value}")
                # st.old_value = copy.deepcopy(st.value)
                # st.value = copy.deepcopy(st.value)
                st.value /= s
                st.value = np.round(st.value, self.precision)
                # Need to loop through value copy, and normalise
                for st2 in st.value_copy:
                    st2.value /= s1
                    st2.value = np.round(st2.value, self.precision)

        self.output.store_site(site.id, s, [(st.tree_node, st.value) for st in self.T])
        # I just need to pass a new version instead for storage.

    def run_forward(self, g):
        n = self.ts.num_samples
        self.tree.clear()
        for u in self.ts.samples():
            self.T_index[u] = len(self.T)
            self.T.append(
                ValueTransition(
                    tree_node=u, value=((1 / n) ** 2) * np.ones(n), value_copy=[]
                )
            )
            for v in self.ts.samples():
                self.T[self.T_index[u]].value_copy.append(
                    InternalValueTransition(tree_node=v, value=(1 / n) ** 2)
                )

        # # This (below) doesn't work because of tree structure before any data is star-like.
        # self.T_index[self.tree.root] = 0
        # self.T.append(ValueTransition(tree_node=self.tree.root, value=((1/n) ** 2) * np.ones(n), value_copy=[]))
        # self.T[self.T_index[self.tree.root].value_copy.append(InternalValueTransition(tree_node=self.tree.root, value=(1/n)**2))]

        # print(self.T)

        while self.tree.next():
            print("new tree has arrived")
            self.update_tree()

            A1, B1 = self.decode_site()
            A2, B2 = self.decode_site_dict()

            if not (
                np.allclose(A1, A2, rtol=1e-5, atol=1e-8)
                and np.allclose(B1, B2, rtol=1e-5, atol=1e-8)
            ):
                print("difference between old decoding and current decoding")
                print(A1 - A1_old)
                print(B1 - B1_old)
                print("difference between old dict decoding and current dict decoding")
                print(A2 - A2_old)
                print(B2 - B2_old)
                print("difference between vector decoding and dict decoding")
                print(A1 - A2)
                print(B1 - B2)
                print(B1)
                print(B2)
                print(B1_old)
                print(B2_old)
            assert np.allclose(A1, A2, rtol=1e-5, atol=1e-8)
            assert np.allclose(B1, B2, rtol=1e-5, atol=1e-8)

            for site in self.tree.sites():
                print(f"site {site.id}")
                self.process_site(site, g[site.id])

            print("last decoding before a new tree")
            A1_old, B1_old = self.decode_site()
            A2_old, B2_old = self.decode_site_dict()
            assert np.allclose(A1, A2, rtol=1e-5, atol=1e-8)
            assert np.allclose(B1, B2, rtol=1e-5, atol=1e-8)

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

    def compute_normalisation_factor_dict(self):
        s = 0
        for j, st in enumerate(self.T):
            assert st.tree_node != tskit.NULL
            assert self.N[j] > 0
            s += self.N[j] * self.compute_normalisation_factor_inner_dict(st.tree_node)
        return s

    def compute_normalisation_factor_inner(self, node):
        print(f"compute_normalisation_factor_inner, node {node}")
        print(self.T[self.T_index[node]].value)
        print(np.sum(self.T[self.T_index[node]].value))
        return np.sum(self.T[self.T_index[node]].value)

    def compute_normalisation_factor_inner_dict(self, node):
        s_inner = 0
        F_previous = self.T[self.T_index[node]].value_copy
        print(f"compute_normalisation_factor_inner_dict, node {node}")
        print(F_previous)
        print(self.N)
        for st in F_previous:
            j = st.tree_node
            if j != -1:
                print(self.T_index[j])
                print(self.N[self.T_index[j]])
                s_inner += self.N[self.T_index[j]] * st.value
        print(s_inner)
        return s_inner

    def compute_next_probability(
        self,
        site_id,
        p_last,
        inner_normalisation_factor,
        is_match,
        template_is_het,
        query_is_het,
    ):
        rho = self.rho[site_id]
        mu = self.mu[site_id]
        n = self.ts.num_samples

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

        p_t = (
            (rho / n) ** 2
            + ((1 - rho) * (rho / n)) * inner_normalisation_factor
            + (1 - rho) ** 2 * p_last
        )
        p_e = (
            EQUAL_BOTH_HOM * (1 - mu) ** 2
            + UNEQUAL_BOTH_HOM * (mu ** 2)
            + REF_HOM_OBS_HET * (2 * mu * (1 - mu))
            + REF_HET_OBS_HOM * (mu * (1 - mu))
            + BOTH_HET * ((1 - mu) ** 2 + mu ** 2)
        )

        return p_t * p_e

    def compute_next_probability_dict(
        self,
        site_id,
        p_last,
        inner_normalisation_factor,
        is_match,
        template_is_het,
        query_is_het,
    ):
        rho = self.rho[site_id]
        mu = self.mu[site_id]
        n = self.ts.num_samples

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

        p_t = (
            (rho / n) ** 2
            + ((1 - rho) * (rho / n)) * inner_normalisation_factor
            + (1 - rho) ** 2 * p_last
        )
        p_e = (
            EQUAL_BOTH_HOM * (1 - mu) ** 2
            + UNEQUAL_BOTH_HOM * (mu ** 2)
            + REF_HOM_OBS_HET * (2 * mu * (1 - mu))
            + REF_HET_OBS_HOM * (mu * (1 - mu))
            + BOTH_HET * ((1 - mu) ** 2 + mu ** 2)
        )

        return p_t * p_e


def ls_forward_tree(g, ts, rho, mu, A, c, precision=30):
    """Forward matrix computation based on a tree sequence."""
    fa = ForwardAlgorithm(ts, rho, mu, A, c, precision=precision)
    return fa.run_forward(g)
