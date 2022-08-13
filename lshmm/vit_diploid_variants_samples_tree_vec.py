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

    # DEV: sort this
    def update_probabilities(self, site, genotype_state):
        tree = self.tree
        T_index = self.T_index
        T = self.T
        alleles = ["0", "1"]
        allelic_state = self.allelic_state
        # Set the allelic_state for this site.
        allelic_state[tree.root] = alleles.index(site.ancestral_state)

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
        # Get the allelic state at the leaves.
        allelic_state[: tree.num_samples()] = tree.tree_sequence.genotype_matrix()[
            site.id, :
        ]

        node_map = {st.tree_node: st for st in self.T}
        to_compute = np.zeros(
            tree.num_samples(), dtype=int
        )  # Because the node ordering in the leaves is not 0 -> n_samples - 1.

        for v in self.ts.samples():
            v_tmp = v
            while v not in node_map:
                v = tree.parent(v)
            to_compute[v_tmp] = v

        single_switch = [self.compute_normalisation_factor_inner(i) for i in to_compute]

        genotype_template_state = np.add.outer(
            allelic_state[: tree.num_samples()], allelic_state[: tree.num_samples()]
        )
        # These are vectors of length n (at internal nodes).
        query_is_het = genotype_state == 1

        for st in T:
            u = st.tree_node
            j2_switch = self.compute_normalisation_factor_inner(u)
            st.inner_summation = [
                max(j2_switch, j1_switch) for j1_switch in single_switch
            ]
            # Evaluate for st and compare to st.inner_summation and take the max
            # There needs to be a replacement here with the max evaluated across one of the samples

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
                template_is_het = genotype_template_state == 1

                # This is with the first element in the pair fixed - the outer 'node' (j1).
                # V_previous[j1,:] in the matrix code is st.value here
                st.value = self.compute_next_probability(
                    site.id,
                    st.value,
                    st.inner_summation,
                    match,
                    template_is_het,
                    query_is_het,
                    u,
                )

        # Unset the states
        allelic_state[tree.root] = -1
        for mutation in site.mutations:
            allelic_state[mutation.node] = -1

    def process_site(self, site, genotype_state):
        self.update_probabilities(site, genotype_state)
        self.stupid_compress()
        s = self.compute_normalisation_factor()
        T = self.T

        for st in T:
            if st.tree_node != tskit.NULL:
                st.value /= s
                st.value = np.round(st.value, self.precision)

        self.output.store_site(site.id, s, [(st.tree_node, st.value) for st in self.T])

    def run_viterbi(self, g):
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
                self.process_site(site, g[site.id])
                # A, B = self.decode_site()
                # print(A)
                # print(B)
                # assert 0 == 1
        return self.output

    def compute_normalisation_factor(self):
        raise NotImplementedError()

    def compute_next_probability(
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


class ViterbiMatrix(CompressedMatrix):
    """Class representing the compressed Viterbi matrix."""

    def __init__(self, ts):
        super().__init__(ts)
        self.double_recombination_required = [
            (-1, 0, 0, False)
        ]  # Tuple containing the site, the pair of nodes in the tree, and whether recombination is required
        self.single_recombination_required = [(-1, 0, 0, False)]

    def add_single_recombination_required(self, site, node_s1, leaf_s2, required):
        # Note that here, the second entry is an element of the internal value transition, representing the leaf on the other tree.
        self.single_recombination_required.append((site, node_s1, leaf_s2, required))

    def add_double_recombination_required(self, site, node_s1, leaf_s2, required):
        # Note that here, the second entry is an element of the internal value transition, representing the leaf on the other tree.
        self.double_recombination_required.append((site, node_s1, leaf_s2, required))

    def choose_sample_double(self, site_id, tree):
        max_value = -1
        u1 = -1
        # Note, in this initial vector version, u2 is not compressed, so we can just read it off.
        # print(self.value_transitions[site_id])
        # print(tree.draw_text())
        for node, value in self.value_transitions[site_id]:
            u2_tmp = np.argmax(value)
            # print(u2_tmp)
            value_tmp = value[u2_tmp]
            if value_tmp > max_value:
                max_value = value_tmp
                u1 = node
                u2 = u2_tmp
        assert u1 != -1

        transition_nodes = [u_tmp for (u_tmp, _) in self.value_transitions[site_id]]
        # print(u1)
        while not tree.is_sample(u1):
            for v in tree.children(u1):
                if v not in transition_nodes:
                    u1 = v
                    # print(u1)
                    break
            else:
                raise AssertionError("could not find path")
        # print(f"({u1}, {u2})")
        return (u1, u2)

    def choose_sample_single(self, site_id, tree, current_nodes):

        # I want to find which is the max between any choice if I switch just u1, and any choice if I switch just u2.
        node_map = {st[0]: st for st in self.value_transitions[site_id]}
        to_compute = np.zeros(
            2, dtype=int
        )  # We have two to compute - one for each single switch set of possibilities.

        u1 = current_nodes[0]
        u2 = current_nodes[1]

        for i, v in enumerate(current_nodes):  # (u1, u2)
            while v not in node_map:
                v = tree.parent(v)
            to_compute[i] = v

        # Need to go to the (j1 :)th entries, and the (:,j2)the entries, and pick the best.
        # print(to_compute)
        # print(tree.draw_text())

        T_index = np.zeros(self.ts.num_nodes, dtype=int) - 1
        for j, st in enumerate(self.value_transitions[site_id]):
            T_index[st[0]] = j

        node_single_switch_maxes = [
            np.argmax(self.value_transitions[site_id][T_index[node]][1])
            for node in to_compute
        ]
        # print(node_single_switch_maxes)
        single_switch = [
            np.max(self.value_transitions[site_id][T_index[node]][1])
            for node in to_compute
        ]
        # print(f"u1: {u1} -> to_compute {to_compute[0]}")
        # print(f"u2: {u2} -> to_compute {to_compute[1]}")
        # print(tree.draw_text())
        # print(self.value_transitions[site_id])
        # print(single_switch)
        # print(node_single_switch_maxes)
        # print([np.max(self.value_transitions[site_id][T_index[node]][1]) for node in range(self.ts.num_samples)])

        # In this case, we have a vector of leaves, so don't need to worry about traversing up or down.
        # print(np.argmax(single_switch))
        if np.argmax(single_switch) == 0:
            # u1 is fixed, and we switch u2.
            current_nodes = (u1, node_single_switch_maxes[0])
        else:
            # u2 is fixed, and we switch u1.
            current_nodes = (node_single_switch_maxes[1], u2)

        return current_nodes

    def traceback(self):
        # Run the traceback.
        m = self.ts.num_sites
        n = self.ts.num_samples
        match = np.zeros((m, 2), dtype=int)

        single_recombination_tree = np.zeros((self.ts.num_nodes, n), dtype=int) - 1
        double_recombination_tree = np.zeros((self.ts.num_nodes, n), dtype=int) - 1

        tree = tskit.Tree(self.ts)
        tree.last()
        double_switch = True
        current_nodes = (-1, -1)
        current_node_outer = current_nodes[0]

        # print("bam")
        # print(self.single_recombination_required)
        # print("double")
        # print(self.double_recombination_required)

        # print(self.decode())
        # print(self.value_transitions)

        rr_single_index = len(self.single_recombination_required) - 1
        rr_double_index = len(self.double_recombination_required) - 1

        for site in reversed(self.ts.sites()):
            # print(current_nodes)
            while tree.interval.left > site.position:
                tree.prev()
            assert tree.interval.left <= site.position < tree.interval.right

            # Fill in the recombination single tree
            j_single = rr_single_index  # This starts from the end of all the recombination required information, and includes all the information for the current site.
            while self.single_recombination_required[j_single][0] == site.id:
                u1, u2, required = self.single_recombination_required[j_single][1:]
                single_recombination_tree[u1, u2] = required
                j_single -= 1

            # Fill in the recombination double tree
            j_double = rr_double_index  # This starts from the end of all the recombination required information, and includes all the information for the current site.
            while self.double_recombination_required[j_double][0] == site.id:
                u1, u2, required = self.double_recombination_required[j_double][1:]
                double_recombination_tree[u1, u2] = required
                j_double -= 1

            # Note - current nodes are the leaf nodes.
            if current_node_outer == -1:
                if double_switch:
                    # print("unsure, double switch")
                    current_nodes = self.choose_sample_double(site.id, tree)
                    # print("switched")
                    # print(current_nodes)
                else:
                    # print("unsure, single switch")
                    # print(single_recombination_tree[u1,:])
                    # print(tree.draw_text())
                    # print("decode site")
                    # print(self.decode_site(tree, site.id))
                    # print(f"switched from {current_nodes}")
                    current_nodes = self.choose_sample_single(
                        site.id, tree, current_nodes
                    )
                    # print(f"to {current_nodes}")

            match[site.id, :] = current_nodes

            # Now traverse up the tree from the current node. The first marked node
            # we meet tells us whether we need to recombine.
            current_node_outer = current_nodes[0]
            u1 = current_node_outer
            u2 = current_nodes[1]

            # print(double_recombination_tree)
            # print(single_recombination_tree)

            # Just need to move up the tree to evaluate u1. u2 is fixed (it must be a leaf).
            if double_switch:
                while (
                    u1 != -1 and double_recombination_tree[u1, 0] == -1
                ):  # If the first entry is -1, all of them are.
                    u1 = tree.parent(u1)
            else:
                while (
                    u1 != -1 and single_recombination_tree[u1, 0] == -1
                ):  # If the first entry is -1, all of them are.
                    u1 = tree.parent(u1)

            assert u1 != -1

            # print(double_recombination_tree[u1])
            if double_recombination_tree[u1, u2] == 1:
                # Need to switch at the next site.
                current_node_outer = -1
                double_switch = True
            elif single_recombination_tree[u1, u2] == 1:
                # Need to single switch at the next site
                # print("decided single switch at previous position")
                # print(u2)
                # print(single_recombination_tree[u1,:])
                current_node_outer = -1
                double_switch = False

            # Reset the nodes in the double recombination tree.
            j = rr_single_index
            while self.single_recombination_required[j][0] == site.id:
                u1_tmp, u2_tmp, _ = self.single_recombination_required[j][1:]
                single_recombination_tree[u1_tmp, u2_tmp] = -1
                j -= 1
            rr_single_index = j

            # Reset the nodes in the single recombination tree.
            j = rr_double_index
            while self.double_recombination_required[j][0] == site.id:
                u1_tmp, u2_tmp, _ = self.double_recombination_required[j][1:]
                double_recombination_tree[u1_tmp, u2_tmp] = -1
                j -= 1
            rr_double_index = j

            # single_recombination_tree = np.zeros((self.ts.num_nodes, n), dtype=int) - 1 # This'll need to change.
            # double_recombination_tree = np.zeros((self.ts.num_nodes, n), dtype=int) - 1

        # print(single_recombination_tree)
        # print(double_recombination_tree)

        return match


class ViterbiAlgorithm(LsHmmAlgorithm):
    """
    Runs the Li and Stephens Viterbi algorithm.
    """

    def __init__(self, ts, rho, mu, precision=10):
        super().__init__(ts, rho, mu, precision)
        self.output = ViterbiMatrix(ts)

    # This loops through everything in the tree (via the ValueTransitions) and finds the maximum value
    def compute_normalisation_factor(self):
        s = 0
        for st in self.T:
            assert st.tree_node != tskit.NULL
            max_st = np.max(st.value)
            if max_st > s:
                s = max_st
        if s == 0:
            raise ValueError(
                "Trying to match non-existent allele with zero mutation rate"
            )
        return s

    def compute_normalisation_factor_inner(self, node):
        return np.max(self.T[self.T_index[node]].value)

    # DEV: sort this
    def compute_next_probability(
        self,
        site_id,
        p_last,  # In this context, this is a vector of likelihoods from the previous step representing the likelihood of the most likely path
        # where one is copying from this internal node and the other is copying from each node represented in the vector.
        inner_normalisation_factor,
        is_match,
        template_is_het,
        query_is_het,
        node,
    ):
        # Within this function we are evaluating the shift of a vector V from t-1 to t
        # It has been normalised from the previous step so that the maximum value in V at t-1 is 1.

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

        # Emission probabilities (not required if we don't need to evaluate the likelihood)
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
        single_switch_tmp = [single_switch * ss for ss in V_single_switch]

        for j2 in range(n):

            double_recombination_required = False
            single_recombination_required = False

            # Single or double switch?
            if single_switch_tmp[j2] > double_switch:
                # Then single switch is the alternative
                if p_t[j2] < single_switch * V_single_switch[j2]:
                    p_t[j2] = single_switch * V_single_switch[j2]
                    single_recombination_required = True
            else:
                # Double switch is the alternative
                if p_t[j2] < double_switch:
                    p_t[j2] = double_switch
                    double_recombination_required = True

            self.output.add_single_recombination_required(
                site_id, node, j2, single_recombination_required
            )
            self.output.add_double_recombination_required(
                site_id, node, j2, double_recombination_required
            )

        return p_t * p_e


def ls_viterbi_tree(h, ts, rho, mu, precision=30):
    """
    Viterbi path computation based on a tree sequence.
    """
    va = ViterbiAlgorithm(ts, rho, mu, precision=precision)
    return va.run_viterbi(h)
