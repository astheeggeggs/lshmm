import bisect
import itertools

import numpy as np

import msprime

import lshmm.core as core


class LSBase:
    """Base class of tests for Li & Stephens HMM algorithms."""

    def verify(self):
        raise NotImplementedError()

    def assertAllClose(self, A, B):
        np.testing.assert_allclose(A, B, rtol=1e-9, atol=0.0)

    def get_ancestral_haplotypes(self, ts):
        """
        Returns a numpy array of the haplotypes of the ancestors in the
        specified tree sequence.
        Modified from
        https://github.com/tskit-dev/tsinfer/blob/0c206d319f9c0dcb1ee205d5cc56576e3a88775e/tsinfer/eval_util.py#L244
        """
        tables = ts.dump_tables()
        nodes = tables.nodes
        flags = nodes.flags[:]
        flags[:] = 1
        nodes.set_columns(time=nodes.time, flags=flags)

        sites = tables.sites.position
        tsp = tables.tree_sequence()
        B = tsp.genotype_matrix().T

        # Modified. Originally, this was filled with NONCOPY by default.
        A = np.full((ts.num_nodes, ts.num_sites), core.NONCOPY, dtype=np.int8)
        for edge in ts.edges():
            start = bisect.bisect_left(sites, edge.left)
            end = bisect.bisect_right(sites, edge.right)
            if sites[end - 1] == edge.right:
                end -= 1
            A[edge.parent, start:end] = B[edge.parent, start:end]
        A[: ts.num_samples] = B[: ts.num_samples]

        assert np.all(
            np.sum(A != core.NONCOPY, axis=0) > 0
        ), "Some sites have only NONCOPY states."

        return A.T

    # Prepare example reference panels and queries.
    def get_examples_haploid(self, ts, include_ancestors):
        ref_panel = ts.genotype_matrix()
        num_sites = ref_panel.shape[0]
        query1 = ref_panel[:, 0].reshape(1, num_sites)
        query2 = ref_panel[:, -1].reshape(1, num_sites)
        if include_ancestors:
            ref_panel = self.get_ancestral_haplotypes(ts)
        else:
            ref_panel = ref_panel[:, 1:]
        # Create queries with MISSING.
        query_miss_last = query1.copy()
        query_miss_last[0, -1] = core.MISSING
        query_miss_mid = query1.copy()
        query_miss_mid[0, ts.num_sites // 2] = core.MISSING
        query_miss_most = query1.copy()
        query_miss_most[0, 1:] = core.MISSING
        queries = [query1, query2, query_miss_last, query_miss_mid, query_miss_most]
        return ref_panel, queries

    def get_examples_diploid(self, ts, include_ancestors):
        ref_panel = ts.genotype_matrix()
        num_sites = ref_panel.shape[0]
        # Take some haplotypes as queries from the reference panel.
        query_1 = np.zeros((2, num_sites), dtype=np.int32) - np.inf
        query_1[0, :] = ref_panel[:, 0].reshape(1, num_sites)
        query_1[1, :] = ref_panel[:, 1].reshape(1, num_sites)
        query_2 = np.zeros((2, num_sites), dtype=np.int32) - np.inf
        query_2[0, :] = ref_panel[:, -2].reshape(1, num_sites)
        query_2[1, :] = ref_panel[:, -1].reshape(1, num_sites)
        # Create queries with MISSING.
        query_miss_last = query_1.copy()
        query_miss_last[:, -1] = core.MISSING
        query_miss_mid = query_1.copy()
        query_miss_mid[:, ts.num_sites // 2] = core.MISSING
        query_miss_most = query_1.copy()
        query_miss_most[:, 1:] = core.MISSING
        queries = [query_1, query_2, query_miss_last, query_miss_mid, query_miss_most]
        # Exclude the arbitrarily chosen queries from the reference panel.
        if include_ancestors:
            ref_panel = self.get_ancestral_haplotypes(ts)
        else:
            ref_panel = ref_panel[:, 2:-2]
        return ref_panel, queries

    def get_examples_pars(
        self,
        ts,
        ploidy,
        scale_mutation_rate,
        include_ancestors,
        include_extreme_rates,
        seed=42,
    ):
        """Returns an iterator over combinations of examples and parameters."""
        assert ploidy in [1, 2]
        assert scale_mutation_rate in [True, False]
        assert include_ancestors in [True, False]
        assert include_extreme_rates in [True, False]

        np.random.seed(seed)
        if ploidy == 1:
            H, queries = self.get_examples_haploid(ts, include_ancestors)
        else:
            H, queries = self.get_examples_diploid(ts, include_ancestors)

        m = ts.num_sites
        n = H.shape[1]

        rs = [
            np.zeros(m) + 0.01,  # Equal recombination and mutation
            np.zeros(m) + 0.999,  # Extreme
            np.zeros(m) + 1e-6,  # Extreme
            np.random.rand(m),  # Random
            1e-5 * (np.random.rand(m) + 0.5) / 2,
        ]
        mus = [
            np.zeros(m) + 0.01,  # Equal recombination and mutation
            np.random.rand(m) * 0.2,  # Random
            1e-5 * (np.random.rand(m) + 0.5) / 2,
        ]

        if include_extreme_rates:
            rs.append(np.zeros(m) + 0.2)
            rs.append(np.zeros(m) + 1e-6)
            mus.append(np.zeros(m) + 0.2)
            mus.append(np.zeros(m) + 1e-6)

        for query, r, mu in itertools.product(queries, rs, mus):
            r[0] = 0
            # Must be calculated from the genotype matrix,
            # because we can now get back mutations that
            # result in the number of alleles being higher
            # than the number of alleles in the reference panel.
            num_alleles = core.get_num_alleles(H, query)
            if ploidy == 1:
                e = core.get_emission_matrix_haploid(
                    mu=mu,
                    num_sites=m,
                    num_alleles=num_alleles,
                    scale_mutation_rate=scale_mutation_rate,
                )
                yield n, m, H, query, e, r, mu
            else:
                e = core.get_emission_matrix_diploid(
                    mu=mu,
                    num_sites=m,
                    num_alleles=num_alleles,
                    scale_mutation_rate=scale_mutation_rate,
                )
                yield n, m, H, query, e, r, mu

    # Prepare simple example datasets.
    def get_ts_simple_n10_no_recomb(self, seed=42):
        ts = msprime.simulate(
            10,
            recombination_rate=0,
            mutation_rate=0.5,
            random_seed=seed,
        )
        assert ts.num_sites > 3
        return ts

    def get_ts_simple_n6(self, seed=42):
        ts = msprime.simulate(
            6,
            recombination_rate=2,
            mutation_rate=7,
            random_seed=seed,
        )
        assert ts.num_sites > 5
        return ts

    def get_ts_simple_n8(self, seed=42):
        ts = msprime.simulate(
            8,
            recombination_rate=2,
            mutation_rate=5,
            random_seed=seed,
        )
        assert ts.num_sites > 5
        return ts

    def get_ts_simple_n8_high_recomb(self, seed=42):
        ts = msprime.simulate(
            8,
            recombination_rate=20,
            mutation_rate=5,
            random_seed=seed,
        )
        assert ts.num_trees > 15
        assert ts.num_sites > 5
        return ts

    def get_ts_simple_n16(self, seed=42):
        ts = msprime.simulate(
            16,
            recombination_rate=2,
            mutation_rate=5,
            random_seed=seed,
        )
        assert ts.num_sites > 5
        return ts

    def get_ts_custom_pars(self, ref_panel_size, length, mean_r, mean_mu, seed=42):
        ts = msprime.simulate(
            ref_panel_size + 1,
            length=length,
            recombination_rate=mean_r,
            mutation_rate=mean_mu,
            random_seed=seed,
        )
        return ts

    # Prepare example datasets with multiallelic sites.
    def get_ts_multiallelic_n10_no_recomb(self, seed=42):
        ts = msprime.sim_ancestry(
            samples=10,
            recombination_rate=0,
            sequence_length=10,
            population_size=1e4,
            random_seed=seed,
        )
        ts = msprime.sim_mutations(
            ts,
            rate=1e-5,
            random_seed=seed,
        )
        assert ts.num_sites > 3
        return ts

    def get_ts_multiallelic_n6(self, seed=42):
        ts = msprime.sim_ancestry(
            samples=6,
            recombination_rate=1e-4,
            sequence_length=40,
            population_size=1e4,
            random_seed=seed,
        )
        ts = msprime.sim_mutations(
            ts,
            rate=1e-3,
            random_seed=seed,
        )
        assert ts.num_sites > 5
        return ts

    def get_ts_multiallelic_n8(self, seed=42):
        ts = msprime.sim_ancestry(
            samples=8,
            recombination_rate=1e-4,
            sequence_length=20,
            population_size=1e4,
            random_seed=seed,
        )
        ts = msprime.sim_mutations(
            ts,
            rate=1e-4,
            random_seed=seed,
        )
        assert ts.num_sites > 5
        assert ts.num_trees > 15
        return ts

    def get_ts_multiallelic_n16(self, seed=42):
        ts = msprime.sim_ancestry(
            samples=16,
            recombination_rate=1e-2,
            sequence_length=20,
            population_size=1e4,
            random_seed=seed,
        )
        ts = msprime.sim_mutations(
            ts,
            rate=1e-4,
            random_seed=seed,
        )
        assert ts.num_sites > 5
        return ts


class ForwardBackwardAlgorithmBase(LSBase):
    """Base for testing forwards-backwards algorithms."""


class ViterbiAlgorithmBase(LSBase):
    """Base for testing Viterbi algoritms."""
