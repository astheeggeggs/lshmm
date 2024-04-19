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

    # Prepare example reference panels and queries.
    def get_examples_haploid(self, ts):
        ref_panel = ts.genotype_matrix()
        num_sites = ref_panel.shape[0]
        query1 = ref_panel[:, 0].reshape(1, num_sites)
        query2 = ref_panel[:, -1].reshape(1, num_sites)
        ref_panel = ref_panel[:, 1:]
        # Create queries with MISSING
        query_miss_last = query1.copy()
        query_miss_last[0, -1] = core.MISSING
        query_miss_mid = query1.copy()
        query_miss_mid[0, ts.num_sites // 2] = core.MISSING
        query_miss_all = query1.copy()
        query_miss_all[0, :] = core.MISSING
        queries = [query1, query2, query_miss_last, query_miss_mid, query_miss_last]
        return ref_panel, queries

    def get_examples_diploid(self, ts):
        ref_panel = ts.genotype_matrix()
        num_sites = ref_panel.shape[0]
        query1 = ref_panel[:, 0].reshape(1, num_sites) + ref_panel[:, 1].reshape(
            1, num_sites
        )
        query2 = ref_panel[:, -1].reshape(1, num_sites) + ref_panel[:, -2].reshape(
            1, num_sites
        )
        ref_panel = ref_panel[:, 2:]
        # Create queries with MISSING
        query_miss_last = query1.copy()
        query_miss_last[0, -1] = core.MISSING
        query_miss_mid = query1.copy()
        query_miss_mid[0, ts.num_sites // 2] = core.MISSING
        query_miss_all = query1.copy()
        query_miss_all[0, :] = core.MISSING
        queries = [query1, query2]
        # FIXME Handle MISSING properly.
        # genotypes.append(s_miss_last)
        # genotypes.append(s_miss_mid)
        # genotypes.append(s_miss_all)
        ref_panel_size = ref_panel.shape[1]
        G = np.zeros((num_sites, ref_panel_size, ref_panel_size))
        for i in range(num_sites):
            G[i, :, :] = np.add.outer(ref_panel[i, :], ref_panel[i, :])
        return ref_panel, G, queries

    def get_examples_pars(
        self,
        ts,
        ploidy=None,
        scale_mutation_rate=None,
        mean_r=None,
        mean_mu=None,
        seed=42,
    ):
        """Returns an iterator over combinations of examples and parameters."""
        assert ploidy in [1, 2]
        assert scale_mutation_rate in [True, False]

        np.random.seed(seed)
        if ploidy == 1:
            H, queries = self.get_examples_haploid(ts)
        else:
            H, G, queries = self.get_examples_diploid(ts)

        m = ts.num_sites
        n = H.shape[1]

        rs = [
            np.zeros(m) + 0.01,  # Equal recombination and mutation
            np.zeros(m) + 0.999,  # Extreme
            np.zeros(m) + 1e-6,  # Extreme
            np.random.rand(m),  # Random
        ]
        mus = [
            np.zeros(m) + 0.01,  # Equal recombination and mutation
            np.zeros(m) + 0.2,  # Extreme
            np.zeros(m) + 1e-6,  # Extreme
            np.random.rand(m) * 0.2,  # Random
        ]
        if mean_r is not None and mean_mu is not None:
            rs.append(mean_r * (np.random.rand(m) + 0.5) / 2)
            mus.append(mean_mu * (np.random.rand(m) + 0.5) / 2)

        for s, r, mu in itertools.product(queries, rs, mus):
            r[0] = 0
            # Must be calculated from the genotype matrix,
            # because we can now get back mutations that
            # result in the number of alleles being higher
            # than the number of alleles in the reference panel.
            num_alleles = core.get_num_alleles(H, s)
            if ploidy == 1:
                e = core.get_emission_matrix_haploid(
                    mu, m, num_alleles, scale_mutation_rate
                )
                yield n, m, H, s, e, r, mu
            else:
                e = core.get_emission_matrix_diploid(mu, m)
                yield n, m, G, s, e, r, mu

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
