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

    # https://github.com/tskit-dev/tsinfer/blob/0c206d319f9c0dcb1ee205d5cc56576e3a88775e/tsinfer/eval_util.py#L244
    def get_ancestral_haplotypes(self, ts):
        """
        Return an array of the haplotypes of the nodes in a tree sequence.

        Both ancestral and sample haplotypes in the tree sequence are stored
        in an array of size (number of nodes, number of sites).

        Note that haplotypes having only NONCOPY values are removed.

        :param numpy.ndarray ts: A tree sequence.
        :return: An array of haplotypes.
        :rtype: numpy.ndarray
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

        # Remove ancestral haplotypes with only NONCOPY values, which can happen
        # when recombination rate is high enough and mutation rate is low.
        A = A[~np.all(A == core.NONCOPY, axis=1),]

        assert np.all(
            np.sum(A != core.NONCOPY, axis=0) > 0
        ), "Some sites have only NONCOPY states."

        return A.T

    # Prepare example reference panels and queries.
    def get_examples_haploid(self, ts, include_ancestors):
        if include_ancestors:
            ref_panel = self.get_ancestral_haplotypes(ts)
        else:
            ref_panel = ts.genotype_matrix()
        # Take some haplotypes as queries from the reference panel.
        num_sites = ref_panel.shape[0]
        query_1 = ref_panel[:, 0].reshape(1, num_sites)
        query_1 = np.append(query_1[:2], query_1[2:]).reshape(1, num_sites)
        query_2 = query_1[::-1]
        # Create queries with MISSING.
        query_miss_last = query_1.copy()
        query_miss_last[0, -1] = core.MISSING
        query_miss_mid = query_1.copy()
        query_miss_mid[0, ts.num_sites // 2] = core.MISSING
        query_miss_most_1 = query_1.copy()
        query_miss_most_1[0, 2:] = core.MISSING
        query_miss_most_2 = query_miss_most_1[::-1]
        queries = [
            query_1,
            query_2,
            query_miss_last,
            query_miss_mid,
            query_miss_most_1,
            query_miss_most_2,
        ]
        return ref_panel, queries

    def get_examples_diploid(self, ts, include_ancestors):
        if include_ancestors:
            ref_panel = self.get_ancestral_haplotypes(ts)
        else:
            ref_panel = ts.genotype_matrix()
        # Take some haplotypes as queries from the reference panel.
        num_sites = ref_panel.shape[0]
        query_1 = np.zeros((2, num_sites), dtype=np.int8) - np.inf
        query_1[0, :] = ref_panel[:, 0].reshape(1, num_sites)
        query_1[1, :] = ref_panel[:, 1].reshape(1, num_sites)
        query_1 = np.append(query_1[:, :2], query_1[:, 2:]).reshape(2, num_sites)
        query_2 = query_1[:, ::-1]
        # Create queries with MISSING.
        query_miss_last = query_1.copy()
        query_miss_last[:, -1] = core.MISSING
        query_miss_mid = query_1.copy()
        query_miss_mid[:, ts.num_sites // 2] = core.MISSING
        query_miss_most_1 = query_1.copy()
        query_miss_most_1[:, 2:] = core.MISSING
        query_miss_most_2 = query_miss_most_1[:, ::-1]
        queries = [
            query_1,
            query_2,
            query_miss_last,
            query_miss_mid,
            query_miss_most_1,
            query_miss_most_2,
        ]
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
        """Return an iterator over combinations of examples and parameters."""
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
        n = H.shape[1]  # Number of haplotypes, including ancestors.

        assert core.check_genotype_matrix(
            H, ts.num_samples
        ), "Reference haplotypes have unexpected number of copiable entries."

        r_s = [
            np.append([0], np.zeros(m - 1) + 0.01),
            np.append([0], np.random.rand(m - 1)),
            np.append([0], 1e-5 * (np.random.rand(m - 1) + 0.5) / 2),
        ]

        mu_s = [
            np.zeros(m) + 0.01,  # Equal recombination and mutation
            np.random.rand(m) * 0.2,  # Random
            1e-5 * (np.random.rand(m) + 0.5) / 2,
            None,
        ]

        if include_extreme_rates:
            r_s.append(np.append([0], np.zeros(m - 1) + 0.2))
            r_s.append(np.append([0], np.zeros(m - 1) + 1e-6))
            mu_s.append(np.zeros(m) + 0.2)
            mu_s.append(np.zeros(m) + 1e-6)

        for query, r, mu in itertools.product(queries, r_s, mu_s):
            # Must be calculated from the genotype matrix,
            # because we can now get back mutations that
            # result in the number of alleles being higher
            # than the number of alleles in the reference panel.
            prob_mutation = mu
            if prob_mutation is None:
                # Note that n is the number of haplotypes, including ancestors.
                prob_mutation = np.zeros(m) + core.estimate_mutation_probability(n)

            num_alleles = core.get_num_alleles(H, query)

            if ploidy == 1:
                e = core.get_emission_matrix_haploid(
                    mu=prob_mutation,
                    num_sites=m,
                    num_alleles=num_alleles,
                    scale_mutation_rate=scale_mutation_rate,
                )
            else:
                e = core.get_emission_matrix_diploid(
                    mu=prob_mutation,
                    num_sites=m,
                    num_alleles=num_alleles,
                    scale_mutation_rate=scale_mutation_rate,
                )
                # In the diploid case, query is converted to unphased genotypes.
                query = core.convert_haplotypes_to_unphased_genotypes(query)

            yield n, m, H, query, e, r, mu

    # Prepare simple example datasets.
    def get_ts_simple_n10_no_recomb(self, seed=42):
        ts = msprime.sim_mutations(
            msprime.sim_ancestry(
                samples=10,
                ploidy=1,
                sequence_length=10,
                recombination_rate=0.0,
                random_seed=seed,
            ),
            rate=0.3,
            model=msprime.BinaryMutationModel(),
            discrete_genome=False,
            random_seed=seed,
        )
        assert ts.num_sites > 5
        assert ts.num_sites < 25
        return ts

    def get_ts_simple(self, num_samples, seed=42):
        ts = msprime.sim_mutations(
            msprime.sim_ancestry(
                samples=num_samples,
                ploidy=1,
                sequence_length=10,
                recombination_rate=2.0,
                random_seed=seed,
            ),
            rate=0.2,
            model=msprime.BinaryMutationModel(),
            discrete_genome=False,
            random_seed=seed,
        )
        assert ts.num_sites > 5
        assert ts.num_sites < 25
        return ts

    def get_ts_simple_n8_high_recomb(self, seed=42):
        ts = msprime.sim_mutations(
            msprime.sim_ancestry(
                samples=8,
                ploidy=1,
                sequence_length=20,
                recombination_rate=20.0,
                random_seed=seed,
            ),
            rate=0.2,
            model=msprime.BinaryMutationModel(),
            discrete_genome=False,
            random_seed=seed,
        )
        assert ts.num_trees > 15
        assert ts.num_sites > 5
        assert ts.num_sites < 25
        return ts

    def get_ts_custom_pars(self, num_samples, seq_length, mean_r, mean_mu, seed=42):
        ts = msprime.sim_mutations(
            msprime.sim_ancestry(
                samples=num_samples,
                ploidy=1,
                sequence_length=seq_length,
                recombination_rate=mean_r,
                random_seed=seed,
            ),
            rate=mean_mu,
            model=msprime.BinaryMutationModel(),
            discrete_genome=False,
            random_seed=seed,
        )
        return ts

    # Prepare example datasets with multiallelic sites.
    def get_ts_multiallelic_n10_no_recomb(self, seed=42):
        ts = msprime.sim_mutations(
            msprime.sim_ancestry(
                samples=10,
                recombination_rate=0,
                sequence_length=10,
                population_size=1e4,
                random_seed=seed,
            ),
            rate=1e-4,
            random_seed=seed,
        )
        assert ts.num_sites > 3
        return ts

    def get_ts_multiallelic(self, num_samples, seed=42):
        ts = msprime.sim_mutations(
            msprime.sim_ancestry(
                samples=num_samples,
                recombination_rate=1e-4,
                sequence_length=40,
                population_size=1e4,
                random_seed=seed,
            ),
            rate=1e-3,
            random_seed=seed,
        )
        assert ts.num_trees > 15
        assert ts.num_sites > 5
        return ts


class ForwardBackwardAlgorithmBase(LSBase):
    """Base for testing forwards-backwards algorithms."""


class ViterbiAlgorithmBase(LSBase):
    """Base for testing Viterbi algoritms."""
