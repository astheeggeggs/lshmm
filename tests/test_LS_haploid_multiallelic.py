# Simulation
import itertools

# Python libraries
import msprime
import numpy as np
import pytest
import tskit

import lshmm.forward_backward.fb_haploid_variants_samples as fbh
import lshmm.vit_haploid_variants_samples as vh


class LSBase:
    """Superclass of Li and Stephens tests."""

    def example_haplotypes(self, ts):

        H = ts.genotype_matrix()
        s = H[:, 0].reshape(1, H.shape[0])
        H = H[:, 1:]

        return H, s

    def haplotype_emission(self, mu, m):
        # Define the emission probability matrix
        e = np.zeros((m, 2))
        e[:, 0] = mu
        e[:, 1] = 1 - mu

        return e

    def example_parameters_haplotypes(self, ts, seed=42):
        """Returns an iterator over combinations of haplotype, recombination and mutation rates."""
        np.random.seed(seed)
        H, s = self.example_haplotypes(ts)
        n = H.shape[1]
        m = ts.get_num_sites()

        alleles = []
        for variant in ts.variants():
            alleles.append(variant.alleles)

        # Here we have equal mutation and recombination
        r = np.zeros(m) + 0.01
        mu = np.zeros(m) + 0.01
        r[0] = 0

        e = self.haplotype_emission(mu, m)

        yield n, m, alleles, H, s, e, r

        # Mixture of random and extremes
        rs = [np.zeros(m) + 0.999, np.zeros(m) + 1e-6, np.random.rand(m)]

        mus = [np.zeros(m) + 0.33, np.zeros(m) + 1e-6, np.random.rand(m) * 0.33]

        e = self.haplotype_emission(mu, m)

        for r, mu in itertools.product(rs, mus):
            r[0] = 0
            e = self.haplotype_emission(mu, m)
            yield n, m, alleles, H, s, e, r

    def assertAllClose(self, A, B):
        """Assert that all entries of two matrices are 'close'"""
        assert np.allclose(A, B, rtol=1e-9, atol=0.0)

    # Define a bunch of very small tree-sequences for testing a collection of parameters on
    def test_simple_n_10_no_recombination(self):
        ts = msprime.sim_ancestry(
            samples=10,
            recombination_rate=0,
            random_seed=42,
            sequence_length=10,
            population_size=10000,
        )
        ts = msprime.sim_mutations(ts, rate=1e-5, random_seed=42)
        assert ts.num_sites > 3
        self.verify(ts)

    def test_simple_n_6(self):
        ts = msprime.sim_ancestry(
            samples=6,
            recombination_rate=1e-4,
            random_seed=42,
            sequence_length=40,
            population_size=10000,
        )
        ts = msprime.sim_mutations(ts, rate=1e-3, random_seed=42)
        assert ts.num_sites > 5
        self.verify(ts)

    def test_simple_n_8(self):
        ts = msprime.sim_ancestry(
            samples=8,
            recombination_rate=1e-4,
            random_seed=42,
            sequence_length=20,
            population_size=10000,
        )
        ts = msprime.sim_mutations(ts, rate=1e-4, random_seed=42)
        assert ts.num_sites > 5
        assert ts.num_trees > 15
        self.verify(ts)

    def test_simple_n_16(self):
        ts = msprime.sim_ancestry(
            samples=16,
            recombination_rate=1e-2,
            random_seed=42,
            sequence_length=20,
            population_size=10000,
        )
        ts = msprime.sim_mutations(ts, rate=1e-4, random_seed=42)
        assert ts.num_sites > 5
        print(ts.num_trees)
        self.verify(ts)

    def verify(self, ts):
        raise NotImplementedError()


class FBAlgorithmBase(LSBase):
    """Base for forwards backwards algorithm tests."""


class TestNonTreeMethodsHap(FBAlgorithmBase):
    """Test that we compute the sample likelihoods across all implementations."""

    def verify(self, ts):
        for n, m, alleles, H, s, e, r in self.example_parameters_haplotypes(ts):
            F, c, ll = fbh.forwards_ls_hap(n, m, H, s, e, r, norm=False)
            B = fbh.backwards_ls_hap(n, m, H, s, e, c, r)
            self.assertAllClose(np.log10(np.sum(F * B, 1)), ll * np.ones(m))
            F_tmp, c_tmp, ll_tmp = fbh.forwards_ls_hap(n, m, H, s, e, r, norm=True)
            B_tmp = fbh.backwards_ls_hap(n, m, H, s, e, c_tmp, r)
            self.assertAllClose(ll, ll_tmp)
            self.assertAllClose(np.sum(F_tmp * B_tmp, 1), np.ones(m))


class VitAlgorithmBase(LSBase):
    """Base for viterbi algoritm tests."""


class TestNonTreeViterbiHap(VitAlgorithmBase):
    """Test that we have the same log-likelihood across all implementations"""

    def verify(self, ts):
        for n, m, alleles, H, s, e, r in self.example_parameters_haplotypes(ts):

            V, P, ll = vh.forwards_viterbi_hap_naive(n, m, H, s, e, r)
            path = vh.backwards_viterbi_hap(m, V[m - 1, :], P)
            ll_check = vh.path_ll_hap(n, m, H, path, s, e, r)
            self.assertAllClose(ll, ll_check)

            V_tmp, P_tmp, ll_tmp = vh.forwards_viterbi_hap_lower_mem_rescaling(
                n, m, H, s, e, r
            )
            path_tmp = vh.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh.path_ll_hap(n, m, H, path_tmp, s, e, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll, ll_tmp)
