# Simulation
import itertools

# Python libraries
import msprime
import numpy as np
import pytest
import tskit

import lshmm.forward_backward.fb_diploid_variants_samples as fbd_vs
import lshmm.forward_backward.fb_haploid_variants_samples as fbh_vs
import lshmm.ls as ls
import lshmm.viterbi.vit_diploid_variants_samples as vd_vs
import lshmm.viterbi.vit_haploid_variants_samples as vh_vs

EQUAL_BOTH_HOM = 4
UNEQUAL_BOTH_HOM = 0
BOTH_HET = 7
REF_HOM_OBS_HET = 1
REF_HET_OBS_HOM = 2


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

    def genotype_emission(self, mu, m):
        # Define the emission probability matrix
        e = np.zeros((m, 8))
        e[:, EQUAL_BOTH_HOM] = (1 - mu) ** 2
        e[:, UNEQUAL_BOTH_HOM] = mu ** 2
        e[:, BOTH_HET] = 1 - mu
        e[:, REF_HOM_OBS_HET] = 2 * mu * (1 - mu)
        e[:, REF_HET_OBS_HOM] = mu * (1 - mu)

        return e

    def example_parameters_haplotypes(self, ts, seed=42):
        """Returns an iterator over combinations of haplotype, recombination and mutation rates."""
        np.random.seed(seed)
        H, s = self.example_haplotypes(ts)
        n = H.shape[1]
        m = ts.get_num_sites()

        # Here we have equal mutation and recombination
        r = np.zeros(m) + 0.01
        mu = np.zeros(m) + 0.01
        r[0] = 0

        e = self.haplotype_emission(mu, m)

        yield n, m, H, s, e, r

        # Mixture of random and extremes
        rs = [np.zeros(m) + 0.999, np.zeros(m) + 1e-6, np.random.rand(m)]

        mus = [np.zeros(m) + 0.33, np.zeros(m) + 1e-6, np.random.rand(m) * 0.33]

        e = self.haplotype_emission(mu, m)

        for r, mu in itertools.product(rs, mus):
            r[0] = 0
            yield n, m, H, s, e, r

    def example_parameters_haplotypes_larger(
        self, ts, seed=42, mean_r=1e-5, mean_mu=1e-5
    ):

        np.random.seed(seed)
        H, s = self.example_haplotypes(ts)
        n = H.shape[1]
        m = ts.get_num_sites()

        r = mean_r * np.ones(m) * ((np.random.rand(m) + 0.5) / 2)
        r[0] = 0

        # Error probability
        mu = mean_mu * np.ones(m) * ((np.random.rand(m) + 0.5) / 2)

        # Define the emission probability matrix
        e = self.haplotype_emission(mu, m)

        return n, m, H, s, e, r

    def example_genotypes(self, ts):

        H = ts.genotype_matrix()
        s = H[:, 0].reshape(1, H.shape[0]) + H[:, 1].reshape(1, H.shape[0])
        H = H[:, 2:]

        m = ts.get_num_sites()
        n = H.shape[1]

        G = np.zeros((m, n, n))
        for i in range(m):
            G[i, :, :] = np.add.outer(H[i, :], H[i, :])

        return H, G, s

    def example_parameters_genotypes(self, ts, seed=42):
        np.random.seed(seed)
        H, G, s = self.example_genotypes(ts)
        n = H.shape[1]
        m = ts.get_num_sites()

        # Here we have equal mutation and recombination
        r = np.zeros(m) + 0.01
        mu = np.zeros(m) + 0.01
        r[0] = 0

        e = self.genotype_emission(mu, m)

        yield n, m, G, s, e, r

        # Mixture of random and extremes
        rs = [np.zeros(m) + 0.999, np.zeros(m) + 1e-6, np.random.rand(m)]

        mus = [np.zeros(m) + 0.33, np.zeros(m) + 1e-6, np.random.rand(m) * 0.33]

        e = self.genotype_emission(mu, m)

        for r, mu in itertools.product(rs, mus):
            r[0] = 0
            yield n, m, G, s, e, r

    def example_parameters_genotypes_larger(
        self, ts, seed=42, mean_r=1e-5, mean_mu=1e-5
    ):

        np.random.seed(seed)
        H, G, s = self.example_genotypes(ts)

        m = ts.get_num_sites()
        n = H.shape[1]

        r = mean_r * np.ones(m) * ((np.random.rand(m) + 0.5) / 2)
        r[0] = 0

        # Error probability
        mu = mean_mu * np.ones(m) * ((np.random.rand(m) + 0.5) / 2)

        # Define the emission probability matrix
        e = self.genotype_emission(mu, m)

        return n, m, G, s, e, r

    def assertAllClose(self, A, B):
        """Assert that all entries of two matrices are 'close'"""
        assert np.allclose(A, B, rtol=1e-9, atol=0.0)

    # Define a bunch of very small tree-sequences for testing a collection of parameters on
    def test_simple_n_10_no_recombination(self):
        ts = msprime.simulate(
            10, recombination_rate=0, mutation_rate=0.5, random_seed=42
        )
        assert ts.num_sites > 3
        self.verify(ts)

    def test_simple_n_6(self):
        ts = msprime.simulate(6, recombination_rate=2, mutation_rate=7, random_seed=42)
        assert ts.num_sites > 5
        self.verify(ts)

    def test_simple_n_8(self):
        ts = msprime.simulate(8, recombination_rate=2, mutation_rate=5, random_seed=42)
        assert ts.num_sites > 5
        self.verify(ts)

    def test_simple_n_8_high_recombination(self):
        ts = msprime.simulate(8, recombination_rate=20, mutation_rate=5, random_seed=42)
        assert ts.num_trees > 15
        assert ts.num_sites > 5
        self.verify(ts)

    def test_simple_n_16(self):
        ts = msprime.simulate(16, recombination_rate=2, mutation_rate=5, random_seed=42)
        assert ts.num_sites > 5
        self.verify(ts)

    # Test a bigger one.
    def test_large(self, n=50, length=100000, mean_r=1e-5, mean_mu=1e-5, seed=42):
        ts = msprime.simulate(
            n + 1,
            length=length,
            mutation_rate=mean_mu,
            recombination_rate=mean_r,
            random_seed=seed,
        )
        self.verify_larger(ts)

    def verify(self, ts):
        raise NotImplementedError()

    def verify_larger(self, ts):
        pass


class FBAlgorithmBase(LSBase):
    """Base for forwards backwards algorithm tests."""


class TestNonTreeMethodsHap(FBAlgorithmBase):
    """Test that we compute the sample likelihoods across all implementations."""

    def verify(self, ts):
        for n, m, H_vs, s, e_vs, r in self.example_parameters_haplotypes(ts):

            F_vs, c_vs, ll_vs = fbh_vs.forwards_ls_hap(n, m, H_vs, s, e_vs, r)
            B_vs = fbh_vs.backwards_ls_hap(n, m, H_vs, s, e_vs, c_vs, r)

            F, c, ll = ls.forwards(n, m, H_vs, s, e_vs, r)
            B = ls.backwards(n, m, H_vs, s, e_vs, c, r)

            self.assertAllClose(F, F_vs)
            self.assertAllClose(B, B_vs)
            self.assertAllClose(ll_vs, ll)


class TestNonTreeMethodsDip(FBAlgorithmBase):
    """Test that we compute the sample likelihoods across all implementations."""

    def verify(self, ts):
        for n, m, G_vs, s, e_vs, r in self.example_parameters_genotypes(ts):

            F_vs, c_vs, ll_vs = fbd_vs.forward_ls_dip_loop(
                n, m, G_vs, s, e_vs, r, norm=True
            )
            B_vs = fbd_vs.backward_ls_dip_loop(n, m, G_vs, s, e_vs, c_vs, r)

            F, c, ll = ls.forwards(n, m, G_vs, s, e_vs, r)
            B = ls.backwards(n, m, G_vs, s, e_vs, c, r)

            self.assertAllClose(F, F_vs)
            self.assertAllClose(B, B_vs)
            self.assertAllClose(ll_vs, ll)


class VitAlgorithmBase(LSBase):
    """Base for viterbi algoritm tests."""


class TestNonTreeViterbiHap(VitAlgorithmBase):
    """Test that we have the same log-likelihood across all implementations"""

    def verify(self, ts):
        for n, m, H_vs, s, e_vs, r in self.example_parameters_haplotypes(ts):

            V_vs, P_vs, ll_vs = vh_vs.forwards_viterbi_hap_lower_mem_rescaling(
                n, m, H_vs, s, e_vs, r
            )
            path_vs = vh_vs.backwards_viterbi_hap(m, V_vs, P_vs)

            path, ll = ls.viterbi(n, m, H_vs, s, e_vs, r)

            self.assertAllClose(ll_vs, ll)
            self.assertAllClose(path_vs, path)


class TestNonTreeViterbiDip(VitAlgorithmBase):
    """Test that we have the same log-likelihood across all implementations"""

    def verify(self, ts):
        for n, m, G_vs, s, e_vs, r in self.example_parameters_genotypes(ts):

            V_vs, P_vs, ll_vs = vd_vs.forwards_viterbi_dip_low_mem(
                n, m, G_vs, s, e_vs, r
            )
            path_vs = vd_vs.backwards_viterbi_dip(m, V_vs, P_vs)
            phased_path_vs = vd_vs.get_phased_path(n, path_vs)

            path, ll = ls.viterbi(n, m, G_vs, s, e_vs, r)

            self.assertAllClose(ll_vs, ll)
            self.assertAllClose(phased_path_vs, path)
