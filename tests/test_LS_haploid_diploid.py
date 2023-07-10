# Simulation
import itertools

# Python libraries
import msprime
import numpy as np
import pytest

import lshmm.forward_backward.fb_diploid_samples_variants as fbd_sv
import lshmm.forward_backward.fb_diploid_variants_samples as fbd_vs
import lshmm.forward_backward.fb_haploid_samples_variants as fbh_sv
import lshmm.forward_backward.fb_haploid_variants_samples as fbh_vs
import lshmm.vit_diploid_samples_variants as vd_sv
import lshmm.vit_diploid_variants_samples as vd_vs
import lshmm.vit_haploid_samples_variants as vh_sv
import lshmm.vit_haploid_variants_samples as vh_vs

EQUAL_BOTH_HOM = 4
UNEQUAL_BOTH_HOM = 0
BOTH_HET = 7
REF_HOM_OBS_HET = 1
REF_HET_OBS_HOM = 2

MISSING = -1
MISSING_INDEX = 3

import numba as nb
import tskit


class LSBase:
    """Superclass of Li and Stephens tests."""

    def example_haplotypes(self, ts):

        H = ts.genotype_matrix()
        s = H[:, 0].reshape(1, H.shape[0])
        H = H[:, 1:]

        haplotypes = [s, H[:, -1].reshape(1, H.shape[0])]
        s_tmp = s.copy()
        s_tmp[0, -1] = MISSING
        haplotypes.append(s_tmp)
        s_tmp = s.copy()
        s_tmp[0, ts.num_sites // 2] = MISSING
        haplotypes.append(s_tmp)
        s_tmp = s.copy()
        s_tmp[0, :] = MISSING
        haplotypes.append(s_tmp)

        return H, haplotypes

    def haplotype_emission(self, mu, m):
        # Define the emission probability matrix
        e = np.zeros((m, 2))
        e[:, 0] = mu  # If they match
        e[:, 1] = 1 - mu  # If they don't match

        return e

    def genotype_emission(self, mu, m):
        # Define the emission probability matrix
        e = np.zeros((m, 8))
        e[:, EQUAL_BOTH_HOM] = (1 - mu) ** 2
        e[:, UNEQUAL_BOTH_HOM] = mu ** 2
        e[:, BOTH_HET] = 1 - mu
        e[:, REF_HOM_OBS_HET] = 2 * mu * (1 - mu)
        e[:, REF_HET_OBS_HOM] = mu * (1 - mu)
        e[:, MISSING_INDEX] = 1

        return e

    def example_parameters_haplotypes(self, ts, seed=42):
        """Returns an iterator over combinations of haplotype, recombination and mutation rates."""
        np.random.seed(seed)
        H, haplotypes = self.example_haplotypes(ts)
        n = H.shape[1]
        m = ts.get_num_sites()

        # Here we have equal mutation and recombination
        r = np.zeros(m) + 0.01
        mu = np.zeros(m) + 0.01
        r[0] = 0

        e = self.haplotype_emission(mu, m)

        for s in haplotypes:
            yield n, m, H, s, e, r

        # Mixture of random and extremes
        rs = [np.zeros(m) + 0.999, np.zeros(m) + 1e-6, np.random.rand(m)]
        mus = [np.zeros(m) + 0.33, np.zeros(m) + 1e-6, np.random.rand(m) * 0.33]

        e = self.haplotype_emission(mu, m)

        for s, r, mu in itertools.product(haplotypes, rs, mus):
            r[0] = 0
            e = self.haplotype_emission(mu, m)
            yield n, m, H, s, e, r

    def example_parameters_haplotypes_larger(
        self, ts, seed=42, mean_r=1e-5, mean_mu=1e-5
    ):

        np.random.seed(seed)
        H, haplotypes = self.example_haplotypes(ts)
        n = H.shape[1]
        m = ts.get_num_sites()

        r = mean_r * np.ones(m) * ((np.random.rand(m) + 0.5) / 2)
        r[0] = 0

        # Error probability
        mu = mean_mu * np.ones(m) * ((np.random.rand(m) + 0.5) / 2)

        # Define the emission probability matrix
        e = self.haplotype_emission(mu, m)

        for s in haplotypes:
            yield n, m, H, s, e, r

    def example_genotypes(self, ts, seed=42):

        H = ts.genotype_matrix()
        s = H[:, 0].reshape(1, H.shape[0]) + H[:, 1].reshape(1, H.shape[0])
        H = H[:, 2:]

        genotypes = [
            s,
            H[:, -1].reshape(1, H.shape[0]) + H[:, -2].reshape(1, H.shape[0]),
        ]

        s_tmp = s.copy()
        s_tmp[0, -1] = MISSING
        genotypes.append(s_tmp)
        s_tmp = s.copy()
        s_tmp[0, ts.num_sites // 2] = MISSING
        genotypes.append(s_tmp)
        s_tmp = s.copy()
        s_tmp[0, :] = MISSING
        genotypes.append(s_tmp)

        m = ts.get_num_sites()
        n = H.shape[1]

        G = np.zeros((m, n, n))
        for i in range(m):
            G[i, :, :] = np.add.outer(H[i, :], H[i, :])

        return H, G, genotypes

    def example_parameters_genotypes(self, ts, seed=42):
        np.random.seed(seed)
        H, G, genotypes = self.example_genotypes(ts)
        n = H.shape[1]
        m = ts.get_num_sites()

        # Here we have equal mutation and recombination
        r = np.zeros(m) + 0.01
        mu = np.zeros(m) + 0.01
        r[0] = 0

        e = self.genotype_emission(mu, m)

        for s in genotypes:
            yield n, m, G, s, e, r

        # Mixture of random and extremes
        rs = [np.zeros(m) + 0.999, np.zeros(m) + 1e-6, np.random.rand(m)]
        mus = [np.zeros(m) + 0.33, np.zeros(m) + 1e-6, np.random.rand(m) * 0.33]

        # e = self.genotype_emission(mu, m)

        for s, r, mu in itertools.product(genotypes, rs, mus):
            r[0] = 0
            e = self.genotype_emission(mu, m)
            yield n, m, G, s, e, r

    def example_parameters_genotypes_larger(
        self, ts, seed=42, mean_r=1e-5, mean_mu=1e-5
    ):
        np.random.seed(seed)
        H, G, genotypes = self.example_genotypes(ts)

        m = ts.get_num_sites()
        n = H.shape[1]

        r = mean_r * np.ones(m) * ((np.random.rand(m) + 0.5) / 2)
        r[0] = 0

        # Error probability
        mu = mean_mu * np.ones(m) * ((np.random.rand(m) + 0.5) / 2)

        # Define the emission probability matrix
        e = self.genotype_emission(mu, m)

        for s in genotypes:
            yield n, m, G, s, e, r

    def assertAllClose(self, A, B):
        """Assert that all entries of two matrices are 'close'"""
        # assert np.allclose(A, B, rtol=1e-9, atol=0.0)
        assert np.allclose(A, B, rtol=1e-09, atol=1e-08)

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
            e_sv = e_vs.T
            H_sv = H_vs.T

            # variants x samples
            F_vs, c_vs, ll_vs = fbh_vs.forwards_ls_hap(
                n, m, H_vs, s, e_vs, r, norm=False
            )
            B_vs = fbh_vs.backwards_ls_hap(n, m, H_vs, s, e_vs, c_vs, r)
            self.assertAllClose(np.log10(np.sum(F_vs * B_vs, 1)), ll_vs * np.ones(m))
            F_tmp, c_tmp, ll_tmp = fbh_vs.forwards_ls_hap(
                n, m, H_vs, s, e_vs, r, norm=True
            )
            B_tmp = fbh_vs.backwards_ls_hap(n, m, H_vs, s, e_vs, c_tmp, r)
            self.assertAllClose(ll_vs, ll_tmp)
            self.assertAllClose(np.sum(F_tmp * B_tmp, 1), np.ones(m))

            # samples x variants
            F_sv, c_sv, ll_sv = fbh_sv.forwards_ls_hap(
                n, m, H_sv, s, e_sv, r, norm=False
            )
            B_sv = fbh_sv.backwards_ls_hap(n, m, H_sv, s, e_sv, c_sv, r)
            self.assertAllClose(np.log10(np.sum(F_sv * B_sv, 0)), ll_sv * np.ones(m))
            F_tmp, c_tmp, ll_tmp = fbh_sv.forwards_ls_hap(
                n, m, H_sv, s, e_sv, r, norm=True
            )
            B_tmp = fbh_sv.backwards_ls_hap(n, m, H_sv, s, e_sv, c_tmp, r)
            self.assertAllClose(ll_sv, ll_tmp)
            self.assertAllClose(np.sum(F_tmp * B_tmp, 0), np.ones(m))

            # samples x variants agrees with variants x samples
            self.assertAllClose(ll_vs, ll_sv)

    def verify_larger(self, ts):
        # variants x samples
        for n, m, H_vs, s, e_vs, r in self.example_parameters_haplotypes_larger(ts):
            e_sv = e_vs.T
            H_sv = H_vs.T

            F_vs, c_vs, ll_vs = fbh_vs.forwards_ls_hap(
                n, m, H_vs, s, e_vs, r, norm=False
            )
            B_vs = fbh_vs.backwards_ls_hap(n, m, H_vs, s, e_vs, c_vs, r)
            self.assertAllClose(np.log10(np.sum(F_vs * B_vs, 1)), ll_vs * np.ones(m))
            F_tmp, c_tmp, ll_tmp = fbh_vs.forwards_ls_hap(
                n, m, H_vs, s, e_vs, r, norm=True
            )
            B_tmp = fbh_vs.backwards_ls_hap(n, m, H_vs, s, e_vs, c_tmp, r)
            self.assertAllClose(ll_vs, ll_tmp)
            self.assertAllClose(np.sum(F_tmp * B_tmp, 1), np.ones(m))

            # samples x variants
            F_sv, c_sv, ll_sv = fbh_sv.forwards_ls_hap(
                n, m, H_sv, s, e_sv, r, norm=False
            )
            B_sv = fbh_sv.backwards_ls_hap(n, m, H_sv, s, e_sv, c_sv, r)
            self.assertAllClose(np.log10(np.sum(F_sv * B_sv, 0)), ll_sv * np.ones(m))
            F_tmp, c_tmp, ll_tmp = fbh_sv.forwards_ls_hap(
                n, m, H_sv, s, e_sv, r, norm=True
            )
            B_tmp = fbh_sv.backwards_ls_hap(n, m, H_sv, s, e_sv, c_tmp, r)
            self.assertAllClose(ll_sv, ll_tmp)
            self.assertAllClose(np.sum(F_tmp * B_tmp, 0), np.ones(m))

            # samples x variants agrees with variants x samples
            self.assertAllClose(ll_vs, ll_sv)


class TestNonTreeMethodsDip(FBAlgorithmBase):
    """Test that we compute the sample likelihoods across all implementations."""

    def verify(self, ts):
        for n, m, G_vs, s, e_vs, r in self.example_parameters_genotypes(ts):
            e_sv = e_vs.T
            G_sv = G_vs.T

            # variants x samples
            F_vs, c_vs, ll_vs = fbd_vs.forwards_ls_dip(
                n, m, G_vs, s, e_vs, r, norm=True
            )
            B_vs = fbd_vs.backwards_ls_dip(n, m, G_vs, s, e_vs, c_vs, r)
            self.assertAllClose(np.sum(F_vs * B_vs, (1, 2)), np.ones(m))
            F_tmp, c_tmp, ll_tmp = fbd_vs.forwards_ls_dip(
                n, m, G_vs, s, e_vs, r, norm=False
            )
            if ll_tmp != -np.inf:
                B_tmp = fbd_vs.backwards_ls_dip(n, m, G_vs, s, e_vs, c_tmp, r)
                self.assertAllClose(ll_vs, ll_tmp)
                self.assertAllClose(
                    np.log10(np.sum(F_tmp * B_tmp, (1, 2))), ll_tmp * np.ones(m)
                )

            F_tmp, ll_tmp = fbd_vs.forward_ls_dip_starting_point(n, m, G_vs, s, e_vs, r)
            if ll_tmp != -np.inf:
                B_tmp = fbd_vs.backward_ls_dip_starting_point(n, m, G_vs, s, e_vs, r)
                self.assertAllClose(ll_vs, ll_tmp)
                self.assertAllClose(
                    np.log10(np.sum(F_tmp * B_tmp, (1, 2))), ll_tmp * np.ones(m)
                )

            F_tmp, c_tmp, ll_tmp = fbd_vs.forward_ls_dip_loop(
                n, m, G_vs, s, e_vs, r, norm=False
            )
            if ll_tmp != -np.inf:
                B_tmp = fbd_vs.backward_ls_dip_loop(n, m, G_vs, s, e_vs, c_tmp, r)
                self.assertAllClose(ll_vs, ll_tmp)
                self.assertAllClose(
                    np.log10(np.sum(F_tmp * B_tmp, (1, 2))), ll_tmp * np.ones(m)
                )

            F_tmp, c_tmp, ll_tmp = fbd_vs.forward_ls_dip_loop(
                n, m, G_vs, s, e_vs, r, norm=True
            )
            B_tmp = fbd_vs.backward_ls_dip_loop(n, m, G_vs, s, e_vs, c_tmp, r)
            self.assertAllClose(ll_vs, ll_tmp)
            self.assertAllClose(np.sum(F_tmp * B_tmp, (1, 2)), np.ones(m))

            # samples x variants
            F_sv, c_sv, ll_sv = fbd_sv.forwards_ls_dip(
                n, m, G_sv, s, e_sv, r, norm=True
            )
            B_sv = fbd_sv.backwards_ls_dip(n, m, G_sv, s, e_sv, c_sv, r)
            self.assertAllClose(np.sum(F_sv * B_sv, (0, 1)), np.ones(m))
            F_tmp, c_tmp, ll_tmp = fbd_sv.forwards_ls_dip(
                n, m, G_sv, s, e_sv, r, norm=False
            )
            if ll_tmp != -np.inf:
                B_tmp = fbd_sv.backwards_ls_dip(n, m, G_sv, s, e_sv, c_tmp, r)
                self.assertAllClose(
                    np.log10(np.sum(F_tmp * B_tmp, (0, 1))), ll_tmp * np.ones(m)
                )
                self.assertAllClose(ll_sv, ll_tmp)

            F_tmp, ll_tmp = fbd_sv.forward_ls_dip_starting_point(n, m, G_sv, s, e_sv, r)
            if ll_tmp != -np.inf:
                B_tmp = fbd_sv.backward_ls_dip_starting_point(n, m, G_sv, s, e_sv, r)
                self.assertAllClose(ll_sv, ll_tmp)
                self.assertAllClose(
                    np.log10(np.sum(F_tmp * B_tmp, (0, 1))), ll_tmp * np.ones(m)
                )

            F_tmp, c_tmp, ll_tmp = fbd_sv.forward_ls_dip_loop(
                n, m, G_sv, s, e_sv, r, norm=True
            )
            B_tmp = fbd_sv.backward_ls_dip_loop(n, m, G_sv, s, e_sv, c_tmp, r)
            self.assertAllClose(ll_sv, ll_tmp)
            self.assertAllClose(np.sum(F_tmp * B_tmp, (0, 1)), np.ones(m))
            F_tmp, c_tmp, ll_tmp = fbd_sv.forward_ls_dip_loop(
                n, m, G_sv, s, e_sv, r, norm=False
            )
            if ll_tmp != -np.inf:
                B_tmp = fbd_sv.backward_ls_dip_loop(n, m, G_sv, s, e_sv, c_tmp, r)
                self.assertAllClose(ll_sv, ll_tmp)
                self.assertAllClose(
                    np.log10(np.sum(F_tmp * B_tmp, (0, 1))), ll_tmp * np.ones(m)
                )

            # compare sample x variants to variants x samples
            self.assertAllClose(ll_vs, ll_sv)

    def verify_larger(self, ts):
        # variants x samples
        for n, m, G_vs, s, e_vs, r in self.example_parameters_genotypes_larger(ts):

            e_sv = e_vs.T
            G_sv = G_vs.T

            # variants x samples
            F_vs, c_vs, ll_vs = fbd_vs.forwards_ls_dip(
                n, m, G_vs, s, e_vs, r, norm=True
            )
            B_vs = fbd_vs.backwards_ls_dip(n, m, G_vs, s, e_vs, c_vs, r)
            self.assertAllClose(np.sum(F_vs * B_vs, (1, 2)), np.ones(m))
            F_tmp, c_tmp, ll_tmp = fbd_vs.forwards_ls_dip(
                n, m, G_vs, s, e_vs, r, norm=False
            )
            if ll_tmp != -np.inf:
                B_tmp = fbd_vs.backwards_ls_dip(n, m, G_vs, s, e_vs, c_tmp, r)
                self.assertAllClose(ll_vs, ll_tmp)
                self.assertAllClose(
                    np.log10(np.sum(F_tmp * B_tmp, (1, 2))), ll_tmp * np.ones(m)
                )

            F_tmp, ll_tmp = fbd_vs.forward_ls_dip_starting_point(n, m, G_vs, s, e_vs, r)
            if ll_tmp != -np.inf:
                B_tmp = fbd_vs.backward_ls_dip_starting_point(n, m, G_vs, s, e_vs, r)
                self.assertAllClose(ll_vs, ll_tmp)
                self.assertAllClose(
                    np.log10(np.sum(F_tmp * B_tmp, (1, 2))), ll_tmp * np.ones(m)
                )

            F_tmp, c_tmp, ll_tmp = fbd_vs.forward_ls_dip_loop(
                n, m, G_vs, s, e_vs, r, norm=False
            )
            if ll_tmp != -np.inf:
                B_tmp = fbd_vs.backward_ls_dip_loop(n, m, G_vs, s, e_vs, c_tmp, r)
                self.assertAllClose(ll_vs, ll_tmp)
                self.assertAllClose(
                    np.log10(np.sum(F_tmp * B_tmp, (1, 2))), ll_tmp * np.ones(m)
                )
            F_tmp, c_tmp, ll_tmp = fbd_vs.forward_ls_dip_loop(
                n, m, G_vs, s, e_vs, r, norm=True
            )
            B_tmp = fbd_vs.backward_ls_dip_loop(n, m, G_vs, s, e_vs, c_tmp, r)
            self.assertAllClose(ll_vs, ll_tmp)
            self.assertAllClose(np.sum(F_tmp * B_tmp, (1, 2)), np.ones(m))

            # samples x variants

            F_sv, c_sv, ll_sv = fbd_sv.forwards_ls_dip(
                n, m, G_sv, s, e_sv, r, norm=True
            )
            B_sv = fbd_sv.backwards_ls_dip(n, m, G_sv, s, e_sv, c_sv, r)
            self.assertAllClose(np.sum(F_sv * B_sv, (0, 1)), np.ones(m))
            F_tmp, c_tmp, ll_tmp = fbd_sv.forwards_ls_dip(
                n, m, G_sv, s, e_sv, r, norm=False
            )
            if ll_tmp != -np.inf:
                B_tmp = fbd_sv.backwards_ls_dip(n, m, G_sv, s, e_sv, c_tmp, r)
                self.assertAllClose(
                    np.log10(np.sum(F_tmp * B_tmp, (0, 1))), ll_tmp * np.ones(m)
                )
                self.assertAllClose(ll_sv, ll_tmp)

            F_tmp, ll_tmp = fbd_sv.forward_ls_dip_starting_point(n, m, G_sv, s, e_sv, r)
            if ll_tmp != -np.inf:
                B_tmp = fbd_sv.backward_ls_dip_starting_point(n, m, G_sv, s, e_sv, r)
                self.assertAllClose(ll_sv, ll_tmp)
                self.assertAllClose(
                    np.log10(np.sum(F_tmp * B_tmp, (0, 1))), ll_tmp * np.ones(m)
                )

            F_tmp, c_tmp, ll_tmp = fbd_sv.forward_ls_dip_loop(
                n, m, G_sv, s, e_sv, r, norm=True
            )
            B_tmp = fbd_sv.backward_ls_dip_loop(n, m, G_sv, s, e_sv, c_tmp, r)
            self.assertAllClose(ll_sv, ll_tmp)
            self.assertAllClose(np.sum(F_tmp * B_tmp, (0, 1)), np.ones(m))
            F_tmp, c_tmp, ll_tmp = fbd_sv.forward_ls_dip_loop(
                n, m, G_sv, s, e_sv, r, norm=False
            )
            if ll_tmp != -np.inf:
                B_tmp = fbd_sv.backward_ls_dip_loop(n, m, G_sv, s, e_sv, c_tmp, r)
                self.assertAllClose(ll_sv, ll_tmp)
                self.assertAllClose(
                    np.log10(np.sum(F_tmp * B_tmp, (0, 1))), ll_tmp * np.ones(m)
                )

            # compare sample x variants to variants x samples
            self.assertAllClose(ll_vs, ll_sv)


class VitAlgorithmBase(LSBase):
    """Base for viterbi algoritm tests."""


class TestNonTreeViterbiHap(VitAlgorithmBase):
    """Test that we have the same log-likelihood across all implementations"""

    def verify(self, ts):
        for n, m, H_vs, s, e_vs, r in self.example_parameters_haplotypes(ts):
            e_sv = e_vs.T
            H_sv = H_vs.T

            # variants x samples
            V_vs, P_vs, ll_vs = vh_vs.forwards_viterbi_hap_naive(n, m, H_vs, s, e_vs, r)
            path_vs = vh_vs.backwards_viterbi_hap(m, V_vs[m - 1, :], P_vs)
            ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_vs, s, e_vs, r)
            self.assertAllClose(ll_vs, ll_check)
            V_tmp, P_tmp, ll_tmp = vh_vs.forwards_viterbi_hap_naive_vec(
                n, m, H_vs, s, e_vs, r
            )
            path_tmp = vh_vs.backwards_viterbi_hap(m, V_tmp[m - 1, :], P_tmp)
            ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)
            V_tmp, P_tmp, ll_tmp = vh_vs.forwards_viterbi_hap_naive_low_mem(
                n, m, H_vs, s, e_vs, r
            )
            path_tmp = vh_vs.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)
            V_tmp, P_tmp, ll_tmp = vh_vs.forwards_viterbi_hap_naive_low_mem_rescaling(
                n, m, H_vs, s, e_vs, r
            )
            path_tmp = vh_vs.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)
            V_tmp, P_tmp, ll_tmp = vh_vs.forwards_viterbi_hap_low_mem_rescaling(
                n, m, H_vs, s, e_vs, r
            )
            path_tmp = vh_vs.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)
            V_tmp, P_tmp, ll_tmp = vh_vs.forwards_viterbi_hap_lower_mem_rescaling(
                n, m, H_vs, s, e_vs, r
            )
            path_tmp = vh_vs.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            (
                V_tmp,
                V_argmaxes_tmp,
                recombs,
                ll_tmp,
            ) = vh_vs.forwards_viterbi_hap_lower_mem_rescaling_no_pointer(
                n, m, H_vs, s, e_vs, r
            )
            path_tmp = vh_vs.backwards_viterbi_hap_no_pointer(
                m,
                V_argmaxes_tmp,
                nb.typed.List(recombs),
            )
            ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            # samples x variants
            V_sv, P_sv, ll_sv = vh_sv.forwards_viterbi_hap_naive(n, m, H_sv, s, e_sv, r)
            path_tmp = vh_sv.backwards_viterbi_hap(m, V_sv[:, m - 1], P_sv)
            ll_check = vh_sv.path_ll_hap(n, m, H_sv, path_tmp, s, e_sv, r)
            self.assertAllClose(ll_sv, ll_check)
            V_tmp, P_tmp, ll_tmp = vh_sv.forwards_viterbi_hap_naive_vec(
                n, m, H_sv, s, e_sv, r
            )
            path_tmp = vh_sv.backwards_viterbi_hap(m, V_tmp[:, m - 1], P_tmp)
            ll_check = vh_sv.path_ll_hap(n, m, H_sv, path_tmp, s, e_sv, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_sv, ll_tmp)
            V_tmp, P_tmp, ll_tmp = vh_sv.forwards_viterbi_hap_naive_low_mem(
                n, m, H_sv, s, e_sv, r
            )
            path_tmp = vh_sv.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh_sv.path_ll_hap(n, m, H_sv, path_tmp, s, e_sv, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_sv, ll_tmp)
            V_tmp, P_tmp, ll_tmp = vh_sv.forwards_viterbi_hap_naive_low_mem_rescaling(
                n, m, H_sv, s, e_sv, r
            )
            path_tmp = vh_sv.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh_sv.path_ll_hap(n, m, H_sv, path_tmp, s, e_sv, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_sv, ll_tmp)
            V_tmp, P_tmp, ll_tmp = vh_sv.forwards_viterbi_hap_low_mem_rescaling(
                n, m, H_sv, s, e_sv, r
            )
            path_tmp = vh_sv.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh_sv.path_ll_hap(n, m, H_sv, path_tmp, s, e_sv, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_sv, ll_tmp)
            V_tmp, P_tmp, ll_tmp = vh_sv.forwards_viterbi_hap_lower_mem_rescaling(
                n, m, H_sv, s, e_sv, r
            )
            path_tmp = vh_sv.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh_sv.path_ll_hap(n, m, H_sv, path_tmp, s, e_sv, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            # samples x variants agrees with variants x samples
            self.assertAllClose(ll_vs, ll_sv)

    def verify_larger(self, ts):
        for n, m, H_vs, s, e_vs, r in self.example_parameters_haplotypes_larger(ts):
            e_sv = e_vs.T
            H_sv = H_vs.T

            # variants x samples
            V_vs, P_vs, ll_vs = vh_vs.forwards_viterbi_hap_naive(n, m, H_vs, s, e_vs, r)
            path_vs = vh_vs.backwards_viterbi_hap(m, V_vs[m - 1, :], P_vs)
            ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_vs, s, e_vs, r)
            self.assertAllClose(ll_vs, ll_check)
            V_tmp, P_tmp, ll_tmp = vh_vs.forwards_viterbi_hap_naive_vec(
                n, m, H_vs, s, e_vs, r
            )
            path_tmp = vh_vs.backwards_viterbi_hap(m, V_tmp[m - 1, :], P_tmp)
            ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)
            V_tmp, P_tmp, ll_tmp = vh_vs.forwards_viterbi_hap_naive_low_mem(
                n, m, H_vs, s, e_vs, r
            )
            path_tmp = vh_vs.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)
            V_tmp, P_tmp, ll_tmp = vh_vs.forwards_viterbi_hap_naive_low_mem_rescaling(
                n, m, H_vs, s, e_vs, r
            )
            path_tmp = vh_vs.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)
            V_tmp, P_tmp, ll_tmp = vh_vs.forwards_viterbi_hap_low_mem_rescaling(
                n, m, H_vs, s, e_vs, r
            )
            path_tmp = vh_vs.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)
            V_tmp, P_tmp, ll_tmp = vh_vs.forwards_viterbi_hap_lower_mem_rescaling(
                n, m, H_vs, s, e_vs, r
            )
            path_tmp = vh_vs.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            (
                V_tmp,
                V_argmaxes_tmp,
                recombs,
                ll_tmp,
            ) = vh_vs.forwards_viterbi_hap_lower_mem_rescaling_no_pointer(
                n, m, H_vs, s, e_vs, r
            )
            path_tmp = vh_vs.backwards_viterbi_hap_no_pointer(
                m, V_argmaxes_tmp, nb.typed.List(recombs)
            )
            ll_check = vh_vs.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            # samples x variants
            V_sv, P_sv, ll_sv = vh_sv.forwards_viterbi_hap_naive(n, m, H_sv, s, e_sv, r)
            path_tmp = vh_sv.backwards_viterbi_hap(m, V_sv[:, m - 1], P_sv)
            ll_check = vh_sv.path_ll_hap(n, m, H_sv, path_tmp, s, e_sv, r)
            self.assertAllClose(ll_sv, ll_check)
            V_tmp, P_tmp, ll_tmp = vh_sv.forwards_viterbi_hap_naive_vec(
                n, m, H_sv, s, e_sv, r
            )
            path_tmp = vh_sv.backwards_viterbi_hap(m, V_tmp[:, m - 1], P_tmp)
            ll_check = vh_sv.path_ll_hap(n, m, H_sv, path_tmp, s, e_sv, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_sv, ll_tmp)
            V_tmp, P_tmp, ll_tmp = vh_sv.forwards_viterbi_hap_naive_low_mem(
                n, m, H_sv, s, e_sv, r
            )
            path_tmp = vh_sv.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh_sv.path_ll_hap(n, m, H_sv, path_tmp, s, e_sv, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_sv, ll_tmp)
            V_tmp, P_tmp, ll_tmp = vh_sv.forwards_viterbi_hap_naive_low_mem_rescaling(
                n, m, H_sv, s, e_sv, r
            )
            path_tmp = vh_sv.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh_sv.path_ll_hap(n, m, H_sv, path_tmp, s, e_sv, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_sv, ll_tmp)
            V_tmp, P_tmp, ll_tmp = vh_sv.forwards_viterbi_hap_low_mem_rescaling(
                n, m, H_sv, s, e_sv, r
            )
            path_tmp = vh_sv.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh_sv.path_ll_hap(n, m, H_sv, path_tmp, s, e_sv, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_sv, ll_tmp)
            V_tmp, P_tmp, ll_tmp = vh_sv.forwards_viterbi_hap_lower_mem_rescaling(
                n, m, H_sv, s, e_sv, r
            )
            path_tmp = vh_sv.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh_sv.path_ll_hap(n, m, H_sv, path_tmp, s, e_sv, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            # samples x variants agrees with variants x samples
            self.assertAllClose(ll_vs, ll_sv)


class TestNonTreeViterbiDip(VitAlgorithmBase):
    """Test that we have the same log-likelihood across all implementations"""

    def verify(self, ts):
        for n, m, G_vs, s, e_vs, r in self.example_parameters_genotypes(ts):
            e_sv = e_vs.T
            G_sv = G_vs.T

            # variants x samples
            V_vs, P_vs, ll_vs = vd_vs.forwards_viterbi_dip_naive(n, m, G_vs, s, e_vs, r)
            path_vs = vd_vs.backwards_viterbi_dip(m, V_vs[m - 1, :, :], P_vs)
            phased_path_vs = vd_vs.get_phased_path(n, path_vs)
            path_ll_vs = vd_vs.path_ll_dip(n, m, G_vs, phased_path_vs, s, e_vs, r)
            self.assertAllClose(ll_vs, path_ll_vs)

            V_tmp, P_tmp, ll_tmp = vd_vs.forwards_viterbi_dip_naive_low_mem(
                n, m, G_vs, s, e_vs, r
            )
            path_tmp = vd_vs.backwards_viterbi_dip(m, V_tmp, P_tmp)
            phased_path_tmp = vd_vs.get_phased_path(n, path_tmp)
            path_ll_tmp = vd_vs.path_ll_dip(n, m, G_vs, phased_path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, path_ll_tmp)
            self.assertAllClose(ll_vs, ll_tmp)

            V_tmp, P_tmp, ll_tmp = vd_vs.forwards_viterbi_dip_low_mem(
                n, m, G_vs, s, e_vs, r
            )
            path_tmp = vd_vs.backwards_viterbi_dip(m, V_tmp, P_tmp)
            phased_path_tmp = vd_vs.get_phased_path(n, path_tmp)
            path_ll_tmp = vd_vs.path_ll_dip(n, m, G_vs, phased_path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, path_ll_tmp)
            self.assertAllClose(ll_vs, ll_tmp)

            (
                V_tmp,
                V_argmaxes_tmp,
                V_rowcol_maxes_tmp,
                V_rowcol_argmaxes_tmp,
                recombs_single,
                recombs_double,
                # P_tmp,
                ll_tmp,
            ) = vd_vs.forwards_viterbi_dip_low_mem_no_pointer(n, m, G_vs, s, e_vs, r)
            path_tmp = vd_vs.backwards_viterbi_dip_no_pointer(
                m,
                V_argmaxes_tmp,
                V_rowcol_maxes_tmp,
                V_rowcol_argmaxes_tmp,
                nb.typed.List(recombs_single),
                nb.typed.List(recombs_double),
                V_tmp,
            )
            phased_path_tmp = vd_vs.get_phased_path(n, path_tmp)
            path_ll_tmp = vd_vs.path_ll_dip(n, m, G_vs, phased_path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, path_ll_tmp)
            self.assertAllClose(ll_vs, ll_tmp)

            V_tmp, P_tmp, ll_tmp = vd_vs.forwards_viterbi_dip_naive_vec(
                n, m, G_vs, s, e_vs, r
            )
            path_tmp = vd_vs.backwards_viterbi_dip(m, V_tmp[m - 1, :, :], P_tmp)
            phased_path_tmp = vd_vs.get_phased_path(n, path_tmp)
            path_ll_tmp = vd_vs.path_ll_dip(n, m, G_vs, phased_path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, path_ll_tmp)
            self.assertAllClose(ll_vs, ll_tmp)

            # samples x variants
            V_sv, P_sv, ll_sv = vd_sv.forwards_viterbi_dip_naive(n, m, G_sv, s, e_sv, r)
            path_sv = vd_sv.backwards_viterbi_dip(m, V_sv[:, :, m - 1], P_sv)
            phased_path_sv = vd_sv.get_phased_path(n, path_sv)
            path_ll_sv = vd_sv.path_ll_dip(n, m, G_sv, phased_path_sv, s, e_sv, r)
            self.assertAllClose(ll_sv, path_ll_sv)

            V_tmp, P_tmp, ll_tmp = vd_sv.forwards_viterbi_dip_naive_low_mem(
                n, m, G_sv, s, e_sv, r
            )
            path_tmp = vd_sv.backwards_viterbi_dip(m, V_tmp, P_tmp)
            phased_path_tmp = vd_sv.get_phased_path(n, path_tmp)
            path_ll_tmp = vd_sv.path_ll_dip(n, m, G_sv, phased_path_tmp, s, e_sv, r)
            self.assertAllClose(ll_tmp, path_ll_tmp)
            self.assertAllClose(ll_sv, ll_tmp)

            V_tmp, P_tmp, ll_tmp = vd_sv.forwards_viterbi_dip_low_mem(
                n, m, G_sv, s, e_sv, r
            )
            path_tmp = vd_sv.backwards_viterbi_dip(m, V_tmp, P_tmp)
            phased_path_tmp = vd_sv.get_phased_path(n, path_tmp)
            path_ll_tmp = vd_sv.path_ll_dip(n, m, G_sv, phased_path_tmp, s, e_sv, r)
            self.assertAllClose(ll_tmp, path_ll_tmp)
            self.assertAllClose(ll_sv, ll_tmp)

            V_tmp, P_tmp, ll_tmp = vd_sv.forwards_viterbi_dip_naive_vec(
                n, m, G_sv, s, e_sv, r
            )
            path_tmp = vd_sv.backwards_viterbi_dip(m, V_tmp[:, :, m - 1], P_tmp)
            phased_path_tmp = vd_sv.get_phased_path(n, path_tmp)
            path_ll_tmp = vd_sv.path_ll_dip(n, m, G_sv, phased_path_tmp, s, e_sv, r)
            self.assertAllClose(ll_tmp, path_ll_tmp)
            self.assertAllClose(ll_sv, ll_tmp)
            self.assertAllClose(path_ll_sv, path_ll_vs)

            # samples x variants agrees with variants x samples
            self.assertAllClose(ll_vs, ll_sv)

    def verify_larger(self, ts):
        for n, m, G_vs, s, e_vs, r in self.example_parameters_genotypes_larger(ts):
            e_sv = e_vs.T
            G_sv = G_vs.T

            # variants x samples
            V_vs, P_vs, ll_vs = vd_vs.forwards_viterbi_dip_naive(n, m, G_vs, s, e_vs, r)
            path_vs = vd_vs.backwards_viterbi_dip(m, V_vs[m - 1, :, :], P_vs)
            phased_path_vs = vd_vs.get_phased_path(n, path_vs)
            path_ll_vs = vd_vs.path_ll_dip(n, m, G_vs, phased_path_vs, s, e_vs, r)
            self.assertAllClose(ll_vs, path_ll_vs)

            V_tmp, P_tmp, ll_tmp = vd_vs.forwards_viterbi_dip_naive_low_mem(
                n, m, G_vs, s, e_vs, r
            )
            path_tmp = vd_vs.backwards_viterbi_dip(m, V_tmp, P_tmp)
            phased_path_tmp = vd_vs.get_phased_path(n, path_tmp)
            path_ll_tmp = vd_vs.path_ll_dip(n, m, G_vs, phased_path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, path_ll_tmp)
            self.assertAllClose(ll_vs, ll_tmp)

            V_tmp, P_tmp, ll_tmp = vd_vs.forwards_viterbi_dip_low_mem(
                n, m, G_vs, s, e_vs, r
            )
            path_tmp = vd_vs.backwards_viterbi_dip(m, V_tmp, P_tmp)
            phased_path_tmp = vd_vs.get_phased_path(n, path_tmp)
            path_ll_tmp = vd_vs.path_ll_dip(n, m, G_vs, phased_path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, path_ll_tmp)
            self.assertAllClose(ll_vs, ll_tmp)

            (
                V_tmp,
                V_argmaxes_tmp,
                V_rowcol_maxes_tmp,
                V_rowcol_argmaxes_tmp,
                recombs_single,
                recombs_double,
                ll_tmp,
            ) = vd_vs.forwards_viterbi_dip_low_mem_no_pointer(n, m, G_vs, s, e_vs, r)
            path_tmp = vd_vs.backwards_viterbi_dip_no_pointer(
                m,
                V_argmaxes_tmp,
                V_rowcol_maxes_tmp,
                V_rowcol_argmaxes_tmp,
                nb.typed.List(recombs_single),
                nb.typed.List(recombs_double),
                V_tmp,
            )
            phased_path_tmp = vd_vs.get_phased_path(n, path_tmp)
            path_ll_tmp = vd_vs.path_ll_dip(n, m, G_vs, phased_path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, path_ll_tmp)
            self.assertAllClose(ll_vs, ll_tmp)

            V_tmp, P_tmp, ll_tmp = vd_vs.forwards_viterbi_dip_naive_vec(
                n, m, G_vs, s, e_vs, r
            )
            path_tmp = vd_vs.backwards_viterbi_dip(m, V_tmp[m - 1, :, :], P_tmp)
            phased_path_tmp = vd_vs.get_phased_path(n, path_tmp)
            path_ll_tmp = vd_vs.path_ll_dip(n, m, G_vs, phased_path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, path_ll_tmp)
            self.assertAllClose(ll_vs, ll_tmp)

            # samples x variants
            V_sv, P_sv, ll_sv = vd_sv.forwards_viterbi_dip_naive(n, m, G_sv, s, e_sv, r)
            path_sv = vd_sv.backwards_viterbi_dip(m, V_sv[:, :, m - 1], P_sv)
            phased_path_sv = vd_sv.get_phased_path(n, path_sv)
            path_ll_sv = vd_sv.path_ll_dip(n, m, G_sv, phased_path_sv, s, e_sv, r)
            self.assertAllClose(ll_sv, path_ll_sv)

            V_tmp, P_tmp, ll_tmp = vd_sv.forwards_viterbi_dip_naive_low_mem(
                n, m, G_sv, s, e_sv, r
            )
            path_tmp = vd_sv.backwards_viterbi_dip(m, V_tmp, P_tmp)
            phased_path_tmp = vd_sv.get_phased_path(n, path_tmp)
            path_ll_tmp = vd_sv.path_ll_dip(n, m, G_sv, phased_path_tmp, s, e_sv, r)
            self.assertAllClose(ll_tmp, path_ll_tmp)
            self.assertAllClose(ll_sv, ll_tmp)

            V_tmp, P_tmp, ll_tmp = vd_sv.forwards_viterbi_dip_low_mem(
                n, m, G_sv, s, e_sv, r
            )
            path_tmp = vd_sv.backwards_viterbi_dip(m, V_tmp, P_tmp)
            phased_path_tmp = vd_sv.get_phased_path(n, path_tmp)
            path_ll_tmp = vd_sv.path_ll_dip(n, m, G_sv, phased_path_tmp, s, e_sv, r)
            self.assertAllClose(ll_tmp, path_ll_tmp)
            self.assertAllClose(ll_sv, ll_tmp)

            V_tmp, P_tmp, ll_tmp = vd_sv.forwards_viterbi_dip_naive_vec(
                n, m, G_sv, s, e_sv, r
            )
            path_tmp = vd_sv.backwards_viterbi_dip(m, V_tmp[:, :, m - 1], P_tmp)
            phased_path_tmp = vd_sv.get_phased_path(n, path_tmp)
            path_ll_tmp = vd_sv.path_ll_dip(n, m, G_sv, phased_path_tmp, s, e_sv, r)
            self.assertAllClose(ll_tmp, path_ll_tmp)
            self.assertAllClose(ll_sv, ll_tmp)

            # samples x variants agrees with variants x samples
            self.assertAllClose(ll_vs, ll_sv)
