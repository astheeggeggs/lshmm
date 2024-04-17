# Simulation
import itertools

# Python libraries
import msprime
import numpy as np
import pytest
import tskit

import lshmm as ls
import lshmm.forward_backward.fb_diploid as fbd
import lshmm.forward_backward.fb_haploid as fbh
import lshmm.vit_diploid as vd
import lshmm.vit_haploid as vh

EQUAL_BOTH_HOM = 4
UNEQUAL_BOTH_HOM = 0
BOTH_HET = 7
REF_HOM_OBS_HET = 1
REF_HET_OBS_HOM = 2

MISSING = -1
MISSING_INDEX = 3


class LSBase:
    """Superclass of Li and Stephens tests."""

    def example_haplotypes(self, ts, seed=42):
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

    def haplotype_emission(self, mu, m, n_alleles, scale_mutation_based_on_n_alleles):
        # Define the emission probability matrix
        e = np.zeros((m, 2))
        if isinstance(mu, float):
            mu = mu * np.ones(m)

        if scale_mutation_based_on_n_alleles:
            e[:, 0] = mu - mu * np.equal(
                n_alleles, np.ones(m)
            )  # Added boolean in case we're at an invariant site
            e[:, 1] = 1 - (n_alleles - 1) * mu
        else:
            for j in range(m):
                if n_alleles[j] == 1:  # In case we're at an invariant site
                    e[j, 0] = 0
                    e[j, 1] = 1
                else:
                    e[j, 0] = mu[j] / (n_alleles[j] - 1)
                    e[j, 1] = 1 - mu[j]
        return e

    def genotype_emission(self, mu, m):
        # Define the emission probability matrix
        e = np.zeros((m, 8))
        e[:, EQUAL_BOTH_HOM] = (1 - mu) ** 2
        e[:, UNEQUAL_BOTH_HOM] = mu**2
        e[:, BOTH_HET] = (1 - mu) ** 2 + mu**2
        e[:, REF_HOM_OBS_HET] = 2 * mu * (1 - mu)
        e[:, REF_HET_OBS_HOM] = mu * (1 - mu)
        e[:, MISSING_INDEX] = 1

        return e

    def example_parameters_haplotypes(self, ts, seed=42, scale_mutation=True):
        """Returns an iterator over combinations of haplotype, recombination and
        mutation probabilities."""
        np.random.seed(seed)
        H, haplotypes = self.example_haplotypes(ts)
        n = H.shape[1]
        m = ts.get_num_sites()

        def _get_num_alleles(ref_haps, query):
            assert ref_haps.shape[0] == query.shape[1]
            num_sites = ref_haps.shape[0]
            num_alleles = np.zeros(num_sites, dtype=np.int8)
            exclusion_set = np.array([MISSING])
            for i in range(num_sites):
                uniq_alleles = np.unique(np.append(ref_haps[i, :], query[:, i]))
                num_alleles[i] = np.sum(~np.isin(uniq_alleles, exclusion_set))
            assert np.all(num_alleles >= 0), "Number of alleles cannot be zero."
            return num_alleles

        # Here we have equal mutation and recombination
        r = np.zeros(m) + 0.01
        mu = np.zeros(m) + 0.01
        r[0] = 0

        for s in haplotypes:
            n_alleles = _get_num_alleles(H, s)
            e = self.haplotype_emission(
                mu, m, n_alleles, scale_mutation_based_on_n_alleles=scale_mutation
            )
            yield n, m, H, s, e, r, mu

        # Mixture of random and extremes
        rs = [np.zeros(m) + 0.999, np.zeros(m) + 1e-6, np.random.rand(m)]
        mus = [np.zeros(m) + 0.33, np.zeros(m) + 1e-6, np.random.rand(m) * 0.33]

        for s, r, mu in itertools.product(haplotypes, rs, mus):
            r[0] = 0
            n_alleles = _get_num_alleles(H, s)
            e = self.haplotype_emission(
                mu, m, n_alleles, scale_mutation_based_on_n_alleles=scale_mutation
            )
            yield n, m, H, s, e, r, mu

    def example_genotypes(self, ts, seed=42):
        np.random.seed(seed)
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
            yield n, m, G, s, e, r, mu

        # Mixture of random and extremes
        rs = [np.zeros(m) + 0.999, np.zeros(m) + 1e-6, np.random.rand(m)]
        mus = [np.zeros(m) + 0.33, np.zeros(m) + 1e-6, np.random.rand(m) * 0.33]

        e = self.genotype_emission(mu, m)

        for s, r, mu in itertools.product(genotypes, rs, mus):
            r[0] = 0
            e = self.genotype_emission(mu, m)
            yield n, m, G, s, e, r, mu

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
            yield n, m, G, s, e, r, mu

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

    def verify(self, ts):
        raise NotImplementedError()


class FBAlgorithmBase(LSBase):
    """Base for forwards backwards algorithm tests."""


class TestMethodsHap(FBAlgorithmBase):
    """Test that we compute the sample likelihoods across all implementations."""

    def verify(self, ts):
        for n, m, H_vs, s, e_vs, r, mu in self.example_parameters_haplotypes(ts):
            F_vs, c_vs, ll_vs = fbh.forwards_ls_hap(n, m, H_vs, s, e_vs, r)
            B_vs = fbh.backwards_ls_hap(n, m, H_vs, s, e_vs, c_vs, r)
            F, c, ll = ls.forwards(H_vs, s, r, p_mutation=mu)
            B = ls.backwards(H_vs, s, c, r, p_mutation=mu)
            self.assertAllClose(F, F_vs)
            self.assertAllClose(B, B_vs)
            self.assertAllClose(ll_vs, ll)
            mu = None
            F, c, ll = ls.forwards(H_vs, s, r, mu)
            B = ls.backwards(H_vs, s, c, r, mu)


class TestMethodsDip(FBAlgorithmBase):
    """Test that we compute the sample likelihoods across all implementations."""

    def verify(self, ts):
        for n, m, G_vs, s, e_vs, r, mu in self.example_parameters_genotypes(ts):
            F_vs, c_vs, ll_vs = fbd.forward_ls_dip_loop(
                n, m, G_vs, s, e_vs, r, norm=True
            )
            F, c, ll = ls.forwards(G_vs, s, r, p_mutation=mu)
            B_vs = fbd.backward_ls_dip_loop(n, m, G_vs, s, e_vs, c_vs, r)
            B = ls.backwards(G_vs, s, c, r, p_mutation=mu)
            self.assertAllClose(F, F_vs)
            self.assertAllClose(B, B_vs)
            self.assertAllClose(ll_vs, ll)


class VitAlgorithmBase(LSBase):
    """Base for viterbi algoritm tests."""


class TestViterbiHap(VitAlgorithmBase):
    """Test that we have the same log-likelihood across all implementations"""

    def verify(self, ts):
        for n, m, H_vs, s, e_vs, r, mu in self.example_parameters_haplotypes(ts):
            V_vs, P_vs, ll_vs = vh.forwards_viterbi_hap_lower_mem_rescaling(
                n, m, H_vs, s, e_vs, r
            )
            path_vs = vh.backwards_viterbi_hap(m, V_vs, P_vs)
            path, ll = ls.viterbi(H_vs, s, r, p_mutation=mu)

            self.assertAllClose(ll_vs, ll)
            self.assertAllClose(path_vs, path)


class TestViterbiDip(VitAlgorithmBase):
    """Test that we have the same log-likelihood across all implementations"""

    def verify(self, ts):
        for n, m, G_vs, s, e_vs, r, mu in self.example_parameters_genotypes(ts):
            V_vs, P_vs, ll_vs = vd.forwards_viterbi_dip_low_mem(n, m, G_vs, s, e_vs, r)
            path_vs = vd.backwards_viterbi_dip(m, V_vs, P_vs)
            phased_path_vs = vd.get_phased_path(n, path_vs)
            path, ll = ls.viterbi(G_vs, s, r, p_mutation=mu)

            self.assertAllClose(ll_vs, ll)
            self.assertAllClose(phased_path_vs, path)
