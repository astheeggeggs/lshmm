import numba as nb
import numpy as np
import pytest

import lshmm.core as core
import lshmm.vit_haploid as vh

from . import lsbase


class TestNonTreeViterbiHaploidFixedSwitches(lsbase.ViterbiAlgorithmBase):
    def get_examples_pars(self, scale_mutation_rate):
        # Set ref. panel and query.
        # fmt: off
        H = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
            ],
            dtype=np.int8,
        ).T
        query = np.array(
            [
                [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, ],
            ],
            dtype=np.int8,
        )
        # fmt: on

        n = H.shape[1]  # Number of ref. haps
        m = H.shape[0]  # Number of sites

        # Set constant mutation probability.
        mu = np.zeros(m, dtype=np.float64) + 1e-4

        # Set emission prob. matrix.
        num_alleles = core.get_num_alleles(H, query)
        e = core.get_emission_matrix_haploid(
            mu=mu,
            num_sites=m,
            num_alleles=num_alleles,
            scale_mutation_rate=scale_mutation_rate,
        )

        # Set recombination probabilities for different sets of fixed switches.
        recomb_rates_no_switch = np.zeros(m, dtype=np.float64)
        recomb_rates_nonzero_start = np.zeros_like(recomb_rates_no_switch)
        recomb_rates_nonzero_start[0] = 1e-2
        recomb_rates_nonzero_mid = np.zeros_like(recomb_rates_no_switch)
        recomb_rates_nonzero_mid[2] = 1e-2
        recomb_rates_nonzero_end = np.zeros_like(recomb_rates_no_switch)
        recomb_rates_nonzero_end[-1] = 1e-2
        recomb_rates_two_switches = np.zeros_like(recomb_rates_no_switch)
        recomb_rates_two_switches[2] = 1e-2
        recomb_rates_two_switches[8] = 1e-2
        recomb_rates_arr = [
            recomb_rates_no_switch,
            recomb_rates_nonzero_start,
            recomb_rates_nonzero_mid,
            recomb_rates_nonzero_end,
            recomb_rates_two_switches,
        ]

        # Expected paths
        # fmt: off
        path_no_switch = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
            dtype=np.int8,
        )
        path_nonzero_start = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
            dtype=np.int8,
        )
        path_nonzero_mid = np.array(
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1,],
            dtype=np.int8,
        )
        path_nonzero_end = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0,],
            dtype=np.int8,
        )
        path_two_switches = np.array(
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0,],
            dtype=np.int8,
        )
        # fmt: on
        paths_arr = [
            path_no_switch,
            path_nonzero_start,
            path_nonzero_mid,
            path_nonzero_end,
            path_two_switches,
        ]

        for r, path in zip(recomb_rates_arr, paths_arr):
            yield n, m, H, query, e, r, path

    def verify(self, scale_mutation_rate):
        for n, m, H_vs, s, e_vs, r, path in self.get_examples_pars(scale_mutation_rate):
            emission_func = core.get_emission_probability_haploid

            # Implementation: naive
            V_vs, P_vs, ll_vs = vh.forwards_viterbi_hap_naive(
                n=n,
                m=m,
                H=H_vs,
                s=s,
                e=e_vs,
                r=r,
                emission_func=emission_func,
            )
            path_vs = vh.backwards_viterbi_hap(
                m=m,
                V_last=V_vs[m - 1, :],
                P=P_vs,
            )
            ll_check = vh.path_ll_hap(
                n=n,
                m=m,
                H=H_vs,
                path=path_vs,
                s=s,
                e=e_vs,
                r=r,
                emission_func=emission_func,
            )
            np.testing.assert_equal(path_vs, path)
            self.assertAllClose(ll_vs, ll_check)

            # Implementation: naive, vectorised
            V_tmp, P_tmp, ll_tmp = vh.forwards_viterbi_hap_naive_vec(
                n=n,
                m=m,
                H=H_vs,
                s=s,
                e=e_vs,
                r=r,
                emission_func=emission_func,
            )
            path_tmp = vh.backwards_viterbi_hap(
                m=m,
                V_last=V_tmp[m - 1, :],
                P=P_tmp,
            )
            ll_check = vh.path_ll_hap(
                n=n,
                m=m,
                H=H_vs,
                path=path_tmp,
                s=s,
                e=e_vs,
                r=r,
                emission_func=emission_func,
            )
            np.testing.assert_equal(path_tmp, path)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            # Implementation: naive, low memory footprint
            V_tmp, P_tmp, ll_tmp = vh.forwards_viterbi_hap_naive_low_mem(
                n=n,
                m=m,
                H=H_vs,
                s=s,
                e=e_vs,
                r=r,
                emission_func=emission_func,
            )
            path_tmp = vh.backwards_viterbi_hap(m=m, V_last=V_tmp, P=P_tmp)
            ll_check = vh.path_ll_hap(
                n=n,
                m=m,
                H=H_vs,
                path=path_tmp,
                s=s,
                e=e_vs,
                r=r,
                emission_func=emission_func,
            )
            np.testing.assert_equal(path_tmp, path)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            # Implementation: naive, low memory footprint, rescaling
            V_tmp, P_tmp, ll_tmp = vh.forwards_viterbi_hap_naive_low_mem_rescaling(
                n=n,
                m=m,
                H=H_vs,
                s=s,
                e=e_vs,
                r=r,
                emission_func=emission_func,
            )
            path_tmp = vh.backwards_viterbi_hap(m=m, V_last=V_tmp, P=P_tmp)
            ll_check = vh.path_ll_hap(
                n=n,
                m=m,
                H=H_vs,
                path=path_tmp,
                s=s,
                e=e_vs,
                r=r,
                emission_func=emission_func,
            )
            np.testing.assert_equal(path_tmp, path)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            # Implementation: low memory footprint, rescaling
            V_tmp, P_tmp, ll_tmp = vh.forwards_viterbi_hap_low_mem_rescaling(
                n=n,
                m=m,
                H=H_vs,
                s=s,
                e=e_vs,
                r=r,
                emission_func=emission_func,
            )
            path_tmp = vh.backwards_viterbi_hap(m=m, V_last=V_tmp, P=P_tmp)
            ll_check = vh.path_ll_hap(
                n=n,
                m=m,
                H=H_vs,
                path=path_tmp,
                s=s,
                e=e_vs,
                r=r,
                emission_func=emission_func,
            )
            np.testing.assert_equal(path_tmp, path)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            # Implementation: even lower memory footprint, rescaling
            V_tmp, P_tmp, ll_tmp = vh.forwards_viterbi_hap_lower_mem_rescaling(
                n=n,
                m=m,
                H=H_vs,
                s=s,
                e=e_vs,
                r=r,
                emission_func=emission_func,
            )
            path_tmp = vh.backwards_viterbi_hap(m=m, V_last=V_tmp, P=P_tmp)
            ll_check = vh.path_ll_hap(
                n=n,
                m=m,
                H=H_vs,
                path=path_tmp,
                s=s,
                e=e_vs,
                r=r,
                emission_func=emission_func,
            )
            np.testing.assert_equal(path_tmp, path)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            (
                V_tmp,
                V_argmaxes_tmp,
                recombs,
                ll_tmp,
            ) = vh.forwards_viterbi_hap_lower_mem_rescaling_no_pointer(
                n=n,
                m=m,
                H=H_vs,
                s=s,
                e=e_vs,
                r=r,
                emission_func=emission_func,
            )
            path_tmp = vh.backwards_viterbi_hap_no_pointer(
                m=m,
                V_argmaxes=V_argmaxes_tmp,
                recombs=nb.typed.List(recombs),
            )
            ll_check = vh.path_ll_hap(
                n=n,
                m=m,
                H=H_vs,
                path=path_tmp,
                s=s,
                e=e_vs,
                r=r,
                emission_func=emission_func,
            )
            np.testing.assert_equal(path_tmp, path)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

    @pytest.mark.parametrize("scale_mutation_rate", [True, False])
    def test_fixed_switches(self, scale_mutation_rate):
        self.verify(scale_mutation_rate)
