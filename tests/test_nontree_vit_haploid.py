import pytest

import numpy as np
import numba as nb

from . import lsbase
import lshmm.core as core
import lshmm.vit_haploid as vh


class TestNonTreeViterbiHaploid(lsbase.ViterbiAlgorithmBase):
    def verify(self, ts, scale_mutation_rate, include_ancestors):
        for n, m, H_vs, s, e_vs, r, _ in self.get_examples_pars(
            ts,
            ploidy=1,
            scale_mutation_rate=scale_mutation_rate,
            include_ancestors=include_ancestors,
            include_extreme_rates=True,
        ):
            V_vs, P_vs, ll_vs = vh.forwards_viterbi_hap_naive(n, m, H_vs, s, e_vs, r)
            path_vs = vh.backwards_viterbi_hap(m, V_vs[m - 1, :], P_vs)
            ll_check = vh.path_ll_hap(n, m, H_vs, path_vs, s, e_vs, r)
            self.assertAllClose(ll_vs, ll_check)

            V_tmp, P_tmp, ll_tmp = vh.forwards_viterbi_hap_naive_vec(
                n, m, H_vs, s, e_vs, r
            )
            path_tmp = vh.backwards_viterbi_hap(m, V_tmp[m - 1, :], P_tmp)
            ll_check = vh.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            V_tmp, P_tmp, ll_tmp = vh.forwards_viterbi_hap_naive_low_mem(
                n, m, H_vs, s, e_vs, r
            )
            path_tmp = vh.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            V_tmp, P_tmp, ll_tmp = vh.forwards_viterbi_hap_naive_low_mem_rescaling(
                n, m, H_vs, s, e_vs, r
            )
            path_tmp = vh.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            V_tmp, P_tmp, ll_tmp = vh.forwards_viterbi_hap_low_mem_rescaling(
                n, m, H_vs, s, e_vs, r
            )
            path_tmp = vh.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            V_tmp, P_tmp, ll_tmp = vh.forwards_viterbi_hap_lower_mem_rescaling(
                n, m, H_vs, s, e_vs, r
            )
            path_tmp = vh.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            (
                V_tmp,
                V_argmaxes_tmp,
                recombs,
                ll_tmp,
            ) = vh.forwards_viterbi_hap_lower_mem_rescaling_no_pointer(
                n, m, H_vs, s, e_vs, r
            )
            path_tmp = vh.backwards_viterbi_hap_no_pointer(
                m,
                V_argmaxes_tmp,
                nb.typed.List(recombs),
            )
            ll_check = vh.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

    @pytest.mark.parametrize("scale_mutation_rate", [True, False])
    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_simple_n10_no_recombn(self, scale_mutation_rate, include_ancestors):
        ts = self.get_ts_simple_n10_no_recomb()
        self.verify(
            ts,
            scale_mutation_rate=scale_mutation_rate,
            include_ancestors=include_ancestors,
        )

    @pytest.mark.parametrize("scale_mutation_rate", [True, False])
    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_simple_n6(self, scale_mutation_rate, include_ancestors):
        ts = self.get_ts_simple_n6()
        self.verify(
            ts,
            scale_mutation_rate=scale_mutation_rate,
            include_ancestors=include_ancestors,
        )

    @pytest.mark.parametrize("scale_mutation_rate", [True, False])
    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_simple_n8(self, scale_mutation_rate, include_ancestors):
        ts = self.get_ts_simple_n8()
        self.verify(
            ts,
            scale_mutation_rate=scale_mutation_rate,
            include_ancestors=include_ancestors,
        )

    @pytest.mark.parametrize("scale_mutation_rate", [True, False])
    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_simple_n8_high_recomb(self, scale_mutation_rate, include_ancestors):
        ts = self.get_ts_simple_n8_high_recomb()
        self.verify(
            ts,
            scale_mutation_rate=scale_mutation_rate,
            include_ancestors=include_ancestors,
        )

    @pytest.mark.parametrize("scale_mutation_rate", [True, False])
    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_simple_n16(self, scale_mutation_rate, include_ancestors):
        ts = self.get_ts_simple_n16()
        self.verify(
            ts,
            scale_mutation_rate=scale_mutation_rate,
            include_ancestors=include_ancestors,
        )

    @pytest.mark.parametrize("scale_mutation_rate", [True, False])
    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_larger(self, scale_mutation_rate, include_ancestors):
        ts = self.get_ts_custom_pars(
            ref_panel_size=45, length=1e5, mean_r=1e-5, mean_mu=1e-5
        )
        self.verify(
            ts,
            scale_mutation_rate=scale_mutation_rate,
            include_ancestors=include_ancestors,
        )
