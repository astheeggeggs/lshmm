import pytest

import numpy as np
import numba as nb

from . import lsbase
import lshmm.core as core
import lshmm.vit_haploid as vh


class TestNonTreeViterbiHaploid(lsbase.ViterbiAlgorithmBase):
    def verify(self, ts, scale_mutation_rate, include_ancestors):
        ploidy = 1
        for n, m, H_vs, s, e_vs, r, _ in self.get_examples_pars(
            ts,
            ploidy=ploidy,
            scale_mutation_rate=scale_mutation_rate,
            include_ancestors=include_ancestors,
            include_extreme_rates=True,
        ):
            emission_func = core.get_emission_probability_haploid

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
            self.assertAllClose(ll_vs, ll_check)

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
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

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
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

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
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

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
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

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
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

    @pytest.mark.parametrize("scale_mutation_rate", [True, False])
    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_simple_n10_no_recomb(self, scale_mutation_rate, include_ancestors):
        ts = self.get_ts_simple_n10_no_recomb()
        self.verify(ts, scale_mutation_rate, include_ancestors)

    @pytest.mark.parametrize("num_samples", [8, 16, 32])
    @pytest.mark.parametrize("scale_mutation_rate", [True, False])
    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_simple(self, num_samples, scale_mutation_rate, include_ancestors):
        ts = self.get_ts_simple(num_samples)
        self.verify(ts, scale_mutation_rate, include_ancestors)

    @pytest.mark.parametrize("scale_mutation_rate", [True, False])
    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_simple_n8_high_recomb(self, scale_mutation_rate, include_ancestors):
        ts = self.get_ts_simple_n8_high_recomb()
        self.verify(ts, scale_mutation_rate, include_ancestors)

    @pytest.mark.parametrize("scale_mutation_rate", [True, False])
    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_larger(self, scale_mutation_rate, include_ancestors):
        ts = self.get_ts_custom_pars(
            num_samples=45, seq_length=1e5, mean_r=1e-5, mean_mu=1e-5
        )
        self.verify(ts, scale_mutation_rate, include_ancestors)
