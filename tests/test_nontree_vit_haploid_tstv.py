import itertools
import pytest

import numpy as np
import numba as nb

from . import lsbase
import lshmm.core as core
import lshmm.vit_haploid as vh


class TestNonTreeViterbiHaploid(lsbase.ViterbiAlgorithmBase):
    def verify(self, ts, include_ancestors):
        H, queries = self.get_examples_haploid(ts, include_ancestors)
        m = H.shape[0]
        n = H.shape[1]

        r_s = [
            np.zeros(m) + 0.01,
            np.random.rand(m),
            1e-5 * (np.random.rand(m) + 0.5) / 2,
            np.zeros(m) + 0.2,
            np.zeros(m) + 1e-6,
        ]
        mu_s = [
            np.zeros(m) + 0.01,
            np.random.rand(m) * 0.2,
            1e-5 * (np.random.rand(m) + 0.5) / 2,
            np.zeros(m) + 0.2,
            np.zeros(m) + 1e-6,
        ]
        kappa_s = [0.25, 0.5, 1.0, 1.5, 2.0]

        for s, r, mu, kappa in itertools.product(queries, r_s, mu_s, kappa_s):
            e = core.get_emission_matrix_haploid_tstv(mu, kappa)

            V_vs, P_vs, ll_vs = vh.forwards_viterbi_hap_naive(
                n=n,
                m=m,
                H=H,
                s=s,
                e=e,
                r=r,
                emission_func=core.get_emission_probability_haploid_tstv,
            )
            path_vs = vh.backwards_viterbi_hap(m=m, V_last=V_vs[m - 1, :], P=P_vs)
            ll_check = vh.path_ll_hap(
                n=n,
                m=m,
                H=H,
                path=path_vs,
                s=s,
                e=e,
                r=r,
                emission_func=core.get_emission_probability_haploid_tstv,
            )
            self.assertAllClose(ll_vs, ll_check)

            V_tmp, P_tmp, ll_tmp = vh.forwards_viterbi_hap_naive_vec(
                n=n,
                m=m,
                H=H,
                s=s,
                e=e,
                r=r,
                emission_func=core.get_emission_probability_haploid_tstv,
            )
            path_tmp = vh.backwards_viterbi_hap(m=m, V_last=V_tmp[m - 1, :], P=P_tmp)
            ll_check = vh.path_ll_hap(
                n=n,
                m=m,
                H=H,
                path=path_tmp,
                s=s,
                e=e,
                r=r,
                emission_func=core.get_emission_probability_haploid_tstv,
            )
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            V_tmp, P_tmp, ll_tmp = vh.forwards_viterbi_hap_naive_low_mem(
                n=n,
                m=m,
                H=H,
                s=s,
                e=e,
                r=r,
                emission_func=core.get_emission_probability_haploid_tstv,
            )
            path_tmp = vh.backwards_viterbi_hap(m=m, V_last=V_tmp, P=P_tmp)
            ll_check = vh.path_ll_hap(
                n=n,
                m=m,
                H=H,
                path=path_tmp,
                s=s,
                e=e,
                r=r,
                emission_func=core.get_emission_probability_haploid_tstv,
            )
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            V_tmp, P_tmp, ll_tmp = vh.forwards_viterbi_hap_naive_low_mem_rescaling(
                n=n,
                m=m,
                H=H,
                s=s,
                e=e,
                r=r,
                emission_func=core.get_emission_probability_haploid_tstv,
            )
            path_tmp = vh.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh.path_ll_hap(
                n=n,
                m=m,
                H=H,
                path=path_tmp,
                s=s,
                e=e,
                r=r,
                emission_func=core.get_emission_probability_haploid_tstv,
            )
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            V_tmp, P_tmp, ll_tmp = vh.forwards_viterbi_hap_low_mem_rescaling(
                n=n,
                m=m,
                H=H,
                s=s,
                e=e,
                r=r,
                emission_func=core.get_emission_probability_haploid_tstv,
            )
            path_tmp = vh.backwards_viterbi_hap(m=m, V_last=V_tmp, P=P_tmp)
            ll_check = vh.path_ll_hap(
                n=n,
                m=m,
                H=H,
                path=path_tmp,
                s=s,
                e=e,
                r=r,
                emission_func=core.get_emission_probability_haploid_tstv,
            )
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            V_tmp, P_tmp, ll_tmp = vh.forwards_viterbi_hap_lower_mem_rescaling(
                n=n,
                m=m,
                H=H,
                s=s,
                e=e,
                r=r,
                emission_func=core.get_emission_probability_haploid_tstv,
            )
            path_tmp = vh.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh.path_ll_hap(
                n=n,
                m=m,
                H=H,
                path=path_tmp,
                s=s,
                e=e,
                r=r,
                emission_func=core.get_emission_probability_haploid_tstv,
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
                H=H,
                s=s,
                e=e,
                r=r,
                emission_func=core.get_emission_probability_haploid_tstv,
            )
            path_tmp = vh.backwards_viterbi_hap_no_pointer(
                m=m,
                V_argmaxes=V_argmaxes_tmp,
                recombs=nb.typed.List(recombs),
            )
            ll_check = vh.path_ll_hap(
                n=n,
                m=m,
                H=H,
                path=path_tmp,
                s=s,
                e=e,
                r=r,
                emission_func=core.get_emission_probability_haploid_tstv,
            )
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_multiallelic_n10_no_recomb(self, include_ancestors):
        ts = self.get_ts_multiallelic_n10_no_recomb()
        self.verify(ts, include_ancestors)

    @pytest.mark.parametrize("num_samples", [8, 16, 32])
    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_multiallelic(self, num_samples, include_ancestors):
        ts = self.get_ts_multiallelic(num_samples)
        self.verify(ts, include_ancestors)
