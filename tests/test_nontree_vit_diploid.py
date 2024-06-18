import pytest

import numpy as np
import numba as nb

from . import lsbase
import lshmm.core as core
import lshmm.vit_diploid as vd


class TestNonTreeViterbiDiploid(lsbase.ViterbiAlgorithmBase):
    def verify(self, ts, scale_mutation_rate, include_ancestors):
        for n, m, H_vs, query, e_vs, r, _ in self.get_examples_pars(
            ts,
            ploidy=2,
            scale_mutation_rate=scale_mutation_rate,
            include_ancestors=include_ancestors,
            include_extreme_rates=True,
        ):
            G_vs = core.convert_haplotypes_to_phased_genotypes(H_vs)
            s = core.convert_haplotypes_to_unphased_genotypes(query)

            V_vs, P_vs, ll_vs = vd.forwards_viterbi_dip_low_mem(n, m, G_vs, s, e_vs, r)
            path_vs = vd.backwards_viterbi_dip(m, V_vs, P_vs)
            phased_path_vs = vd.get_phased_path(n, path_vs)
            path_ll_vs = vd.path_ll_dip(n, m, G_vs, phased_path_vs, s, e_vs, r)
            self.assertAllClose(ll_vs, path_ll_vs)

            (
                V_tmp,
                V_argmaxes_tmp,
                V_rowcol_maxes_tmp,
                V_rowcol_argmaxes_tmp,
                recombs_single,
                recombs_double,
                ll_tmp,
            ) = vd.forwards_viterbi_dip_low_mem_no_pointer(n, m, G_vs, s, e_vs, r)
            path_tmp = vd.backwards_viterbi_dip_no_pointer(
                m,
                V_argmaxes_tmp,
                V_rowcol_maxes_tmp,
                V_rowcol_argmaxes_tmp,
                nb.typed.List(recombs_single),
                nb.typed.List(recombs_double),
                V_tmp,
            )
            phased_path_tmp = vd.get_phased_path(n, path_tmp)
            path_ll_tmp = vd.path_ll_dip(n, m, G_vs, phased_path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, path_ll_tmp)
            self.assertAllClose(ll_vs, ll_tmp)

            MAX_NUM_REF_HAPS = 50
            num_ref_haps = H_vs.shape[1]
            if num_ref_haps <= MAX_NUM_REF_HAPS:
                # Run tests for the naive implementations.
                V_tmp, P_tmp, ll_tmp = vd.forwards_viterbi_dip_naive(
                    n, m, G_vs, s, e_vs, r
                )
                path_tmp = vd.backwards_viterbi_dip(m, V_tmp[m - 1, :, :], P_tmp)
                phased_path_tmp = vd.get_phased_path(n, path_tmp)
                path_ll_tmp = vd.path_ll_dip(n, m, G_vs, phased_path_tmp, s, e_vs, r)
                self.assertAllClose(ll_tmp, path_ll_tmp)
                self.assertAllClose(ll_vs, ll_tmp)

                V_tmp, P_tmp, ll_tmp = vd.forwards_viterbi_dip_naive_low_mem(
                    n, m, G_vs, s, e_vs, r
                )
                path_tmp = vd.backwards_viterbi_dip(m, V_tmp, P_tmp)
                phased_path_tmp = vd.get_phased_path(n, path_tmp)
                path_ll_tmp = vd.path_ll_dip(n, m, G_vs, phased_path_tmp, s, e_vs, r)
                self.assertAllClose(ll_tmp, path_ll_tmp)
                self.assertAllClose(ll_vs, ll_tmp)

                V_tmp, P_tmp, ll_tmp = vd.forwards_viterbi_dip_naive_vec(
                    n, m, G_vs, s, e_vs, r
                )
                path_tmp = vd.backwards_viterbi_dip(m, V_tmp[m - 1, :, :], P_tmp)
                phased_path_tmp = vd.get_phased_path(n, path_tmp)
                path_ll_tmp = vd.path_ll_dip(n, m, G_vs, phased_path_tmp, s, e_vs, r)
                self.assertAllClose(ll_tmp, path_ll_tmp)
                self.assertAllClose(ll_vs, ll_tmp)

    @pytest.mark.parametrize("scale_mutation_rate", [True, False])
    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_simple_n10_no_recomb(self, scale_mutation_rate, include_ancestors):
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
            ref_panel_size=30, length=1e5, mean_r=1e-6, mean_mu=1e-5
        )
        self.verify(
            ts,
            scale_mutation_rate=scale_mutation_rate,
            include_ancestors=include_ancestors,
        )
