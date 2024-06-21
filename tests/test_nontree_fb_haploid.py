import pytest

import numpy as np
import numba as nb

from . import lsbase
import lshmm.core as core
import lshmm.fb_haploid as fbh


class TestNonTreeForwardBackwardHaploid(lsbase.ForwardBackwardAlgorithmBase):
    def verify(self, ts, scale_mutation_rate, include_ancestors):
        for n, m, H_vs, s, e_vs, r, _ in self.get_examples_pars(
            ts,
            ploidy=1,
            scale_mutation_rate=scale_mutation_rate,
            include_ancestors=include_ancestors,
            include_extreme_rates=True,
        ):
            F_vs, c_vs, ll_vs = fbh.forwards_ls_hap(n, m, H_vs, s, e_vs, r, norm=False)
            B_vs = fbh.backwards_ls_hap(n, m, H_vs, s, e_vs, c_vs, r)
            self.assertAllClose(np.log10(np.sum(F_vs * B_vs, 1)), ll_vs * np.ones(m))
            F_tmp, c_tmp, ll_tmp = fbh.forwards_ls_hap(
                n, m, H_vs, s, e_vs, r, norm=True
            )
            B_tmp = fbh.backwards_ls_hap(n, m, H_vs, s, e_vs, c_tmp, r)
            self.assertAllClose(np.sum(F_tmp * B_tmp, 1), np.ones(m))
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
    def test_larger(self, scale_mutation_rate, include_ancestors):
        ts = self.get_ts_custom_pars(
            num_samples=45, seq_length=1e5, mean_r=1e-5, mean_mu=1e-5
        )
        self.verify(ts, scale_mutation_rate, include_ancestors)
