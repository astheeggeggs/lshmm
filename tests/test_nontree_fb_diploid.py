import numba as nb
import numpy as np
import pytest

import lshmm.core as core
import lshmm.fb_diploid as fbd

from . import lsbase


class TestNonTreeForwardBackwardDiploid(lsbase.ForwardBackwardAlgorithmBase):
    def verify(
        self,
        ts,
        scale_mutation_rate,
        include_ancestors,
        include_extreme_rates,
        normalise,
    ):
        for n, m, H_vs, query, e_vs, r, _ in self.get_examples_pars(
            ts,
            ploidy=2,
            scale_mutation_rate=scale_mutation_rate,
            include_ancestors=include_ancestors,
            include_extreme_rates=include_extreme_rates,
        ):
            G_vs = core.convert_haplotypes_to_phased_genotypes(H_vs)

            F_vs, c_vs, ll_vs = fbd.forwards_ls_dip(
                n, m, G_vs, query, e_vs, r, norm=True
            )
            B_vs = fbd.backwards_ls_dip(n, m, G_vs, query, e_vs, c_vs, r)
            self.assertAllClose(np.sum(F_vs * B_vs, (1, 2)), np.ones(m))

            F_tmp, c_tmp, ll_tmp = fbd.forward_ls_dip_loop(
                n, m, G_vs, query, e_vs, r, norm=True
            )
            B_tmp = fbd.backward_ls_dip_loop(n, m, G_vs, query, e_vs, c_tmp, r)
            self.assertAllClose(np.sum(F_tmp * B_tmp, (1, 2)), np.ones(m))
            self.assertAllClose(ll_vs, ll_tmp)

            if not normalise:
                F_tmp, c_tmp, ll_tmp = fbd.forwards_ls_dip(
                    n, m, G_vs, query, e_vs, r, norm=False
                )
                if ll_tmp != -np.inf:
                    B_tmp = fbd.backwards_ls_dip(n, m, G_vs, query, e_vs, c_tmp, r)
                    self.assertAllClose(
                        np.log10(np.sum(F_tmp * B_tmp, (1, 2))), ll_tmp * np.ones(m)
                    )
                    self.assertAllClose(ll_vs, ll_tmp)

                F_tmp, c_tmp, ll_tmp = fbd.forward_ls_dip_loop(
                    n, m, G_vs, query, e_vs, r, norm=False
                )
                if ll_tmp != -np.inf:
                    B_tmp = fbd.backward_ls_dip_loop(n, m, G_vs, query, e_vs, c_tmp, r)
                    self.assertAllClose(
                        np.log10(np.sum(F_tmp * B_tmp, (1, 2))), ll_tmp * np.ones(m)
                    )
                    self.assertAllClose(ll_vs, ll_tmp)

                F_tmp, ll_tmp = fbd.forward_ls_dip_starting_point(
                    n, m, G_vs, query, e_vs, r
                )
                if ll_tmp != -np.inf:
                    B_tmp = fbd.backward_ls_dip_starting_point(
                        n, m, G_vs, query, e_vs, r
                    )
                    self.assertAllClose(
                        np.log10(np.sum(F_tmp * B_tmp, (1, 2))), ll_tmp * np.ones(m)
                    )
                    self.assertAllClose(ll_vs, ll_tmp)

    @pytest.mark.parametrize("scale_mutation_rate", [True, False])
    @pytest.mark.parametrize("include_ancestors", [True, False])
    @pytest.mark.parametrize("normalise", [True, False])
    def test_ts_simple_n10_no_recomb(
        self, scale_mutation_rate, include_ancestors, normalise
    ):
        ts = self.get_ts_simple_n10_no_recomb()
        # Test extreme rates only when normalising,
        # because they can lead to pathological cases.
        include_extreme_rates = normalise
        self.verify(
            ts,
            scale_mutation_rate=scale_mutation_rate,
            include_ancestors=include_ancestors,
            normalise=normalise,
            include_extreme_rates=include_extreme_rates,
        )

    @pytest.mark.parametrize("num_samples", [8, 16])
    @pytest.mark.parametrize("scale_mutation_rate", [True, False])
    @pytest.mark.parametrize("include_ancestors", [True, False])
    @pytest.mark.parametrize("normalise", [True, False])
    def test_ts_simple(
        self, num_samples, scale_mutation_rate, include_ancestors, normalise
    ):
        ts = self.get_ts_simple(num_samples)
        include_extreme_rates = normalise
        self.verify(
            ts,
            scale_mutation_rate=scale_mutation_rate,
            include_ancestors=include_ancestors,
            normalise=normalise,
            include_extreme_rates=include_extreme_rates,
        )

    @pytest.mark.parametrize("scale_mutation_rate", [True, False])
    @pytest.mark.parametrize("include_ancestors", [True, False])
    @pytest.mark.parametrize("normalise", [True, False])
    def test_ts_simple_n8_high_recomb(
        self, scale_mutation_rate, include_ancestors, normalise
    ):
        ts = self.get_ts_simple_n8_high_recomb()
        include_extreme_rates = normalise
        self.verify(
            ts,
            scale_mutation_rate=scale_mutation_rate,
            include_ancestors=include_ancestors,
            normalise=normalise,
            include_extreme_rates=include_extreme_rates,
        )

    @pytest.mark.parametrize("scale_mutation_rate", [True, False])
    @pytest.mark.parametrize("include_ancestors", [True, False])
    @pytest.mark.parametrize("normalise", [True, False])
    def test_ts_larger(self, scale_mutation_rate, include_ancestors, normalise):
        ts = self.get_ts_custom_pars(
            num_samples=30,
            seq_length=1e5,
            mean_r=1e-5,
            mean_mu=1e-5,
        )
        include_extreme_rates = normalise
        self.verify(
            ts,
            scale_mutation_rate=scale_mutation_rate,
            include_ancestors=include_ancestors,
            normalise=normalise,
            include_extreme_rates=include_extreme_rates,
        )
