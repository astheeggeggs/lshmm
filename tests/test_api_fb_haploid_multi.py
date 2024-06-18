import pytest

from . import lsbase
import lshmm as ls
import lshmm.core as core
import lshmm.fb_haploid as fbh


class TestForwardBackwardHaploid(lsbase.ForwardBackwardAlgorithmBase):
    def verify(self, ts, scale_mutation_rate, include_ancestors):
        for n, m, H_vs, s, e_vs, r, mu in self.get_examples_pars(
            ts,
            ploidy=1,
            scale_mutation_rate=scale_mutation_rate,
            include_ancestors=include_ancestors,
            include_extreme_rates=True,
        ):
            num_alleles = core.get_num_alleles(H_vs, s)
            F_vs, c_vs, ll_vs = fbh.forwards_ls_hap(
                n=n,
                m=m,
                H=H_vs,
                s=s,
                e=e_vs,
                r=r,
            )
            B_vs = fbh.backwards_ls_hap(
                n=n,
                m=m,
                H=H_vs,
                s=s,
                e=e_vs,
                c=c_vs,
                r=r,
            )
            F, c, ll = ls.forwards(
                reference_panel=H_vs,
                query=s,
                num_alleles=num_alleles,
                prob_recombination=r,
                prob_mutation=mu,
                scale_mutation_rate=scale_mutation_rate,
                normalise=True,
            )
            B = ls.backwards(
                reference_panel=H_vs,
                query=s,
                num_alleles=num_alleles,
                normalisation_factor_from_forward=c,
                prob_recombination=r,
                prob_mutation=mu,
                scale_mutation_rate=scale_mutation_rate,
            )
            self.assertAllClose(F, F_vs)
            self.assertAllClose(B, B_vs)
            self.assertAllClose(ll_vs, ll)

    @pytest.mark.parametrize("scale_mutation_rate", [True, False])
    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_multiallelic_n10_no_recomb(
        self, scale_mutation_rate, include_ancestors
    ):
        ts = self.get_ts_multiallelic_n10_no_recomb()
        self.verify(ts, scale_mutation_rate, include_ancestors)

    @pytest.mark.parametrize("scale_mutation_rate", [True, False])
    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_multiallelic_n6(self, scale_mutation_rate, include_ancestors):
        ts = self.get_ts_multiallelic_n6()
        self.verify(ts, scale_mutation_rate, include_ancestors)

    @pytest.mark.parametrize("scale_mutation_rate", [True, False])
    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_multiallelic_n8(self, scale_mutation_rate, include_ancestors):
        ts = self.get_ts_multiallelic_n8()
        self.verify(ts, scale_mutation_rate, include_ancestors)

    @pytest.mark.parametrize("scale_mutation_rate", [True, False])
    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_multiallelic_n16(self, scale_mutation_rate, include_ancestors):
        ts = self.get_ts_multiallelic_n16()
        self.verify(ts, scale_mutation_rate, include_ancestors)
