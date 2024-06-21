import pytest

from . import lsbase
import lshmm as ls
import lshmm.core as core
import lshmm.fb_haploid as fbh


class TestForwardBackwardHaploid(lsbase.ForwardBackwardAlgorithmBase):
    def verify(self, ts, scale_mutation_rate, include_ancestors):
        ploidy = 1
        for n, m, H_vs, s, e_vs, r, mu in self.get_examples_pars(
            ts,
            ploidy=ploidy,
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
                ploidy=ploidy,
                prob_recombination=r,
                prob_mutation=mu,
                scale_mutation_rate=scale_mutation_rate,
                normalise=True,
            )
            B = ls.backwards(
                reference_panel=H_vs,
                query=s,
                ploidy=ploidy,
                normalisation_factor_from_forward=c,
                prob_recombination=r,
                prob_mutation=mu,
                scale_mutation_rate=scale_mutation_rate,
            )
            self.assertAllClose(F_vs, F)
            self.assertAllClose(B_vs, B)
            self.assertAllClose(ll_vs, ll)

    @pytest.mark.parametrize("scale_mutation_rate", [True, False])
    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_simple_n10_no_recomb(self, scale_mutation_rate, include_ancestors):
        ts = self.get_ts_simple_n10_no_recomb()
        self.verify(ts, scale_mutation_rate, include_ancestors)

    @pytest.mark.parametrize("num_samples", [4, 8, 16, 32])
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
