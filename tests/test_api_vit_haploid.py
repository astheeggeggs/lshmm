import pytest

from . import lsbase
import lshmm as ls
import lshmm.core as core
import lshmm.vit_haploid as vh


class TestViterbiHaploid(lsbase.ViterbiAlgorithmBase):
    def verify(self, ts, scale_mutation_rate, include_ancestors):
        ploidy = 1
        for n, m, H_vs, s, e_vs, r, mu in self.get_examples_pars(
            ts,
            ploidy=ploidy,
            scale_mutation_rate=scale_mutation_rate,
            include_ancestors=include_ancestors,
            include_extreme_rates=True,
        ):
            emission_func = core.get_emission_probability_haploid
            V_vs, P_vs, ll_vs = vh.forwards_viterbi_hap_lower_mem_rescaling(
                n=n,
                m=m,
                H=H_vs,
                s=s,
                e=e_vs,
                r=r,
                emission_func=emission_func,
            )
            path_vs = vh.backwards_viterbi_hap(m=m, V_last=V_vs, P=P_vs)
            path, ll = ls.viterbi(
                reference_panel=H_vs,
                query=s,
                ploidy=ploidy,
                prob_recombination=r,
                prob_mutation=mu,
                scale_mutation_rate=scale_mutation_rate,
            )
            self.assertAllClose(ll_vs, ll)
            self.assertAllClose(path_vs, path)

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
