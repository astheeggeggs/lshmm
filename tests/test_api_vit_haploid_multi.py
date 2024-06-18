import pytest

from . import lsbase
import lshmm as ls
import lshmm.core as core
import lshmm.vit_haploid as vh


class TestViterbiHaploid(lsbase.ViterbiAlgorithmBase):
    def verify(self, ts, scale_mutation_rate, include_ancestors):
        for n, m, H_vs, s, e_vs, r, mu in self.get_examples_pars(
            ts,
            ploidy=1,
            scale_mutation_rate=scale_mutation_rate,
            include_ancestors=include_ancestors,
            include_extreme_rates=True,
        ):
            num_alleles = core.get_num_alleles(H_vs, s)
            V_vs, P_vs, ll_vs = vh.forwards_viterbi_hap_lower_mem_rescaling(
                n, m, H_vs, s, e_vs, r
            )
            path_vs = vh.backwards_viterbi_hap(m, V_vs, P_vs)
            path_ll_hap = vh.path_ll_hap(n, m, H_vs, path_vs, s, e_vs, r)
            path, ll = ls.viterbi(
                reference_panel=H_vs,
                query=s,
                num_alleles=num_alleles,
                prob_recombination=r,
                prob_mutation=mu,
                scale_mutation_rate=scale_mutation_rate,
            )
            self.assertAllClose(ll_vs, ll)
            self.assertAllClose(ll_vs, path_ll_hap)
            self.assertAllClose(path_vs, path)

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
