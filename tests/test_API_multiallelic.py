import pytest
from . import lsbase
import lshmm as ls
import lshmm.fb_haploid as fbh
import lshmm.vit_haploid as vh


class TestForwardBackwardHaploid(lsbase.ForwardBackwardAlgorithmBase):
    def verify(self, ts, scale_mutation_rate, include_ancestors):
        for n, m, H_vs, s, e_vs, r, mu in self.get_examples_pars(
            ts,
            ploidy=1,
            scale_mutation_rate=scale_mutation_rate,
            include_ancestors=include_ancestors,
            include_extreme_rates=True,
        ):
            F_vs, c_vs, ll_vs = fbh.forwards_ls_hap(n, m, H_vs, s, e_vs, r)
            B_vs = fbh.backwards_ls_hap(n, m, H_vs, s, e_vs, c_vs, r)
            F, c, ll = ls.forwards(
                H_vs, s, r, prob_mutation=mu, scale_mutation_rate=scale_mutation_rate
            )
            B = ls.backwards(
                H_vs, s, c, r, prob_mutation=mu, scale_mutation_rate=scale_mutation_rate
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


class TestViterbiHaploid(lsbase.ViterbiAlgorithmBase):
    def verify(self, ts, scale_mutation_rate, include_ancestors):
        for n, m, H_vs, s, e_vs, r, mu in self.get_examples_pars(
            ts,
            ploidy=1,
            scale_mutation_rate=scale_mutation_rate,
            include_ancestors=include_ancestors,
            include_extreme_rates=True,
        ):
            V_vs, P_vs, ll_vs = vh.forwards_viterbi_hap_lower_mem_rescaling(
                n, m, H_vs, s, e_vs, r
            )
            path_vs = vh.backwards_viterbi_hap(m, V_vs, P_vs)
            path_ll_hap = vh.path_ll_hap(n, m, H_vs, path_vs, s, e_vs, r)
            path, ll = ls.viterbi(
                H_vs, s, r, prob_mutation=mu, scale_mutation_rate=scale_mutation_rate
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
