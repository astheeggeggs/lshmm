import pytest
from . import lsbase
import lshmm as ls
import lshmm.fb_diploid as fbd
import lshmm.fb_haploid as fbh
import lshmm.vit_diploid as vd
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
            F, c, ll = ls.forwards(H_vs, s, r, prob_mutation=mu)
            B = ls.backwards(H_vs, s, c, r, prob_mutation=mu)
            self.assertAllClose(F, F_vs)
            self.assertAllClose(B, B_vs)
            self.assertAllClose(ll_vs, ll)

    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_simple_n10_no_recomb(self, include_ancestors):
        ts = self.get_ts_simple_n10_no_recomb()
        self.verify(ts, scale_mutation_rate=True, include_ancestors=include_ancestors)

    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_simple_n6(self, include_ancestors):
        ts = self.get_ts_simple_n6()
        self.verify(ts, scale_mutation_rate=True, include_ancestors=include_ancestors)

    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_simple_n8(self, include_ancestors):
        ts = self.get_ts_simple_n8()
        self.verify(ts, scale_mutation_rate=True, include_ancestors=include_ancestors)

    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_simple_n8_high_recomb(self, include_ancestors):
        ts = self.get_ts_simple_n8_high_recomb()
        self.verify(ts, scale_mutation_rate=True, include_ancestors=include_ancestors)

    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_simple_n16(self, include_ancestors):
        ts = self.get_ts_simple_n16()
        self.verify(ts, scale_mutation_rate=True, include_ancestors=include_ancestors)

    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_larger(self, include_ancestors):
        ts = self.get_ts_custom_pars(
            ref_panel_size=45, length=1e5, mean_r=1e-5, mean_mu=1e-5
        )
        self.verify(
            ts,
            scale_mutation_rate=True,
            include_ancestors=include_ancestors,
        )


class TestForwardBackwardDiploid(lsbase.ForwardBackwardAlgorithmBase):
    def verify(self, ts, scale_mutation_rate, include_ancestors):
        for n, m, G_vs, s, e_vs, r, mu in self.get_examples_pars(
            ts,
            ploidy=2,
            scale_mutation_rate=scale_mutation_rate,
            include_ancestors=include_ancestors,
            include_extreme_rates=True,
        ):
            F_vs, c_vs, ll_vs = fbd.forward_ls_dip_loop(
                n, m, G_vs, s, e_vs, r, norm=True
            )
            F, c, ll = ls.forwards(G_vs, s, r, prob_mutation=mu)
            B_vs = fbd.backward_ls_dip_loop(n, m, G_vs, s, e_vs, c_vs, r)
            B = ls.backwards(G_vs, s, c, r, prob_mutation=mu)
            self.assertAllClose(F, F_vs)
            self.assertAllClose(B, B_vs)
            self.assertAllClose(ll_vs, ll)

    @pytest.mark.parametrize("include_ancestors", [False])
    def test_ts_simple_n10_no_recomb(self, include_ancestors):
        ts = self.get_ts_simple_n10_no_recomb()
        self.verify(ts, scale_mutation_rate=True, include_ancestors=include_ancestors)

    @pytest.mark.parametrize("include_ancestors", [False])
    def test_ts_simple_n6(self, include_ancestors):
        ts = self.get_ts_simple_n6()
        self.verify(ts, scale_mutation_rate=True, include_ancestors=include_ancestors)

    @pytest.mark.parametrize("include_ancestors", [False])
    def test_ts_simple_n8(self, include_ancestors):
        ts = self.get_ts_simple_n8()
        self.verify(ts, scale_mutation_rate=True, include_ancestors=include_ancestors)

    @pytest.mark.parametrize("include_ancestors", [False])
    def test_ts_simple_n8_high_recomb(self, include_ancestors):
        ts = self.get_ts_simple_n8_high_recomb()
        self.verify(ts, scale_mutation_rate=True, include_ancestors=include_ancestors)

    @pytest.mark.parametrize("include_ancestors", [False])
    def test_ts_simple_n16(self, include_ancestors):
        ts = self.get_ts_simple_n16()
        self.verify(ts, scale_mutation_rate=True, include_ancestors=include_ancestors)

    @pytest.mark.parametrize("include_ancestors", [False])
    def test_ts_larger(self, include_ancestors):
        ts = self.get_ts_custom_pars(
            ref_panel_size=45, length=1e5, mean_r=1e-5, mean_mu=1e-5
        )
        self.verify(
            ts,
            scale_mutation_rate=True,
            include_ancestors=include_ancestors,
        )


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
            path, ll = ls.viterbi(H_vs, s, r, prob_mutation=mu)
            self.assertAllClose(ll_vs, ll)
            self.assertAllClose(path_vs, path)

    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_simple_n10_no_recomb(self, include_ancestors):
        ts = self.get_ts_simple_n10_no_recomb()
        self.verify(ts, scale_mutation_rate=True, include_ancestors=include_ancestors)

    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_simple_n6(self, include_ancestors):
        ts = self.get_ts_simple_n6()
        self.verify(ts, scale_mutation_rate=True, include_ancestors=include_ancestors)

    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_simple_n8(self, include_ancestors):
        ts = self.get_ts_simple_n8()
        self.verify(ts, scale_mutation_rate=True, include_ancestors=include_ancestors)

    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_simple_n8_high_recomb(self, include_ancestors):
        ts = self.get_ts_simple_n8_high_recomb()
        self.verify(ts, scale_mutation_rate=True, include_ancestors=include_ancestors)

    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_simple_n16(self, include_ancestors):
        ts = self.get_ts_simple_n16()
        self.verify(ts, scale_mutation_rate=True, include_ancestors=include_ancestors)

    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_larger(self, include_ancestors):
        ts = self.get_ts_custom_pars(
            ref_panel_size=46, length=1e5, mean_r=1e-5, mean_mu=1e-5
        )
        self.verify(
            ts,
            scale_mutation_rate=True,
            include_ancestors=include_ancestors,
        )


class TestViterbiDiploid(lsbase.ViterbiAlgorithmBase):
    def verify(self, ts, scale_mutation_rate, include_ancestors):
        for n, m, G_vs, s, e_vs, r, mu in self.get_examples_pars(
            ts,
            ploidy=2,
            scale_mutation_rate=scale_mutation_rate,
            include_ancestors=include_ancestors,
            include_extreme_rates=True,
        ):
            V_vs, P_vs, ll_vs = vd.forwards_viterbi_dip_low_mem(n, m, G_vs, s, e_vs, r)
            path_vs = vd.backwards_viterbi_dip(m, V_vs, P_vs)
            phased_path_vs = vd.get_phased_path(n, path_vs)
            path, ll = ls.viterbi(G_vs, s, r, prob_mutation=mu)
            self.assertAllClose(ll_vs, ll)
            self.assertAllClose(phased_path_vs, path)

    @pytest.mark.parametrize("include_ancestors", [False])
    def test_ts_simple_n10_no_recomb(self, include_ancestors):
        ts = self.get_ts_simple_n10_no_recomb()
        self.verify(ts, scale_mutation_rate=True, include_ancestors=include_ancestors)

    @pytest.mark.parametrize("include_ancestors", [False])
    def test_ts_simple_n6(self, include_ancestors):
        ts = self.get_ts_simple_n6()
        self.verify(ts, scale_mutation_rate=True, include_ancestors=include_ancestors)

    @pytest.mark.parametrize("include_ancestors", [False])
    def test_ts_simple_n8(self, include_ancestors):
        ts = self.get_ts_simple_n8()
        self.verify(ts, scale_mutation_rate=True, include_ancestors=include_ancestors)

    @pytest.mark.parametrize("include_ancestors", [False])
    def test_ts_simple_n8_high_recomb(self, include_ancestors):
        ts = self.get_ts_simple_n8_high_recomb()
        self.verify(ts, scale_mutation_rate=True, include_ancestors=include_ancestors)

    @pytest.mark.parametrize("include_ancestors", [False])
    def test_ts_simple_n16(self, include_ancestors):
        ts = self.get_ts_simple_n16()
        self.verify(ts, scale_mutation_rate=True, include_ancestors=include_ancestors)

    @pytest.mark.parametrize("include_ancestors", [False])
    def test_ts_larger(self, include_ancestors):
        ts = self.get_ts_custom_pars(
            ref_panel_size=45, length=1e5, mean_r=1e-5, mean_mu=1e-5
        )
        self.verify(
            ts,
            scale_mutation_rate=True,
            include_ancestors=include_ancestors,
        )
