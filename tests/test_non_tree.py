import numpy as np
import numba as nb

from . import lsbase
import lshmm.fb_diploid as fbd
import lshmm.fb_haploid as fbh
import lshmm.vit_diploid as vd
import lshmm.vit_haploid as vh


class TestNonTreeForwardBackwardHaploid(lsbase.ForwardBackwardAlgorithmBase):
    def verify(self, ts, scale_mutation_rate, mean_r=None, mean_mu=None):
        for n, m, H_vs, s, e_vs, r, _ in self.get_examples_pars(
            ts,
            ploidy=1,
            scale_mutation_rate=scale_mutation_rate,
            mean_r=mean_r,
            mean_mu=mean_mu,
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

    def test_ts_simple_n10_no_recomb(self):
        ts = self.get_ts_simple_n10_no_recomb()
        self.verify(ts, scale_mutation_rate=True)

    def test_ts_simple_n6(self):
        ts = self.get_ts_simple_n6()
        self.verify(ts, scale_mutation_rate=True)

    def test_ts_simple_n8(self):
        ts = self.get_ts_simple_n8()
        self.verify(ts, scale_mutation_rate=True)

    def test_ts_simple_n8_high_recomb(self):
        ts = self.get_ts_simple_n8_high_recomb()
        self.verify(ts, scale_mutation_rate=True)

    def test_ts_simple_n16(self):
        ts = self.get_ts_simple_n16()
        self.verify(ts, scale_mutation_rate=True)

    def test_larger(self):
        ref_panel_size = 45
        length = 1e5
        mean_r = 1e-5
        mean_mu = 1e-5
        ts = self.get_ts_custom_pars(
            ref_panel_size=ref_panel_size, length=length, mean_r=mean_r, mean_mu=mean_mu
        )
        self.verify(ts, scale_mutation_rate=True, mean_r=mean_r, mean_mu=mean_mu)


class TestNonTreeForwardBackwardDiploid(lsbase.ForwardBackwardAlgorithmBase):
    def verify(self, ts, scale_mutation_rate, mean_r=None, mean_mu=None):
        for n, m, G_vs, s, e_vs, r, _ in self.get_examples_pars(
            ts,
            ploidy=2,
            scale_mutation_rate=scale_mutation_rate,
            mean_r=mean_r,
            mean_mu=mean_mu,
        ):
            F_vs, c_vs, ll_vs = fbd.forwards_ls_dip(n, m, G_vs, s, e_vs, r, norm=True)
            B_vs = fbd.backwards_ls_dip(n, m, G_vs, s, e_vs, c_vs, r)
            self.assertAllClose(np.sum(F_vs * B_vs, (1, 2)), np.ones(m))

            F_tmp, c_tmp, ll_tmp = fbd.forwards_ls_dip(
                n, m, G_vs, s, e_vs, r, norm=False
            )
            if ll_tmp != -np.inf:
                B_tmp = fbd.backwards_ls_dip(n, m, G_vs, s, e_vs, c_tmp, r)
                self.assertAllClose(
                    np.log10(np.sum(F_tmp * B_tmp, (1, 2))), ll_tmp * np.ones(m)
                )
                self.assertAllClose(ll_vs, ll_tmp)

            F_tmp, ll_tmp = fbd.forward_ls_dip_starting_point(n, m, G_vs, s, e_vs, r)
            if ll_tmp != -np.inf:
                B_tmp = fbd.backward_ls_dip_starting_point(n, m, G_vs, s, e_vs, r)
                self.assertAllClose(
                    np.log10(np.sum(F_tmp * B_tmp, (1, 2))), ll_tmp * np.ones(m)
                )
                self.assertAllClose(ll_vs, ll_tmp)

            F_tmp, c_tmp, ll_tmp = fbd.forward_ls_dip_loop(
                n, m, G_vs, s, e_vs, r, norm=False
            )
            if ll_tmp != -np.inf:
                B_tmp = fbd.backward_ls_dip_loop(n, m, G_vs, s, e_vs, c_tmp, r)
                self.assertAllClose(
                    np.log10(np.sum(F_tmp * B_tmp, (1, 2))), ll_tmp * np.ones(m)
                )
                self.assertAllClose(ll_vs, ll_tmp)

            F_tmp, c_tmp, ll_tmp = fbd.forward_ls_dip_loop(
                n, m, G_vs, s, e_vs, r, norm=True
            )
            B_tmp = fbd.backward_ls_dip_loop(n, m, G_vs, s, e_vs, c_tmp, r)
            self.assertAllClose(np.sum(F_tmp * B_tmp, (1, 2)), np.ones(m))
            self.assertAllClose(ll_vs, ll_tmp)

    def test_ts_simple_n10_no_recomb(self):
        ts = self.get_ts_simple_n10_no_recomb()
        self.verify(ts, scale_mutation_rate=True)

    def test_ts_simple_n6(self):
        ts = self.get_ts_simple_n6()
        self.verify(ts, scale_mutation_rate=True)

    def test_ts_simple_n8(self):
        ts = self.get_ts_simple_n8()
        self.verify(ts, scale_mutation_rate=True)

    def test_ts_simple_n8_high_recomb(self):
        ts = self.get_ts_simple_n8_high_recomb()
        self.verify(ts, scale_mutation_rate=True)

    def test_ts_simple_n16(self):
        ts = self.get_ts_simple_n16()
        self.verify(ts, scale_mutation_rate=True)

    def test_ts_larger(self):
        ref_panel_size = 45
        length = 1e5
        mean_r = 1e-5
        mean_mu = 1e-5
        ts = self.get_ts_custom_pars(
            ref_panel_size=ref_panel_size, length=length, mean_r=mean_r, mean_mu=mean_mu
        )
        self.verify(ts, scale_mutation_rate=True, mean_r=mean_r, mean_mu=mean_mu)


class TestNonTreeViterbiHaploid(lsbase.ViterbiAlgorithmBase):
    def verify(self, ts, scale_mutation_rate, mean_r=None, mean_mu=None):
        for n, m, H_vs, s, e_vs, r, _ in self.get_examples_pars(
            ts,
            ploidy=1,
            scale_mutation_rate=scale_mutation_rate,
            mean_r=mean_r,
            mean_mu=mean_mu,
        ):
            V_vs, P_vs, ll_vs = vh.forwards_viterbi_hap_naive(n, m, H_vs, s, e_vs, r)
            path_vs = vh.backwards_viterbi_hap(m, V_vs[m - 1, :], P_vs)
            ll_check = vh.path_ll_hap(n, m, H_vs, path_vs, s, e_vs, r)
            self.assertAllClose(ll_vs, ll_check)

            V_tmp, P_tmp, ll_tmp = vh.forwards_viterbi_hap_naive_vec(
                n, m, H_vs, s, e_vs, r
            )
            path_tmp = vh.backwards_viterbi_hap(m, V_tmp[m - 1, :], P_tmp)
            ll_check = vh.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            V_tmp, P_tmp, ll_tmp = vh.forwards_viterbi_hap_naive_low_mem(
                n, m, H_vs, s, e_vs, r
            )
            path_tmp = vh.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            V_tmp, P_tmp, ll_tmp = vh.forwards_viterbi_hap_naive_low_mem_rescaling(
                n, m, H_vs, s, e_vs, r
            )
            path_tmp = vh.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            V_tmp, P_tmp, ll_tmp = vh.forwards_viterbi_hap_low_mem_rescaling(
                n, m, H_vs, s, e_vs, r
            )
            path_tmp = vh.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            V_tmp, P_tmp, ll_tmp = vh.forwards_viterbi_hap_lower_mem_rescaling(
                n, m, H_vs, s, e_vs, r
            )
            path_tmp = vh.backwards_viterbi_hap(m, V_tmp, P_tmp)
            ll_check = vh.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

            (
                V_tmp,
                V_argmaxes_tmp,
                recombs,
                ll_tmp,
            ) = vh.forwards_viterbi_hap_lower_mem_rescaling_no_pointer(
                n, m, H_vs, s, e_vs, r
            )
            path_tmp = vh.backwards_viterbi_hap_no_pointer(
                m,
                V_argmaxes_tmp,
                nb.typed.List(recombs),
            )
            ll_check = vh.path_ll_hap(n, m, H_vs, path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, ll_check)
            self.assertAllClose(ll_vs, ll_tmp)

    def test_ts_simple_n10_no_recombn(self):
        ts = self.get_ts_simple_n10_no_recomb()
        self.verify(ts, scale_mutation_rate=True)

    def test_ts_simple_n6(self):
        ts = self.get_ts_simple_n6()
        self.verify(ts, scale_mutation_rate=True)

    def test_ts_simple_n8(self):
        ts = self.get_ts_simple_n8()
        self.verify(ts, scale_mutation_rate=True)

    def test_ts_simple_n8_high_recomb(self):
        ts = self.get_ts_simple_n8_high_recomb()
        self.verify(ts, scale_mutation_rate=True)

    def test_ts_simple_n16(self):
        ts = self.get_ts_simple_n16()
        self.verify(ts, scale_mutation_rate=True)

    def test_ts_larger(self):
        ref_panel_size = 45
        length = 1e5
        mean_r = 1e-5
        mean_mu = 1e-5
        ts = self.get_ts_custom_pars(
            ref_panel_size=ref_panel_size, length=length, mean_r=mean_r, mean_mu=mean_mu
        )
        self.verify(ts, scale_mutation_rate=True, mean_r=mean_r, mean_mu=mean_mu)


class TestNonTreeViterbiDiploid(lsbase.ViterbiAlgorithmBase):
    def verify(self, ts, scale_mutation_rate, mean_r=None, mean_mu=None):
        for n, m, G_vs, s, e_vs, r, _ in self.get_examples_pars(
            ts,
            ploidy=2,
            scale_mutation_rate=scale_mutation_rate,
            mean_r=mean_r,
            mean_mu=mean_mu,
        ):
            V_vs, P_vs, ll_vs = vd.forwards_viterbi_dip_naive(n, m, G_vs, s, e_vs, r)
            path_vs = vd.backwards_viterbi_dip(m, V_vs[m - 1, :, :], P_vs)
            phased_path_vs = vd.get_phased_path(n, path_vs)
            path_ll_vs = vd.path_ll_dip(n, m, G_vs, phased_path_vs, s, e_vs, r)
            self.assertAllClose(ll_vs, path_ll_vs)

            V_tmp, P_tmp, ll_tmp = vd.forwards_viterbi_dip_naive_low_mem(
                n, m, G_vs, s, e_vs, r
            )
            path_tmp = vd.backwards_viterbi_dip(m, V_tmp, P_tmp)
            phased_path_tmp = vd.get_phased_path(n, path_tmp)
            path_ll_tmp = vd.path_ll_dip(n, m, G_vs, phased_path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, path_ll_tmp)
            self.assertAllClose(ll_vs, ll_tmp)

            V_tmp, P_tmp, ll_tmp = vd.forwards_viterbi_dip_low_mem(
                n, m, G_vs, s, e_vs, r
            )
            path_tmp = vd.backwards_viterbi_dip(m, V_tmp, P_tmp)
            phased_path_tmp = vd.get_phased_path(n, path_tmp)
            path_ll_tmp = vd.path_ll_dip(n, m, G_vs, phased_path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, path_ll_tmp)
            self.assertAllClose(ll_vs, ll_tmp)

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

            V_tmp, P_tmp, ll_tmp = vd.forwards_viterbi_dip_naive_vec(
                n, m, G_vs, s, e_vs, r
            )
            path_tmp = vd.backwards_viterbi_dip(m, V_tmp[m - 1, :, :], P_tmp)
            phased_path_tmp = vd.get_phased_path(n, path_tmp)
            path_ll_tmp = vd.path_ll_dip(n, m, G_vs, phased_path_tmp, s, e_vs, r)
            self.assertAllClose(ll_tmp, path_ll_tmp)
            self.assertAllClose(ll_vs, ll_tmp)

    def test_ts_simple_n10_no_recomb(self):
        ts = self.get_ts_simple_n10_no_recomb()
        self.verify(ts, scale_mutation_rate=True)

    def test_ts_simple_n6(self):
        ts = self.get_ts_simple_n6()
        self.verify(ts, scale_mutation_rate=True)

    def test_ts_simple_n8(self):
        ts = self.get_ts_simple_n8()
        self.verify(ts, scale_mutation_rate=True)

    def test_ts_simple_n8_high_recomb(self):
        ts = self.get_ts_simple_n8_high_recomb()
        self.verify(ts, scale_mutation_rate=True)

    def test_ts_simple_n16(self):
        ts = self.get_ts_simple_n16()
        self.verify(ts, scale_mutation_rate=True)

    def test_ts_larger(self):
        ref_panel_size = 45
        length = 1e5
        mean_r = 1e-5
        mean_mu = 1e-5
        ts = self.get_ts_custom_pars(
            ref_panel_size=ref_panel_size, length=length, mean_r=mean_r, mean_mu=mean_mu
        )
        self.verify(ts, scale_mutation_rate=True, mean_r=mean_r, mean_mu=mean_mu)
