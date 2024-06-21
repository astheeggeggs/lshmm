import pytest

from . import lsbase
import lshmm as ls
import lshmm.core as core
import lshmm.vit_diploid as vd


class TestViterbiDiploid(lsbase.ViterbiAlgorithmBase):
    def verify(self, ts, scale_mutation_rate, include_ancestors):
        ploidy = 2
        for n, m, H_vs, query, e_vs, r, mu in self.get_examples_pars(
            ts,
            ploidy=ploidy,
            scale_mutation_rate=scale_mutation_rate,
            include_ancestors=include_ancestors,
            include_extreme_rates=True,
        ):
            G_vs = core.convert_haplotypes_to_phased_genotypes(H_vs)
            s = core.convert_haplotypes_to_unphased_genotypes(query)

            V_vs, P_vs, ll_vs = vd.forwards_viterbi_dip_low_mem(
                n=n,
                m=m,
                G=G_vs,
                s=s,
                e=e_vs,
                r=r,
            )
            path_vs = vd.backwards_viterbi_dip(m=m, V_last=V_vs, P=P_vs)
            phased_path_vs = vd.get_phased_path(n=n, path=path_vs)
            path, ll = ls.viterbi(
                reference_panel=H_vs,
                query=query,
                ploidy=ploidy,
                prob_recombination=r,
                prob_mutation=mu,
                scale_mutation_rate=scale_mutation_rate,
            )

            self.assertAllClose(ll_vs, ll)
            self.assertAllClose(phased_path_vs, path)

    @pytest.mark.parametrize("scale_mutation_rate", [True, False])
    @pytest.mark.parametrize("include_ancestors", [True, False])
    def test_ts_simple_n10_no_recomb(self, scale_mutation_rate, include_ancestors):
        ts = self.get_ts_simple_n10_no_recomb()
        self.verify(ts, scale_mutation_rate, include_ancestors)

    @pytest.mark.parametrize("num_samples", [4, 8, 16])
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
            num_samples=30, seq_length=1e5, mean_r=1e-5, mean_mu=1e-5
        )
        self.verify(ts, scale_mutation_rate, include_ancestors)
