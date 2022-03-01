"""External API definitions."""
import numpy as np

from .forward_backward.fb_diploid_variants_samples import (
    backward_ls_dip_loop,
    forward_ls_dip_loop,
)
from .forward_backward.fb_haploid_variants_samples import (
    backwards_ls_hap,
    forwards_ls_hap,
)
from .vit_diploid_variants_samples import (
    backwards_viterbi_dip,
    forwards_viterbi_dip_low_mem,
    get_phased_path,
)
from .vit_haploid_variants_samples import (
    backwards_viterbi_hap,
    forwards_viterbi_hap_lower_mem_rescaling,
)

EQUAL_BOTH_HOM = 4
UNEQUAL_BOTH_HOM = 0
BOTH_HET = 7
REF_HOM_OBS_HET = 1
REF_HET_OBS_HOM = 2


def forwards(n, m, G_or_H, s, e, r):
    """
    Run the Li and Stephens forwards algorithm on haplotype or
    unphased genotype data.
    """
    template_dimensions = G_or_H.shape
    assert len(template_dimensions) in [2, 3]

    if len(template_dimensions) == 2:
        # Haploid
        assert (G_or_H.shape == np.array([m, n])).all()
        F, c, ll = forwards_ls_hap(n, m, G_or_H, s, e, r, norm=True)
    else:
        # Diploid
        assert (G_or_H.shape == np.array([m, n, n])).all()
        F, c, ll = forward_ls_dip_loop(n, m, G_or_H, s, e, r, norm=True)

    return F, c, ll


def backwards(n, m, G_or_H, s, e, c, r):
    """
    Run the Li and Stephens backwards algorithm on haplotype or
    unphased genotype data.
    """
    template_dimensions = G_or_H.shape
    assert len(template_dimensions) in [2, 3]

    if len(template_dimensions) == 2:
        # Haploid
        assert (G_or_H.shape == np.array([m, n])).all()
        B = backwards_ls_hap(n, m, G_or_H, s, e, c, r)
    else:
        # Diploid
        assert (G_or_H.shape == np.array([m, n, n])).all()
        B = backward_ls_dip_loop(n, m, G_or_H, s, e, c, r)

    return B


def viterbi(n, m, G_or_H, s, e, r):
    """
    Run the Li and Stephens Viterbi algorithm on haplotype or
    unphased genotype data.
    """
    template_dimensions = G_or_H.shape
    assert len(template_dimensions) in [2, 3]

    if len(template_dimensions) == 2:
        # Haploid
        assert (G_or_H.shape == np.array([m, n])).all()
        V, P, ll = forwards_viterbi_hap_lower_mem_rescaling(n, m, G_or_H, s, e, r)
        path = backwards_viterbi_hap(m, V, P)
    else:
        # Diploid
        assert (G_or_H.shape == np.array([m, n, n])).all()
        V, P, ll = forwards_viterbi_dip_low_mem(n, m, G_or_H, s, e, r)
        unphased_path = backwards_viterbi_dip(m, V, P)
        path = get_phased_path(n, unphased_path)

    return path, ll
