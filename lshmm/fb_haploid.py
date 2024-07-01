"""Implementations of the Li & Stephens forwards-backwards algorithm on haploid genotype data."""

import numpy as np

from lshmm import core
from lshmm import jit


@jit.numba_njit
def forwards_ls_hap(
    n,
    m,
    H,
    s,
    e,
    r,
    emission_func,
    norm=True,
):
    """
    A matrix-based implementation using Numpy.

    This is exposed via the API.
    """
    F = np.zeros((m, n))
    num_copiable_entries = core.get_num_copiable_entries(H)
    r_n = r / num_copiable_entries

    if norm:
        c = np.zeros(m)
        for i in range(n):
            emission_prob = emission_func(
                ref_allele=H[0, i],
                query_allele=s[0, 0],
                site=0,
                emission_matrix=e,
            )
            F[0, i] = 1 / n * emission_prob
            c[0] += F[0, i]

        for i in range(n):
            F[0, i] *= 1 / c[0]

        # Forwards pass
        for l in range(1, m):
            for i in range(n):
                F[l, i] = F[l - 1, i] * (1 - r[l]) + r_n[l]
                emission_prob = emission_func(
                    ref_allele=H[l, i],
                    query_allele=s[0, l],
                    site=l,
                    emission_matrix=e,
                )
                F[l, i] *= emission_prob
                c[l] += F[l, i]

            for i in range(n):
                F[l, i] *= 1 / c[l]

        ll = np.sum(np.log10(c))

    else:
        c = np.ones(m)
        for i in range(n):
            emission_prob = emission_func(
                ref_allele=H[0, i],
                query_allele=s[0, 0],
                site=0,
                emission_matrix=e,
            )
            F[0, i] = 1 / n * emission_prob

        # Forwards pass
        for l in range(1, m):
            for i in range(n):
                F[l, i] = F[l - 1, i] * (1 - r[l]) + np.sum(F[l - 1, :]) * r_n[l]
                emission_prob = emission_func(
                    ref_allele=H[l, i],
                    query_allele=s[0, l],
                    site=l,
                    emission_matrix=e,
                )
                F[l, i] *= emission_prob

        ll = np.log10(np.sum(F[m - 1, :]))

    return F, c, ll


@jit.numba_njit
def backwards_ls_hap(
    n,
    m,
    H,
    s,
    e,
    c,
    r,
    emission_func,
):
    """
    A matrix-based implementation using Numpy.

    This is exposed via the API.
    """
    B = np.zeros((m, n))
    for i in range(n):
        B[m - 1, i] = 1
    num_copiable_entries = core.get_num_copiable_entries(H)
    r_n = r / num_copiable_entries

    # Backwards pass
    for l in range(m - 2, -1, -1):
        tmp_B = np.zeros(n)
        tmp_B_sum = 0
        for i in range(n):
            emission_prob = emission_func(
                ref_allele=H[l + 1, i],
                query_allele=s[0, l + 1],
                site=l + 1,
                emission_matrix=e,
            )
            tmp_B[i] = emission_prob * B[l + 1, i]
            tmp_B_sum += tmp_B[i]
        for i in range(n):
            B[l, i] = r_n[l + 1] * tmp_B_sum
            B[l, i] += (1 - r[l + 1]) * tmp_B[i]
            B[l, i] *= 1 / c[l + 1]

    return B
