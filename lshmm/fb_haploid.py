"""
Various implementations of the Li & Stephens forwards-backwards algorithm on haploid genotype data,
where the data is structured as variants x samples.
"""

import numpy as np

from lshmm import core
from lshmm import jit


@jit.numba_njit
def forwards_ls_hap(n, m, H, s, e, r, norm=True):
    """
    A matrix-based implementation using Numpy vectorisation.

    This is exposed via the API.
    """
    F = np.zeros((m, n))
    r_n = r / n

    if norm:
        c = np.zeros(m)
        for i in range(n):
            emission_index = core.get_index_in_emission_matrix_haploid(
                ref_allele=H[0, i], query_allele=s[0, 0]
            )
            F[0, i] = 1 / n * e[0, emission_index]
            c[0] += F[0, i]

        for i in range(n):
            F[0, i] *= 1 / c[0]

        # Forwards pass
        for l in range(1, m):
            for i in range(n):
                F[l, i] = F[l - 1, i] * (1 - r[l]) + r_n[l]
                emission_index = core.get_index_in_emission_matrix_haploid(
                    ref_allele=H[l, i], query_allele=s[0, l]
                )
                F[l, i] *= e[l, emission_index]
                c[l] += F[l, i]

            for i in range(n):
                F[l, i] *= 1 / c[l]

        ll = np.sum(np.log10(c))

    else:
        c = np.ones(m)
        for i in range(n):
            emission_index = core.get_index_in_emission_matrix_haploid(
                ref_allele=H[0, i], query_allele=s[0, 0]
            )
            F[0, i] = 1 / n * e[0, emission_index]

        # Forwards pass
        for l in range(1, m):
            for i in range(n):
                F[l, i] = F[l - 1, i] * (1 - r[l]) + np.sum(F[l - 1, :]) * r_n[l]
                emission_index = core.get_index_in_emission_matrix_haploid(
                    ref_allele=H[l, i], query_allele=s[0, l]
                )
                F[l, i] *= e[l, emission_index]

        ll = np.log10(np.sum(F[m - 1, :]))

    return F, c, ll


@jit.numba_njit
def backwards_ls_hap(n, m, H, s, e, c, r):
    """
    A matrix-based implementation using Numpy vectorisation.

    This is exposed via the API.
    """
    B = np.zeros((m, n))
    for i in range(n):
        B[m - 1, i] = 1
    r_n = r / n

    # Backwards pass
    for l in range(m - 2, -1, -1):
        tmp_B = np.zeros(n)
        tmp_B_sum = 0
        for i in range(n):
            emission_index = core.get_index_in_emission_matrix_haploid(
                ref_allele=H[l + 1, i], query_allele=s[0, l + 1]
            )
            tmp_B[i] = e[l + 1, emission_index] * B[l + 1, i]
            tmp_B_sum += tmp_B[i]
        for i in range(n):
            B[l, i] = r_n[l + 1] * tmp_B_sum
            B[l, i] += (1 - r[l + 1]) * tmp_B[i]
            B[l, i] *= 1 / c[l + 1]

    return B
