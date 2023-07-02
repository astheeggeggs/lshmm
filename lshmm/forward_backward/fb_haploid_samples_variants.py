"""Collection of functions to run forwards and backwards algorithms on haploid genotype data, where the data is structured as samples x variants."""
import numpy as np

from lshmm import jit

MISSING = -1


@jit.numba_njit
def forwards_ls_hap(n, m, H, s, e, r, norm=True):
    """Matrix based haploid LS forward algorithm using numpy vectorisation."""
    # Initialise
    F = np.zeros((n, m))
    r_n = r / n

    if norm:

        c = np.zeros(m)
        for i in range(n):
            F[i, 0] = (
                1 / n * e[np.int64(np.equal(H[i, 0], s[0, 0]) or s[0, 0] == MISSING), 0]
            )
            c[0] += F[i, 0]

        for i in range(n):
            F[i, 0] *= 1 / c[0]

        # Forwards pass
        for l in range(1, m):
            for i in range(n):
                F[i, l] = F[i, l - 1] * (1 - r[l]) + r_n[l]
                F[i, l] *= e[
                    np.int64(np.equal(H[i, l], s[0, l]) or s[0, l] == MISSING), l
                ]
                c[l] += F[i, l]

            for i in range(n):
                F[i, l] *= 1 / c[l]

        ll = np.sum(np.log10(c))

    else:
        c = np.ones(m)

        for i in range(n):
            F[i, 0] = (
                1 / n * e[np.int64(np.equal(H[i, 0], s[0, 0]) or s[0, 0] == MISSING), 0]
            )
        # Forwards pass
        for l in range(1, m):
            for i in range(n):
                F[i, l] = F[i, l - 1] * (1 - r[l]) + np.sum(F[:, l - 1]) * r_n[l]
                F[i, l] *= e[
                    np.int64(np.equal(H[i, l], s[0, l]) or s[0, l] == MISSING), l
                ]

        ll = np.log10(np.sum(F[:, m - 1]))

    return F, c, ll


@jit.numba_njit
def backwards_ls_hap(n, m, H, s, e, c, r):
    """Matrix based haploid LS backward algorithm using numpy vectorisation."""
    # Initialise
    B = np.zeros((n, m))
    for i in range(n):
        B[i, m - 1] = 1
    r_n = r / n

    # Backwards pass
    for l in range(m - 2, -1, -1):
        tmp_B = np.zeros(n)
        tmp_B_sum = 0
        for i in range(n):
            tmp_B[i] = (
                e[
                    np.int64(
                        np.equal(H[i, l + 1], s[0, l + 1]) or s[0, l + 1] == MISSING
                    ),
                    l + 1,
                ]
                * B[i, l + 1]
            )
            tmp_B_sum += tmp_B[i]
        for i in range(n):
            B[i, l] = r_n[l + 1] * tmp_B_sum
            B[i, l] += (1 - r[l + 1]) * tmp_B[i]
            B[i, l] *= 1 / c[l + 1]

    return B
