"""Collection of functions to run forwards and backwards algorithms on haploid genotype data, where the data is structured as variants x samples."""
import numpy as np

from lshmm import jit

MISSING = -1
NONCOPY = -2


@jit.numba_njit
def forwards_ls_hap(n, m, H, s, e, r, norm=True):
    """Matrix based haploid LS forward algorithm using numpy vectorisation."""
    # Initialise
    F = np.zeros((m, n))
    r_n = r / n

    if norm:

        c = np.zeros(m)
        for i in range(n):
            em_prob = 0
            if H[0, i] != NONCOPY:
                em_prob = e[0, np.int64(np.equal(H[0, i], s[0, 0]) or s[0, 0] == MISSING)]
            F[0, i] = 1 / n * em_prob
            c[0] += F[0, i]

        for i in range(n):
            F[0, i] *= 1 / c[0]

        # Forwards pass
        for l in range(1, m):
            for i in range(n):
                F[l, i] = F[l - 1, i] * (1 - r[l]) + r_n[l]
                em_prob = 0
                if H[l, i] != NONCOPY:
                    em_prob = e[l, np.int64(np.equal(H[l, i], s[0, l]) or s[0, l] == MISSING)]
                F[l, i] *= em_prob
                c[l] += F[l, i]

            for i in range(n):
                F[l, i] *= 1 / c[l]

        ll = np.sum(np.log10(c))

    else:

        c = np.ones(m)

        for i in range(n):
            em_prob = 0
            if H[0, i] != NONCOPY:
                em_prob = e[0, np.int64(np.equal(H[0, i], s[0, 0]) or s[0, 0] == MISSING)]
            F[0, i] = 1 / n * em_prob

        # Forwards pass
        for l in range(1, m):
            for i in range(n):
                F[l, i] = F[l - 1, i] * (1 - r[l]) + np.sum(F[l - 1, :]) * r_n[l]
                em_prob = 0
                if H[l, i] != NONCOPY:
                    em_prob = e[l, np.int64(np.equal(H[l, i], s[0, l]) or s[0, l] == MISSING)]
                F[l, i] *= em_prob

        ll = np.log10(np.sum(F[m - 1, :]))

    return F, c, ll


@jit.numba_njit
def backwards_ls_hap(n, m, H, s, e, c, r):
    """Matrix based haploid LS backward algorithm using numpy vectorisation."""
    # Initialise
    B = np.zeros((m, n))
    for i in range(n):
        B[m - 1, i] = 1
    r_n = r / n

    # Backwards pass
    for l in range(m - 2, -1, -1):
        tmp_B = np.zeros(n)
        tmp_B_sum = 0
        for i in range(n):
            em_prob = 0
            if H[l + 1, i] != NONCOPY:
                em_prob = e[l + 1, np.int64(np.equal(H[l + 1, i], s[0, l + 1]) or s[0, l + 1] == MISSING)]
            tmp_B[i] = em_prob * B[l + 1, i]
            tmp_B_sum += tmp_B[i]
        for i in range(n):
            B[l, i] = r_n[l + 1] * tmp_B_sum
            B[l, i] += (1 - r[l + 1]) * tmp_B[i]
            B[l, i] *= 1 / c[l + 1]

    return B
