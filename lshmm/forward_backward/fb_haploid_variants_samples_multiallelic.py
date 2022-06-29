"""Collection of functions to run forwards and backwards algorithms on haploid genotype data, where the data is structured as variants x samples."""
import numba as nb
import numpy as np


def check_alleles(alleles, m):
    """
    Checks the specified allele list and returns a list of lists
    of alleles of length num_sites.
    If alleles is a 1D list of strings, assume that this list is used
    for each site and return num_sites copies of this list.
    Otherwise, raise a ValueError if alleles is not a list of length
    num_sites.
    """
    if isinstance(alleles[0], str):
        return np.int8([len(alleles) for _ in range(m)])
    if len(alleles) != m:
        raise ValueError("Malformed alleles list")
    n_alleles = np.int8([(len(alleles_site)) for alleles_site in alleles])
    return n_alleles


@nb.jit
def forwards_ls_hap(n, m, n_alleles, H, s, e, r, norm=True):
    """Matrix based haploid LS forward algorithm using numpy vectorisation."""
    # Initialise
    F = np.zeros((m, n))
    r_n = r / n

    if norm:

        c = np.zeros(m)
        for i in range(n):
            F[0, i] = 1 / n * e[0, np.int64(np.equal(H[0, i], s[0, 0]))]
            c[0] += F[0, i]

        for i in range(n):
            F[0, i] *= 1 / c[0]

        # Forwards pass
        for l in range(1, m):
            for i in range(n):
                F[l, i] = F[l - 1, i] * (1 - r[l]) + r_n[l]
                F[l, i] *= e[l, np.int64(np.equal(H[l, i], s[0, l]))]
                c[l] += F[l, i]

            for i in range(n):
                F[l, i] *= 1 / c[l]

        ll = np.sum(np.log10(c))

    else:

        c = np.ones(m)

        for i in range(n):
            F[0, i] = 1 / n * e[0, np.int64(np.equal(H[0, i], s[0, 0]))]

        # Forwards pass
        for l in range(1, m):
            for i in range(n):
                F[l, i] = F[l - 1, i] * (1 - r[l]) + np.sum(F[l - 1, :]) * r_n[l]
                F[l, i] *= e[l, np.int64(np.equal(H[l, i], s[0, l]))]

        ll = np.log10(np.sum(F[m - 1, :]))

    return F, c, ll


@nb.jit
def backwards_ls_hap(n, m, n_alleles, H, s, e, c, r):
    """Matrix based haploid LS backward algorithm using numpy vectorisation."""
    # Initialise
    # alleles = check_alleles(alleles, m)
    B = np.zeros((m, n))
    for i in range(n):
        B[m - 1, i] = 1
    r_n = r / n

    # Backwards pass
    for l in range(m - 2, -1, -1):
        tmp_B = np.zeros(n)
        tmp_B_sum = 0
        for i in range(n):
            tmp_B[i] = (
                e[l + 1, np.int64(np.equal(H[l + 1, i], s[0, l + 1]))] * B[l + 1, i]
            )
            tmp_B_sum += tmp_B[i]
        for i in range(n):
            B[l, i] = r_n[l + 1] * tmp_B_sum
            B[l, i] += (1 - r[l + 1]) * tmp_B[i]
            B[l, i] *= 1 / c[l + 1]

    return B


def forwards_ls_hap_wrapper(n, m, alleles, H, s, e, r, norm=True):
    n_alleles = check_alleles(alleles, m)
    F, c, ll = forwards_ls_hap(n, m, n_alleles, H, s, e, r, norm)
    return F, c, ll


def backwards_ls_hap_wrapper(n, m, alleles, H, s, e, c, r):
    n_alleles = check_alleles(alleles, m)
    B = backwards_ls_hap(n, m, n_alleles, H, s, e, c, r)
    return B
