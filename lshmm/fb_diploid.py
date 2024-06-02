"""
Various implementations of the Li & Stephens forwards-backwards algorithm on diploid genotype data,
where the data is structured as variants x samples x samples.
"""

import numpy as np

from lshmm import core
from lshmm import jit


def forwards_ls_dip(n, m, G, s, e, r, norm=True):
    """A matrix-based implementation using Numpy."""
    # Initialise
    F = np.zeros((m, n, n))
    F[0, :, :] = 1 / (n**2)
    c = np.ones(m)
    r_n = r / n

    emission_probs = core.get_emission_probability_diploid_genotypes(
        ref_genotypes=G[0, :, :],
        query_genotype=s[0, 0],
        site=0,
        emission_matrix=e,
    )
    F[0, :, :] *= emission_probs

    if norm:
        c[0] = np.sum(F[0, :, :])
        F[0, :, :] *= 1 / c[0]

        # Forwards
        for l in range(1, m):
            emission_probs = core.get_emission_probability_diploid_genotypes(
                ref_genotypes=G[l, :, :],
                query_genotype=s[0, l],
                site=l,
                emission_matrix=e,
            )

            # No change in both
            F[l, :, :] = (1 - r[l]) ** 2 * F[l - 1, :, :]

            # Both change
            F[l, :, :] += (r_n[l]) ** 2

            # One changes
            sum_j = core.np_sum(F[l - 1, :, :], 0).repeat(n).reshape((-1, n)).T
            F[l, :, :] += ((1 - r[l]) * r_n[l]) * (sum_j + sum_j.T)

            # Emission
            F[l, :, :] *= emission_probs
            c[l] = np.sum(F[l, :, :])
            F[l, :, :] *= 1 / c[l]

        ll = np.sum(np.log10(c))
    else:
        # Forwards
        for l in range(1, m):
            emission_probs = core.get_emission_probability_diploid_genotypes(
                ref_genotypes=G[l, :, :],
                query_genotype=s[0, l],
                site=l,
                emission_matrix=e,
            )

            # No change in both
            F[l, :, :] = (1 - r[l]) ** 2 * F[l - 1, :, :]

            # Both change
            F[l, :, :] += (r_n[l]) ** 2 * np.sum(F[l - 1, :, :])

            # One changes
            sum_j = core.np_sum(F[l - 1, :, :], 0).repeat(n).reshape((-1, n)).T
            # sum_j2 = np_sum(F[l - 1, :, :], 1).repeat(n).reshape((-1, n))
            F[l, :, :] += ((1 - r[l]) * r_n[l]) * (sum_j + sum_j.T)

            # Emission
            F[l, :, :] *= emission_probs

        ll = np.log10(np.sum(F[l, :, :]))

    return F, c, ll


def backwards_ls_dip(n, m, G, s, e, c, r):
    """A matrix-based implementation using Numpy."""
    # Initialise
    B = np.zeros((m, n, n))
    B[m - 1, :, :] = 1
    r_n = r / n

    # Backwards
    for l in range(m - 2, -1, -1):
        emission_probs = core.get_emission_probability_diploid_genotypes(
            ref_genotypes=G[l + 1, :, :],
            query_genotype=s[0, l + 1],
            site=l + 1,
            emission_matrix=e,
        )

        # No change in both
        B[l, :, :] = r_n[l + 1] ** 2 * np.sum(emission_probs * B[l + 1, :, :])

        # Both change
        B[l, :, :] += (1 - r[l + 1]) ** 2 * B[l + 1, :, :] * emission_probs

        # One changes
        sum_j = (
            core.np_sum(B[l + 1, :, :] * emission_probs, 0).repeat(n).reshape((-1, n))
        )
        B[l, :, :] += ((1 - r[l + 1]) * r_n[l + 1]) * (sum_j + sum_j.T)
        B[l, :, :] *= 1 / c[l + 1]

    return B


@jit.numba_njit
def forward_ls_dip_starting_point(n, m, G, s, e, r):
    """A naive implementation."""
    # Initialise
    F = np.zeros((m, n, n))
    r_n = r / n

    for j1 in range(n):
        for j2 in range(n):
            F[0, j1, j2] = 1 / (n**2)
            emission_prob = core.get_emission_probability_diploid(
                ref_genotype=G[0, j1, j2],
                query_genotype=s[0, 0],
                site=0,
                emission_matrix=e,
            )
            F[0, j1, j2] *= emission_prob

    for l in range(1, m):
        F_no_change = np.zeros((n, n))
        F_j1_change = np.zeros(n)
        F_j2_change = np.zeros(n)
        F_both_change = 0

        for j1 in range(n):
            for j2 in range(n):
                F_no_change[j1, j2] = (1 - r[l]) ** 2 * F[l - 1, j1, j2]

        for j1 in range(n):
            for j2 in range(n):
                F_both_change += r_n[l] ** 2 * F[l - 1, j1, j2]

        for j1 in range(n):
            for j2 in range(n):  # This is the variable to sum over - it changes
                F_j2_change[j1] += (1 - r[l]) * r_n[l] * F[l - 1, j1, j2]

        for j2 in range(n):
            for j1 in range(n):  # This is the variable to sum over - it changes
                F_j1_change[j2] += (1 - r[l]) * r_n[l] * F[l - 1, j1, j2]

        F[l, :, :] = F_both_change

        for j1 in range(n):
            F[l, j1, :] += F_j2_change

        for j2 in range(n):
            F[l, :, j2] += F_j1_change

        for j1 in range(n):
            for j2 in range(n):
                F[l, j1, j2] += F_no_change[j1, j2]

        for j1 in range(n):
            for j2 in range(n):
                emission_prob = core.get_emission_probability_diploid(
                    ref_genotype=G[l, j1, j2],
                    query_genotype=s[0, l],
                    site=l,
                    emission_matrix=e,
                )
                F[l, j1, j2] *= emission_prob

    ll = np.log10(np.sum(F[l, :, :]))

    return F, ll


@jit.numba_njit
def backward_ls_dip_starting_point(n, m, G, s, e, r):
    """A naive implementation."""
    # Initialise
    B = np.zeros((m, n, n))
    B[m - 1, :, :] = 1
    r_n = r / n

    for l in range(m - 2, -1, -1):
        B_no_change = np.zeros((n, n))
        B_j1_change = np.zeros(n)
        B_j2_change = np.zeros(n)
        B_both_change = 0

        # Evaluate the emission matrix at this site, for all pairs
        e_tmp = np.zeros((n, n))
        for j1 in range(n):
            for j2 in range(n):
                emission_prob = core.get_emission_probability_diploid(
                    ref_genotype=G[l + 1, j1, j2],
                    query_genotype=s[0, l + 1],
                    site=l + 1,
                    emission_matrix=e,
                )
                e_tmp[j1, j2] = emission_prob

        for j1 in range(n):
            for j2 in range(n):
                B_no_change[j1, j2] = (
                    (1 - r[l + 1]) ** 2 * B[l + 1, j1, j2] * e_tmp[j1, j2]
                )

        for j1 in range(n):
            for j2 in range(n):
                B_both_change += r_n[l + 1] ** 2 * e_tmp[j1, j2] * B[l + 1, j1, j2]

        for j1 in range(n):
            for j2 in range(n):  # This is the variable to sum over - it changes
                B_j2_change[j1] += (
                    (1 - r[l + 1]) * r_n[l + 1] * B[l + 1, j1, j2] * e_tmp[j1, j2]
                )

        for j2 in range(n):
            for j1 in range(n):  # This is the variable to sum over - it changes
                B_j1_change[j2] += (
                    (1 - r[l + 1]) * r_n[l + 1] * B[l + 1, j1, j2] * e_tmp[j1, j2]
                )

        B[l, :, :] = B_both_change

        for j1 in range(n):
            B[l, j1, :] += B_j2_change

        for j2 in range(n):
            B[l, :, j2] += B_j1_change

        for j1 in range(n):
            for j2 in range(n):
                B[l, j1, j2] += B_no_change[j1, j2]

    return B


@jit.numba_njit
def forward_ls_dip_loop(n, m, G, s, e, r, norm=True):
    """
    An implementation without vectorisation.

    This is exposed via the API.
    """
    # Initialise
    F = np.zeros((m, n, n))
    for j1 in range(n):
        for j2 in range(n):
            F[0, j1, j2] = 1 / (n**2)
            emission_prob = core.get_emission_probability_diploid(
                ref_genotype=G[0, j1, j2],
                query_genotype=s[0, 0],
                site=0,
                emission_matrix=e,
            )
            F[0, j1, j2] *= emission_prob
    r_n = r / n
    c = np.ones(m)

    if norm:
        c[0] = np.sum(F[0, :, :])
        F[0, :, :] *= 1 / c[0]

        for l in range(1, m):
            F_no_change = np.zeros((n, n))
            F_j_change = np.zeros(n)

            for j1 in range(n):
                for j2 in range(n):
                    F_no_change[j1, j2] = (1 - r[l]) ** 2 * F[l - 1, j1, j2]
                    F_j_change[j1] += (1 - r[l]) * r_n[l] * F[l - 1, j2, j1]

            F[l, :, :] = r_n[l] ** 2

            for j1 in range(n):
                F[l, j1, :] += F_j_change
                F[l, :, j1] += F_j_change
                for j2 in range(n):
                    F[l, j1, j2] += F_no_change[j1, j2]

            for j1 in range(n):
                for j2 in range(n):
                    emission_prob = core.get_emission_probability_diploid(
                        ref_genotype=G[l, j1, j2],
                        query_genotype=s[0, l],
                        site=l,
                        emission_matrix=e,
                    )
                    F[l, j1, j2] *= emission_prob

            c[l] = np.sum(F[l, :, :])
            F[l, :, :] *= 1 / c[l]

        ll = np.sum(np.log10(c))

    else:
        for l in range(1, m):
            F_no_change = np.zeros((n, n))
            F_j1_change = np.zeros(n)
            F_j2_change = np.zeros(n)
            F_both_change = 0

            for j1 in range(n):
                for j2 in range(n):
                    F_no_change[j1, j2] = (1 - r[l]) ** 2 * F[l - 1, j1, j2]
                    F_j1_change[j1] += (1 - r[l]) * r_n[l] * F[l - 1, j2, j1]
                    F_j2_change[j1] += (1 - r[l]) * r_n[l] * F[l - 1, j1, j2]
                    F_both_change += r_n[l] ** 2 * F[l - 1, j1, j2]

            F[l, :, :] = F_both_change

            for j1 in range(n):
                F[l, j1, :] += F_j2_change
                F[l, :, j1] += F_j1_change
                for j2 in range(n):
                    F[l, j1, j2] += F_no_change[j1, j2]

            for j1 in range(n):
                for j2 in range(n):
                    emission_prob = core.get_emission_probability_diploid(
                        ref_genotype=G[l, j1, j2],
                        query_genotype=s[0, l],
                        site=l,
                        emission_matrix=e,
                    )
                    F[l, j1, j2] *= emission_prob

            ll = np.log10(np.sum(F[l, :, :]))

    return F, c, ll


@jit.numba_njit
def backward_ls_dip_loop(n, m, G, s, e, c, r):
    """
    An implementation without vectorisation.

    This is exposed via the API.
    """
    # Initialise
    B = np.zeros((m, n, n))
    B[m - 1, :, :] = 1
    r_n = r / n

    for l in range(m - 2, -1, -1):
        B_no_change = np.zeros((n, n))
        B_j1_change = np.zeros(n)
        B_j2_change = np.zeros(n)
        B_both_change = 0

        # Evaluate the emission matrix at this site, for all pairs
        e_tmp = np.zeros((n, n))
        for j1 in range(n):
            for j2 in range(n):
                emission_prob = core.get_emission_probability_diploid(
                    ref_genotype=G[l + 1, j1, j2],
                    query_genotype=s[0, l + 1],
                    site=l + 1,
                    emission_matrix=e,
                )
                e_tmp[j1, j2] = emission_prob

        for j1 in range(n):
            for j2 in range(n):
                B_no_change[j1, j2] = (
                    (1 - r[l + 1]) ** 2 * B[l + 1, j1, j2] * e_tmp[j1, j2]
                )
                B_j2_change[j1] += (
                    (1 - r[l + 1]) * r_n[l + 1] * B[l + 1, j1, j2] * e_tmp[j1, j2]
                )
                B_j1_change[j1] += (
                    (1 - r[l + 1]) * r_n[l + 1] * B[l + 1, j2, j1] * e_tmp[j2, j1]
                )
                B_both_change += r_n[l + 1] ** 2 * e_tmp[j1, j2] * B[l + 1, j1, j2]

        B[l, :, :] = B_both_change

        for j1 in range(n):
            B[l, j1, :] += B_j2_change
            B[l, :, j1] += B_j1_change
            for j2 in range(n):
                B[l, j1, j2] += B_no_change[j1, j2]

        B[l, :, :] *= 1 / c[l + 1]

    return B
