"""
Various implementations of the Li & Stephens Viterbi algorithm on diploid genotype data,
where the data is structured as variants x samples x samples.
"""

import numpy as np

from . import core
from . import jit


@jit.numba_njit
def forwards_viterbi_dip_naive(n, m, G, s, e, r):
    """A naive implementation."""
    # Initialise
    V = np.zeros((m, n, n))
    P = np.zeros((m, n, n), dtype=np.int64)
    c = np.ones(m)
    num_copiable_entries = core.get_num_copiable_entries(G)
    r_n = r / num_copiable_entries

    for j1 in range(n):
        for j2 in range(n):
            emission_prob = core.get_emission_probability_diploid(
                ref_genotype=G[0, j1, j2],
                query_genotype=s[0, 0],
                site=0,
                emission_matrix=e,
            )
            V[0, j1, j2] = 1 / (n**2) * emission_prob

    for l in range(1, m):
        emission_probs = core.get_emission_probability_diploid_genotypes(
            ref_genotypes=G[l, :, :],
            query_genotype=s[0, l],
            site=l,
            emission_matrix=e,
        )

        for j1 in range(n):
            for j2 in range(n):
                v = np.zeros((n, n))
                for k1 in range(n):
                    for k2 in range(n):
                        v[k1, k2] = V[l - 1, k1, k2]
                        if (k1 == j1) and (k2 == j2):
                            v[k1, k2] *= (
                                (1 - r[l]) ** 2 + 2 * (1 - r[l]) * r_n[l] + r_n[l] ** 2
                            )
                        elif (k1 == j1) or (k2 == j2):
                            v[k1, k2] *= r_n[l] * (1 - r[l]) + r_n[l] ** 2
                        else:
                            v[k1, k2] *= r_n[l] ** 2
                V[l, j1, j2] = np.amax(v) * emission_probs[j1, j2]
                P[l, j1, j2] = np.argmax(v)
        c[l] = np.amax(V[l, :, :])
        V[l, :, :] *= 1 / c[l]

    ll = np.sum(np.log10(c))

    return V, P, ll


@jit.numba_njit
def forwards_viterbi_dip_naive_low_mem(n, m, G, s, e, r):
    """A naive implementation with reduced memory."""
    # Initialise
    V = np.zeros((n, n))
    V_prev = np.zeros((n, n))
    P = np.zeros((m, n, n), dtype=np.int64)
    c = np.ones(m)
    num_copiable_entries = core.get_num_copiable_entries(G)
    r_n = r / num_copiable_entries

    for j1 in range(n):
        for j2 in range(n):
            emission_prob = core.get_emission_probability_diploid(
                ref_genotype=G[0, j1, j2],
                query_genotype=s[0, 0],
                site=0,
                emission_matrix=e,
            )
            V_prev[j1, j2] = 1 / (n**2) * emission_prob

    # Take a look at the haploid Viterbi implementation in Jerome's code, and
    # see if we can pinch some ideas.
    # Diploid Viterbi, with smaller memory footprint.
    for l in range(1, m):
        emission_probs = core.get_emission_probability_diploid_genotypes(
            ref_genotypes=G[l, :, :],
            query_genotype=s[0, l],
            site=l,
            emission_matrix=e,
        )

        for j1 in range(n):
            for j2 in range(n):
                v = np.zeros((n, n))
                for k1 in range(n):
                    for k2 in range(n):
                        v[k1, k2] = V_prev[k1, k2]
                        if (k1 == j1) and (k2 == j2):
                            v[k1, k2] *= (
                                (1 - r[l]) ** 2 + 2 * (1 - r[l]) * r_n[l] + r_n[l] ** 2
                            )
                        elif (k1 == j1) or (k2 == j2):
                            v[k1, k2] *= r_n[l] * (1 - r[l]) + r_n[l] ** 2
                        else:
                            v[k1, k2] *= r_n[l] ** 2
                V[j1, j2] = np.amax(v) * emission_probs[j1, j2]
                P[l, j1, j2] = np.argmax(v)
        c[l] = np.amax(V)
        V_prev = np.copy(V) / c[l]

    ll = np.sum(np.log10(c))

    return V, P, ll


@jit.numba_njit
def forwards_viterbi_dip_low_mem(n, m, G, s, e, r):
    """
    An implementation with reduced memory.

    This is exposed via the API.
    """
    # Initialise
    V = np.zeros((n, n))
    V_prev = np.zeros((n, n))
    P = np.zeros((m, n, n), dtype=np.int64)
    c = np.ones(m)
    num_copiable_entries = core.get_num_copiable_entries(G)
    r_n = r / num_copiable_entries

    for j1 in range(n):
        for j2 in range(n):
            emission_prob = core.get_emission_probability_diploid(
                ref_genotype=G[0, j1, j2],
                query_genotype=s[0, 0],
                site=0,
                emission_matrix=e,
            )
            V_prev[j1, j2] = 1 / (n**2) * emission_prob

    # Diploid Viterbi, with smaller memory footprint, rescaling, and using the structure of the HMM.
    for l in range(1, m):
        emission_probs = core.get_emission_probability_diploid_genotypes(
            ref_genotypes=G[l, :, :],
            query_genotype=s[0, l],
            site=l,
            emission_matrix=e,
        )

        c[l] = np.amax(V_prev)
        argmax = np.argmax(V_prev)

        V_prev *= 1 / c[l]
        V_rowcol_max = core.np_amax(V_prev, 0)
        arg_rowcol_max = core.np_argmax(V_prev, 0)

        no_switch = (1 - r[l]) ** 2 + 2 * (r_n[l] * (1 - r[l])) + r_n[l] ** 2
        single_switch = r_n[l] * (1 - r[l]) + r_n[l] ** 2
        double_switch = r_n[l] ** 2

        j1_j2 = 0

        for j1 in range(n):
            for j2 in range(n):
                V_single_switch = max(V_rowcol_max[j1], V_rowcol_max[j2])
                P_single_switch = np.argmax(
                    np.array([V_rowcol_max[j1], V_rowcol_max[j2]])
                )

                if P_single_switch == 0:
                    template_single_switch = j1 * n + arg_rowcol_max[j1]
                else:
                    template_single_switch = arg_rowcol_max[j2] * n + j2

                V[j1, j2] = V_prev[j1, j2] * no_switch  # No switch in either
                P[l, j1, j2] = j1_j2

                # Single or double switch?
                single_switch_tmp = single_switch * V_single_switch
                if single_switch_tmp > double_switch:
                    # Then single switch is the alternative
                    if V[j1, j2] < single_switch * V_single_switch:
                        V[j1, j2] = single_switch * V_single_switch
                        P[l, j1, j2] = template_single_switch
                else:
                    # Double switch is the alternative
                    if V[j1, j2] < double_switch:
                        V[j1, j2] = double_switch
                        P[l, j1, j2] = argmax

                V[j1, j2] *= emission_probs[j1, j2]
                j1_j2 += 1
        V_prev = np.copy(V)

    ll = np.sum(np.log10(c)) + np.log10(np.amax(V))

    return V, P, ll


@jit.numba_njit
def forwards_viterbi_dip_low_mem_no_pointer(n, m, G, s, e, r):
    """An implementation with reduced memory and no pointer."""
    # Initialise
    V = np.zeros((n, n))
    V_prev = np.zeros((n, n))
    c = np.ones(m)
    num_copiable_entries = core.get_num_copiable_entries(G)
    r_n = r / num_copiable_entries

    recombs_single = [
        np.zeros(shape=0, dtype=np.int64) for _ in range(m)
    ]  # Store all single switch recombs
    recombs_double = [
        np.zeros(shape=0, dtype=np.int64) for _ in range(m)
    ]  # Store all double switch recombs

    V_argmaxes = np.zeros(m)
    V_rowcol_maxes = np.zeros((m, n))
    V_rowcol_argmaxes = np.zeros((m, n))

    for j1 in range(n):
        for j2 in range(n):
            emission_prob = core.get_emission_probability_diploid(
                ref_genotype=G[0, j1, j2],
                query_genotype=s[0, 0],
                site=0,
                emission_matrix=e,
            )
            V_prev[j1, j2] = 1 / (n**2) * emission_prob

    # Diploid Viterbi, with smaller memory footprint, rescaling, and using the structure of the HMM.
    for l in range(1, m):
        emission_probs = core.get_emission_probability_diploid_genotypes(
            ref_genotypes=G[l, :, :],
            query_genotype=s[0, l],
            site=l,
            emission_matrix=e,
        )

        c[l] = np.amax(V_prev)
        argmax = np.argmax(V_prev)
        V_argmaxes[l - 1] = argmax  # added

        V_prev *= 1 / c[l]
        V_rowcol_max = core.np_amax(V_prev, 0)
        V_rowcol_maxes[l - 1, :] = V_rowcol_max
        arg_rowcol_max = core.np_argmax(V_prev, 0)
        V_rowcol_argmaxes[l - 1, :] = arg_rowcol_max

        no_switch = (1 - r[l]) ** 2 + 2 * (r_n[l] * (1 - r[l])) + r_n[l] ** 2
        single_switch = r_n[l] * (1 - r[l]) + r_n[l] ** 2
        double_switch = r_n[l] ** 2

        j1_j2 = 0

        for j1 in range(n):
            for j2 in range(n):
                V_single_switch = max(V_rowcol_max[j1], V_rowcol_max[j2])
                V[j1, j2] = V_prev[j1, j2] * no_switch  # No switch in either

                # Single or double switch?
                single_switch_tmp = single_switch * V_single_switch
                if single_switch_tmp > double_switch:
                    # Then single switch is the alternative
                    if V[j1, j2] < single_switch * V_single_switch:
                        V[j1, j2] = single_switch * V_single_switch
                        recombs_single[l] = np.append(recombs_single[l], j1_j2)
                else:
                    # Double switch is the alternative
                    if V[j1, j2] < double_switch:
                        V[j1, j2] = double_switch
                        recombs_double[l] = np.append(recombs_double[l], values=j1_j2)

                V[j1, j2] *= emission_probs[j1, j2]
                j1_j2 += 1
        V_prev = np.copy(V)

    V_argmaxes[m - 1] = np.argmax(V_prev)
    V_rowcol_maxes[m - 1, :] = core.np_amax(V_prev, 0)
    V_rowcol_argmaxes[m - 1, :] = core.np_argmax(V_prev, 0)
    ll = np.sum(np.log10(c)) + np.log10(np.amax(V))

    return (
        V,
        V_argmaxes,
        V_rowcol_maxes,
        V_rowcol_argmaxes,
        recombs_single,
        recombs_double,
        ll,
    )


@jit.numba_njit
def forwards_viterbi_dip_naive_vec(n, m, G, s, e, r):
    """An implementation using Numpy vectorisation."""
    # Initialise
    V = np.zeros((m, n, n))
    P = np.zeros((m, n, n), dtype=np.int64)
    c = np.ones(m)
    num_copiable_entries = core.get_num_copiable_entries(G)
    r_n = r / num_copiable_entries

    for j1 in range(n):
        for j2 in range(n):
            emission_prob = core.get_emission_probability_diploid(
                ref_genotype=G[0, j1, j2],
                query_genotype=s[0, 0],
                site=0,
                emission_matrix=e,
            )
            V[0, j1, j2] = 1 / (n**2) * emission_prob

    # Jumped the gun - vectorising.
    for l in range(1, m):
        emission_probs = core.get_emission_probability_diploid_genotypes(
            ref_genotypes=G[l, :, :],
            query_genotype=s[0, l],
            site=l,
            emission_matrix=e,
        )

        for j1 in range(n):
            for j2 in range(n):
                v = (r_n[l] ** 2) * np.ones((n, n))
                v[j1, j2] += (1 - r[l]) ** 2
                v[j1, :] += r_n[l] * (1 - r[l])
                v[:, j2] += r_n[l] * (1 - r[l])
                v *= V[l - 1, :, :]
                V[l, j1, j2] = np.amax(v) * emission_probs[j1, j2]
                P[l, j1, j2] = np.argmax(v)

        c[l] = np.amax(V[l, :, :])
        V[l, :, :] *= 1 / c[l]

    ll = np.sum(np.log10(c))

    return V, P, ll


def forwards_viterbi_dip_naive_full_vec(n, m, G, s, e, r):
    """Fully vectorised naive implementation using Numpy."""
    char_both = np.eye(n * n).ravel().reshape((n, n, n, n))
    char_col = np.tile(np.sum(np.eye(n * n).reshape((n, n, n, n)), 3), (n, 1, 1, 1))
    char_row = np.copy(char_col).T
    rows, cols = np.ogrid[:n, :n]

    # Initialise
    V = np.zeros((m, n, n))
    P = np.zeros((m, n, n), dtype=np.int64)
    c = np.ones(m)

    emission_probs = core.get_emission_probability_diploid_genotypes(
        ref_genotypes=G[0, :, :],
        query_genotype=s[0, 0],
        site=l,
        emission_matrix=e,
    )
    V[0, :, :] = 1 / (n**2) * emission_probs
    num_copiable_entries = core.get_num_copiable_entries(G)
    r_n = r / num_copiable_entries

    for l in range(1, m):
        emission_probs = core.get_emission_probability_diploid_genotypes(
            ref_genotypes=G[l, :, :],
            query_genotype=s[0, l],
            site=l,
            emission_matrix=e,
        )
        v = (
            (r_n[l] ** 2)
            + (1 - r[l]) ** 2 * char_both
            + (r_n[l] * (1 - r[l])) * (char_col + char_row)
        )
        v *= V[l - 1, :, :]
        P[l, :, :] = np.argmax(v.reshape(n, n, -1), 2)  # Have to flatten to use argmax
        V[l, :, :] = v.reshape(n, n, -1)[rows, cols, P[l, :, :]] * emission_probs
        c[l] = np.amax(V[l, :, :])
        V[l, :, :] *= 1 / c[l]

    ll = np.sum(np.log10(c))

    return V, P, ll


@jit.numba_njit
def backwards_viterbi_dip(m, V_last, P):
    """
    Run a backwards pass to determine the most likely path.

    This is exposed via the API.
    """
    assert V_last.ndim == 2
    assert V_last.shape[0] == V_last.shape[1]

    # Initialise
    path = np.zeros(m, dtype=np.int64)
    path[m - 1] = np.argmax(V_last)

    # Backtrace
    for j in range(m - 2, -1, -1):
        path[j] = P[j + 1, :, :].ravel()[path[j + 1]]

    return path


@jit.numba_njit
def in_list(array, value):
    where = np.searchsorted(array, value)
    if where < array.shape[0]:
        return array[where] == value
    return False


@jit.numba_njit
def backwards_viterbi_dip_no_pointer(
    m,
    V_argmaxes,
    V_rowcol_maxes,
    V_rowcol_argmaxes,
    recombs_single,
    recombs_double,
    V_last,
):
    """Run a backwards pass to determine the most likely path."""
    assert V_last.ndim == 2
    assert V_last.shape[0] == V_last.shape[1]

    # Initialise
    path = np.zeros(m, dtype=np.int64)
    path[m - 1] = np.argmax(V_last)
    n = V_last.shape[0]

    # Backtrace
    for l in range(m - 2, -1, -1):
        current_best_template = path[l + 1]
        # Current_best_template in recombs_double[l + 1]
        if in_list(recombs_double[l + 1], current_best_template):
            current_best_template = V_argmaxes[l]
        # Current_best_template in recombs_single[l + 1]
        elif in_list(recombs_single[l + 1], current_best_template):
            (j1, j2) = divmod(current_best_template, n)
            if V_rowcol_maxes[l, j1] > V_rowcol_maxes[l, j2]:
                current_best_template = j1 * n + V_rowcol_argmaxes[l, j1]
            else:
                current_best_template = V_rowcol_argmaxes[l, j2] * n + j2
        path[l] = current_best_template

    return path


def get_phased_path(n, path):
    """This is exposed via the API."""
    return np.unravel_index(path, (n, n))


@jit.numba_njit
def path_ll_dip(n, m, G, phased_path, s, e, r):
    """
    Evaluate log-likelihood path through a reference panel which results in sequence.

    This is exposed via the API.
    """
    emission_prob = core.get_emission_probability_diploid(
        ref_genotype=G[0, phased_path[0][0], phased_path[1][0]],
        query_genotype=s[0, 0],
        site=0,
        emission_matrix=e,
    )
    log_prob_path = np.log10(1 / (n**2) * emission_prob)

    old_phase = np.array([phased_path[0][0], phased_path[1][0]])
    num_copiable_entries = core.get_num_copiable_entries(G)
    r_n = r / num_copiable_entries

    for l in range(1, m):
        emission_prob = core.get_emission_probability_diploid(
            ref_genotype=G[l, phased_path[0][l], phased_path[1][l]],
            query_genotype=s[0, l],
            site=l,
            emission_matrix=e,
        )

        current_phase = np.array([phased_path[0][l], phased_path[1][l]])
        phase_diff = np.sum(~np.equal(current_phase, old_phase))

        if phase_diff == 0:
            log_prob_path += np.log10(
                (1 - r[l]) ** 2 + 2 * (r_n[l] * (1 - r[l])) + r_n[l] ** 2
            )
        elif phase_diff == 1:
            log_prob_path += np.log10(r_n[l] * (1 - r[l]) + r_n[l] ** 2)
        else:
            log_prob_path += np.log10(r_n[l] ** 2)

        log_prob_path += np.log10(emission_prob)
        old_phase = current_phase

    return log_prob_path
