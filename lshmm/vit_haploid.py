"""
Various implementations of the Li & Stephens Viterbi algorithm on haploid genotype data,
where the data is structured as variants x samples.
"""

import numpy as np

from . import core
from . import jit


@jit.numba_njit
def viterbi_naive_init(n, m, H, s, e, r):
    """Initialise a naive implementation."""
    V = np.zeros((m, n))
    P = np.zeros((m, n), dtype=np.int64)
    r_n = r / n

    for i in range(n):
        emission_index = core.get_index_in_emission_matrix_haploid(
            ref_allele=H[0, i], query_allele=s[0, 0]
        )
        V[0, i] = 1 / n * e[0, emission_index]

    return V, P, r_n


@jit.numba_njit
def viterbi_init(n, m, H, s, e, r):
    """Initialise a naive, but more memory efficient, implementation."""
    V_prev = np.zeros(n)
    V = np.zeros(n)
    P = np.zeros((m, n), dtype=np.int64)
    r_n = r / n

    for i in range(n):
        emission_index = core.get_index_in_emission_matrix_haploid(
            ref_allele=H[0, i], query_allele=s[0, 0]
        )
        V_prev[i] = 1 / n * e[0, emission_index]

    return V, V_prev, P, r_n


@jit.numba_njit
def forwards_viterbi_hap_naive(n, m, H, s, e, r):
    """A naive implementation of the forward pass."""
    V, P, r_n = viterbi_naive_init(n, m, H, s, e, r)

    for j in range(1, m):
        for i in range(n):
            v = np.zeros(n)
            for k in range(n):
                emission_index = core.get_index_in_emission_matrix_haploid(
                    ref_allele=H[j, i], query_allele=s[0, j]
                )
                v[k] = V[j - 1, k] * e[j, emission_index]
                if k == i:
                    v[k] *= 1 - r[j] + r_n[j]
                else:
                    v[k] *= r_n[j]
            P[j, i] = np.argmax(v)
            V[j, i] = v[P[j, i]]

    ll = np.log10(np.amax(V[m - 1, :]))

    return V, P, ll


@jit.numba_njit
def forwards_viterbi_hap_naive_vec(n, m, H, s, e, r):
    """A naive matrix-based implementation of the forward pass using Numpy."""
    V, P, r_n = viterbi_naive_init(n, m, H, s, e, r)

    for j in range(1, m):
        v_tmp = V[j - 1, :] * r_n[j]
        for i in range(n):
            v = np.copy(v_tmp)
            v[i] += V[j - 1, i] * (1 - r[j])
            emission_index = core.get_index_in_emission_matrix_haploid(
                ref_allele=H[j, i], query_allele=s[0, j]
            )
            v *= e[j, emission_index]
            P[j, i] = np.argmax(v)
            V[j, i] = v[P[j, i]]

    ll = np.log10(np.amax(V[m - 1, :]))

    return V, P, ll


@jit.numba_njit
def forwards_viterbi_hap_naive_low_mem(n, m, H, s, e, r):
    """A naive implementation of the forward pass with reduced memory."""
    V, V_prev, P, r_n = viterbi_init(n, m, H, s, e, r)

    for j in range(1, m):
        for i in range(n):
            v = np.zeros(n)
            for k in range(n):
                emission_index = core.get_index_in_emission_matrix_haploid(
                    ref_allele=H[j, i], query_allele=s[0, j]
                )
                v[k] = V_prev[k] * e[j, emission_index]
                if k == i:
                    v[k] *= 1 - r[j] + r_n[j]
                else:
                    v[k] *= r_n[j]
            P[j, i] = np.argmax(v)
            V[i] = v[P[j, i]]
        V_prev = np.copy(V)

    ll = np.log10(np.amax(V))

    return V, P, ll


@jit.numba_njit
def forwards_viterbi_hap_naive_low_mem_rescaling(n, m, H, s, e, r):
    """A naive implementation of the forward pass with reduced memory and rescaling."""
    V, V_prev, P, r_n = viterbi_init(n, m, H, s, e, r)
    c = np.ones(m)

    for j in range(1, m):
        c[j] = np.amax(V_prev)
        V_prev *= 1 / c[j]
        for i in range(n):
            v = np.zeros(n)
            for k in range(n):
                emission_index = core.get_index_in_emission_matrix_haploid(
                    ref_allele=H[j, i], query_allele=s[0, j]
                )
                v[k] = V_prev[k] * e[j, emission_index]
                if k == i:
                    v[k] *= 1 - r[j] + r_n[j]
                else:
                    v[k] *= r_n[j]
            P[j, i] = np.argmax(v)
            V[i] = v[P[j, i]]
        V_prev = np.copy(V)

    ll = np.sum(np.log10(c)) + np.log10(np.amax(V))

    return V, P, ll


@jit.numba_njit
def forwards_viterbi_hap_low_mem_rescaling(n, m, H, s, e, r):
    """An implementation with reduced memory that exploits the Markov structure."""
    V, V_prev, P, r_n = viterbi_init(n, m, H, s, e, r)
    c = np.ones(m)

    for j in range(1, m):
        argmax = np.argmax(V_prev)
        c[j] = V_prev[argmax]
        V_prev *= 1 / c[j]
        V = np.zeros(n)
        for i in range(n):
            V[i] = V_prev[i] * (1 - r[j] + r_n[j])
            P[j, i] = i
            if V[i] < r_n[j]:
                V[i] = r_n[j]
                P[j, i] = argmax
            emission_index = core.get_index_in_emission_matrix_haploid(
                ref_allele=H[j, i], query_allele=s[0, j]
            )
            V[i] *= e[j, emission_index]
        V_prev = np.copy(V)

    ll = np.sum(np.log10(c)) + np.log10(np.max(V))

    return V, P, ll


@jit.numba_njit
def forwards_viterbi_hap_lower_mem_rescaling(n, m, H, s, e, r):
    """
    An implementation with even smaller memory footprint that exploits the Markov structure.

    This is exposed via the API.
    """
    V = np.zeros(n)
    for i in range(n):
        emission_index = core.get_index_in_emission_matrix_haploid(
            ref_allele=H[0, i], query_allele=s[0, 0]
        )
        V[i] = 1 / n * e[0, emission_index]
    P = np.zeros((m, n), dtype=np.int64)
    r_n = r / n
    c = np.ones(m)

    for j in range(1, m):
        argmax = np.argmax(V)
        c[j] = V[argmax]
        V *= 1 / c[j]
        for i in range(n):
            V[i] = V[i] * (1 - r[j] + r_n[j])
            P[j, i] = i
            if V[i] < r_n[j]:
                V[i] = r_n[j]
                P[j, i] = argmax
            emission_index = core.get_index_in_emission_matrix_haploid(
                ref_allele=H[j, i], query_allele=s[0, j]
            )
            V[i] *= e[j, emission_index]

    ll = np.sum(np.log10(c)) + np.log10(np.max(V))

    return V, P, ll


@jit.numba_njit
def forwards_viterbi_hap_lower_mem_rescaling_no_pointer(n, m, H, s, e, r):
    """
    An implementation with even smaller memory footprint and rescaling
    that exploits the Markov structure.
    """
    V = np.zeros(n)
    for i in range(n):
        emission_index = core.get_index_in_emission_matrix_haploid(
            ref_allele=H[0, i], query_allele=s[0, 0]
        )
        V[i] = 1 / n * e[0, emission_index]
    r_n = r / n
    c = np.ones(m)
    # This is going to be filled with the templates we can recombine to
    # that have higher prob than staying where we are.
    recombs = [np.zeros(shape=0, dtype=np.int64) for _ in range(m)]

    V_argmaxes = np.zeros(m)

    for j in range(1, m):
        argmax = np.argmax(V)
        V_argmaxes[j - 1] = argmax
        c[j] = V[argmax]
        V *= 1 / c[j]
        for i in range(n):
            V[i] = V[i] * (1 - r[j] + r_n[j])
            if V[i] < r_n[j]:
                V[i] = r_n[j]
                recombs[j] = np.append(
                    recombs[j], i
                )  # We add template i as a potential template to recombine to at site j.
            emission_index = core.get_index_in_emission_matrix_haploid(
                ref_allele=H[j, i], query_allele=s[0, j]
            )
            V[i] *= e[j, emission_index]

    V_argmaxes[m - 1] = np.argmax(V)
    ll = np.sum(np.log10(c)) + np.log10(np.max(V))

    return V, V_argmaxes, recombs, ll


# Speedier version, variants x samples
@jit.numba_njit
def backwards_viterbi_hap(m, V_last, P):
    """
    Run a backwards pass to determine the most likely path.

    This is exposed via API.
    """
    assert len(V_last.shape) == 1
    path = np.zeros(m, dtype=np.int64)
    path[m - 1] = np.argmax(V_last)

    for j in range(m - 2, -1, -1):
        path[j] = P[j + 1, path[j + 1]]

    return path


@jit.numba_njit
def backwards_viterbi_hap_no_pointer(m, V_argmaxes, recombs):
    """Run a backwards pass to determine the most likely path."""
    path = np.zeros(m, dtype=np.int64)
    path[m - 1] = V_argmaxes[m - 1]

    for j in range(m - 2, -1, -1):
        current_best_template = path[j + 1]
        if current_best_template in recombs[j + 1]:
            current_best_template = V_argmaxes[j]
        path[j] = current_best_template

    return path


@jit.numba_njit
def path_ll_hap(n, m, H, path, s, e, r):
    """
    Evaluate the log-likelihood of a path through a reference panel resulting in a sequence.

    This is exposed via the API.
    """
    emission_index = core.get_index_in_emission_matrix_haploid(
        ref_allele=H[0, path[0]], query_allele=s[0, 0]
    )
    log_prob_path = np.log10((1 / n) * e[0, emission_index])
    old = path[0]
    r_n = r / n

    for l in range(1, m):
        emission_index = core.get_index_in_emission_matrix_haploid(
            ref_allele=H[l, path[l]], query_allele=s[0, l]
        )
        current = path[l]
        same = old == current

        if same:
            log_prob_path += np.log10((1 - r[l]) + r_n[l])
        else:
            log_prob_path += np.log10(r_n[l])

        log_prob_path += np.log10(e[l, emission_index])
        old = current

    return log_prob_path
