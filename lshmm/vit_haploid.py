"""Implementations of the Li & Stephens Viterbi algorithm on haploid genotype data."""

import numpy as np

from . import core, jit


@jit.numba_njit
def viterbi_naive_init(
    n,
    m,
    H,
    s,
    e,
    r,
    emission_func,
):
    """Initialise a naive implementation."""
    V = np.zeros((m, n))
    P = np.zeros((m, n), dtype=np.int64)
    num_copiable_entries = core.get_num_copiable_entries(H)
    r_n = r / num_copiable_entries

    for i in range(n):
        emission_prob = emission_func(
            ref_allele=H[0, i],
            query_allele=s[0, 0],
            site=0,
            emission_matrix=e,
        )
        V[0, i] = 1 / n * emission_prob

    return V, P, r_n


@jit.numba_njit
def viterbi_init(
    n,
    m,
    H,
    s,
    e,
    r,
    emission_func,
):
    """Initialise a naive, but more memory efficient, implementation."""
    V_prev = np.zeros(n)
    V = np.zeros(n)
    P = np.zeros((m, n), dtype=np.int64)
    num_copiable_entries = core.get_num_copiable_entries(H)
    r_n = r / num_copiable_entries

    for i in range(n):
        emission_prob = emission_func(
            ref_allele=H[0, i],
            query_allele=s[0, 0],
            site=0,
            emission_matrix=e,
        )
        V_prev[i] = 1 / n * emission_prob

    return V, V_prev, P, r_n


@jit.numba_njit
def forwards_viterbi_hap_naive(
    n,
    m,
    H,
    s,
    e,
    r,
    emission_func,
):
    """A naive implementation of the forward pass."""
    V, P, r_n = viterbi_naive_init(n, m, H, s, e, r, emission_func=emission_func)

    for j in range(1, m):
        for i in range(n):
            v = np.zeros(n)
            for k in range(n):
                emission_prob = emission_func(
                    ref_allele=H[j, i],
                    query_allele=s[0, j],
                    site=j,
                    emission_matrix=e,
                )
                v[k] = V[j - 1, k] * emission_prob
                if k == i:
                    v[k] *= 1 - r[j] + r_n[j]
                else:
                    v[k] *= r_n[j]
            P[j, i] = np.argmax(v)
            V[j, i] = v[P[j, i]]

    ll = np.log10(np.amax(V[m - 1, :]))

    return V, P, ll


@jit.numba_njit
def forwards_viterbi_hap_naive_vec(
    n,
    m,
    H,
    s,
    e,
    r,
    emission_func,
):
    """A naive matrix-based implementation of the forward pass."""
    V, P, r_n = viterbi_naive_init(n, m, H, s, e, r, emission_func=emission_func)

    for j in range(1, m):
        v_tmp = V[j - 1, :] * r_n[j]
        for i in range(n):
            v = np.copy(v_tmp)
            v[i] += V[j - 1, i] * (1 - r[j])
            emission_prob = emission_func(
                ref_allele=H[j, i],
                query_allele=s[0, j],
                site=j,
                emission_matrix=e,
            )
            v *= emission_prob
            P[j, i] = np.argmax(v)
            V[j, i] = v[P[j, i]]

    ll = np.log10(np.amax(V[m - 1, :]))

    return V, P, ll


@jit.numba_njit
def forwards_viterbi_hap_naive_low_mem(
    n,
    m,
    H,
    s,
    e,
    r,
    emission_func,
):
    """A naive implementation of the forward pass with reduced memory."""
    V, V_prev, P, r_n = viterbi_init(n, m, H, s, e, r, emission_func=emission_func)

    for j in range(1, m):
        for i in range(n):
            v = np.zeros(n)
            for k in range(n):
                emission_prob = emission_func(
                    ref_allele=H[j, i],
                    query_allele=s[0, j],
                    site=j,
                    emission_matrix=e,
                )
                v[k] = V_prev[k] * emission_prob
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
def forwards_viterbi_hap_naive_low_mem_rescaling(
    n,
    m,
    H,
    s,
    e,
    r,
    emission_func,
):
    """A naive implementation of the forward pass with reduced memory and rescaling."""
    V, V_prev, P, r_n = viterbi_init(n, m, H, s, e, r, emission_func=emission_func)
    c = np.ones(m)

    for j in range(1, m):
        c[j] = np.amax(V_prev)
        V_prev *= 1 / c[j]
        for i in range(n):
            v = np.zeros(n)
            for k in range(n):
                emission_prob = emission_func(
                    ref_allele=H[j, i],
                    query_allele=s[0, j],
                    site=j,
                    emission_matrix=e,
                )
                v[k] = V_prev[k] * emission_prob
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
def forwards_viterbi_hap_low_mem_rescaling(
    n,
    m,
    H,
    s,
    e,
    r,
    emission_func,
):
    """An implementation with reduced memory that exploits the Markov structure."""
    V, V_prev, P, r_n = viterbi_init(n, m, H, s, e, r, emission_func=emission_func)
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
            emission_prob = emission_func(
                ref_allele=H[j, i],
                query_allele=s[0, j],
                site=j,
                emission_matrix=e,
            )
            V[i] *= emission_prob
        V_prev = np.copy(V)

    ll = np.sum(np.log10(c)) + np.log10(np.max(V))

    return V, P, ll


@jit.numba_njit
def forwards_viterbi_hap_lower_mem_rescaling(
    n,
    m,
    H,
    s,
    e,
    r,
    emission_func,
):
    """
    An implementation with even smaller memory footprint
    that exploits the Markov structure.

    This is exposed via the API.
    """
    V = np.zeros(n)
    for i in range(n):
        emission_prob = emission_func(
            ref_allele=H[0, i],
            query_allele=s[0, 0],
            site=0,
            emission_matrix=e,
        )
        V[i] = 1 / n * emission_prob
    P = np.zeros((m, n), dtype=np.int64)
    num_copiable_entries = core.get_num_copiable_entries(H)
    r_n = r / num_copiable_entries
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
            emission_prob = emission_func(
                ref_allele=H[j, i],
                query_allele=s[0, j],
                site=j,
                emission_matrix=e,
            )
            V[i] *= emission_prob

    ll = np.sum(np.log10(c)) + np.log10(np.max(V))

    return V, P, ll


@jit.numba_njit
def forwards_viterbi_hap_lower_mem_rescaling_no_pointer(
    n,
    m,
    H,
    s,
    e,
    r,
    emission_func,
):
    """
    An implementation with even smaller memory footprint and rescaling
    that exploits the Markov structure.
    """
    V = np.zeros(n)
    for i in range(n):
        emission_prob = emission_func(
            ref_allele=H[0, i],
            query_allele=s[0, 0],
            site=0,
            emission_matrix=e,
        )
        V[i] = 1 / n * emission_prob
    num_copiable_entries = core.get_num_copiable_entries(H)
    r_n = r / num_copiable_entries
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
            emission_prob = emission_func(
                ref_allele=H[j, i],
                query_allele=s[0, j],
                site=j,
                emission_matrix=e,
            )
            V[i] *= emission_prob

    V_argmaxes[m - 1] = np.argmax(V)
    ll = np.sum(np.log10(c)) + np.log10(np.max(V))

    return V, V_argmaxes, recombs, ll


@jit.numba_njit
def backwards_viterbi_hap(m, V_last, P):
    """
    An implementation of the backwards pass to get the most likely path.

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
    """An implementation of the backwards pass to get the most likely path."""
    path = np.zeros(m, dtype=np.int64)
    path[m - 1] = V_argmaxes[m - 1]

    for j in range(m - 2, -1, -1):
        current_best_template = path[j + 1]
        if current_best_template in recombs[j + 1]:
            current_best_template = V_argmaxes[j]
        path[j] = current_best_template

    return path


@jit.numba_njit
def path_ll_hap(
    n,
    m,
    H,
    path,
    s,
    e,
    r,
    emission_func,
):
    """
    Evaluate the log-likelihood of a path through a reference panel resulting in a query.

    This is exposed via the API.
    """
    emission_prob = emission_func(
        ref_allele=H[0, path[0]],
        query_allele=s[0, 0],
        site=0,
        emission_matrix=e,
    )
    log_prob_path = np.log10((1 / n) * emission_prob)
    old = path[0]
    num_copiable_entries = core.get_num_copiable_entries(H)
    r_n = r / num_copiable_entries

    for l in range(1, m):
        emission_prob = emission_func(
            ref_allele=H[l, path[l]],
            query_allele=s[0, l],
            site=l,
            emission_matrix=e,
        )
        current = path[l]
        same = old == current

        if same:
            log_prob_path += np.log10((1 - r[l]) + r_n[l])
        else:
            log_prob_path += np.log10(r_n[l])

        log_prob_path += np.log10(emission_prob)
        old = current

    return log_prob_path
