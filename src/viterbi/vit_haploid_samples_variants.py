import numba as nb
import numpy as np


@nb.jit
def viterbi_naive_init(n, m, H, s, e, r):
    """
    Initialisation portion of initial naive implementation of LS viterbi to avoid
    lots of code duplication
    """

    V = np.zeros((n, m))
    P = np.zeros((n, m)).astype(np.int64)
    V[:, 0] = 1 / n * e[np.equal(H[:, 0], s[0, 0]).astype(np.int64), 0]
    P[:, 0] = 0  # Reminder
    r_n = r / n

    return V, P, r_n


@nb.jit
def viterbi_init(n, m, H, s, e, r):
    """
    Initialisation portion of initial naive, but more space memory efficient implementation
    of LS viterbi to avoid lots of code duplication
    """

    V_previous = 1 / n * e[np.equal(H[:, 0], s[0, 0]).astype(np.int64), 0]
    V = np.zeros(n)
    P = np.zeros((n, m)).astype(np.int64)
    P[:, 0] = 0  # Reminder
    r_n = r / n

    return V, V_previous, P, r_n


@nb.jit
def forwards_viterbi_hap_naive(n, m, H, s, e, r):
    """
    Simple naive LS forward Viterbi algorithm.
    """

    # Initialise
    V, P, r_n = viterbi_naive_init(n, m, H, s, e, r)

    for j in range(1, m):
        for i in range(n):
            # Get the vector to maximise over
            v = np.zeros(n)
            for k in range(n):
                v[k] = e[np.int64(np.equal(H[i, j], s[0, j])), j] * V[k, j - 1]
                if k == i:
                    v[k] *= 1 - r[j] + r_n[j]
                else:
                    v[k] *= r_n[j]
            P[i, j] = np.argmax(v)
            V[i, j] = v[P[i, j]]

    ll = np.log10(np.amax(V[:, m - 1]))

    return V, P, ll


@nb.jit
def forwards_viterbi_hap_naive_vec(n, m, H, s, e, r):
    """
    Simple matrix based method naive LS forward Viterbi algorithm. Vectorised things
     - I jumped the gun!
    """

    # Initialise
    V, P, r_n = viterbi_naive_init(n, m, H, s, e, r)

    for j in range(1, m):
        v_tmp = V[:, j - 1] * r_n[j]
        for i in range(n):
            v = np.copy(v_tmp)
            v[i] += V[i, j - 1] * (1 - r[j])
            v *= e[np.int64(np.equal(H[i, j], s[0, j])), j]
            P[i, j] = np.argmax(v)
            V[i, j] = v[P[i, j]]

    ll = np.log10(np.amax(V[:, m - 1]))

    return V, P, ll


def forwards_viterbi_hap_naive_full_vec(n, m, H, s, e, r):
    """
    Simple matrix based method naive LS forward Viterbi algorithm. Vectorised things
     even more - I jumped the gun!
    """

    # Initialise
    V, P, r_n = viterbi_naive_init(n, m, H, s, e, r)

    for j in range(1, m):
        v = np.tile(V[:, j - 1] * r_n[j], (n, 1)) + np.diag(V[:, j - 1] * (1 - r[j]))
        P[:, j] = np.argmax(v, 1)
        V[:, j] = (
            v[range(n), P[:, j]] * e[np.equal(H[:, j], s[0, j]).astype(np.int64), j]
        )

    ll = np.log10(np.amax(V[:, m - 1]))

    return V, P, ll


@nb.jit
def forwards_viterbi_hap_naive_low_mem(n, m, H, s, e, r):
    """
    Simple naive LS forward Viterbi algorithm. More memory efficient.
    """

    # Initialise
    V, V_previous, P, r_n = viterbi_init(n, m, H, s, e, r)

    for j in range(1, m):
        for i in range(n):
            # Get the vector to maximise over
            v = np.zeros(n)
            for k in range(n):
                v[k] = e[np.int64(np.equal(H[i, j], s[0, j])), j] * V_previous[k]
                if k == i:
                    v[k] *= (1 - r[j]) + r_n[j]
                else:
                    v[k] *= r_n[j]
            P[i, j] = np.argmax(v)
            V[i] = v[P[i, j]]
        V_previous = np.copy(V)

    ll = np.log10(np.amax(V))

    return V, P, ll


@nb.jit
def forwards_viterbi_hap_naive_low_mem_rescaling(n, m, H, s, e, r):
    """
    Simple naive LS forward Viterbi algorithm. More memory efficient, and with
    a rescaling to avoid underflow problems
    """

    # Initialise
    V, V_previous, P, r_n = viterbi_init(n, m, H, s, e, r)
    c = np.ones(m)

    for j in range(1, m):
        c[j] = np.amax(V_previous)
        V_previous *= 1 / c[j]
        for i in range(n):
            # Get the vector to maximise over
            v = np.zeros(n)
            for k in range(n):
                v[k] = e[np.int64(np.equal(H[i, j], s[0, j])), j] * V_previous[k]
                if k == i:
                    v[k] *= (1 - r[j]) + r_n[j]
                else:
                    v[k] *= r_n[j]
            P[i, j] = np.argmax(v)
            V[i] = v[P[i, j]]

        V_previous = np.copy(V)

    ll = np.sum(np.log10(c)) + np.log10(np.amax(V))

    return V, P, ll


@nb.jit
def forwards_viterbi_hap_low_mem_rescaling(n, m, H, s, e, r):
    """
    Simple LS forward Viterbi algorithm. Smaller memory footprint and rescaling,
    and considers the structure of the Markov process.
    """

    # Initialise
    V, V_previous, P, r_n = viterbi_init(n, m, H, s, e, r)
    c = np.ones(m)

    for j in range(1, m):
        argmax = np.argmax(V_previous)
        c[j] = V_previous[argmax]
        V_previous *= 1 / c[j]
        V = np.zeros(n)
        for i in range(n):
            V[i] = V_previous[i] * (1 - r[j] + r_n[j])
            P[i, j] = i
            if V[i] < r_n[j]:
                V[i] = r_n[j]
                P[i, j] = argmax
            V[i] *= e[np.equal(H[i, j], s[0, j]).astype(np.int64), j]
        V_previous = np.copy(V)

    ll = np.sum(np.log10(c)) + np.log10(np.max(V))

    return V, P, ll


@nb.jit
def forwards_viterbi_hap_lower_mem_rescaling(n, m, H, s, e, r):
    """
    Simple LS forward Viterbi algorithm. Even smaller memory footprint and rescaling,
    and considers the structure of the Markov process.
    """

    # Initialise
    V = 1 / n * e[np.equal(H[:, 0], s[0, 0]).astype(np.int64), 0]
    P = np.zeros((n, m)).astype(np.int64)
    P[:, 0] = 0
    r_n = r / n
    c = np.ones(m)

    for j in range(1, m):
        argmax = np.argmax(V)
        c[j] = V[argmax]
        V *= 1 / c[j]
        for i in range(n):
            V[i] = V[i] * (1 - r[j] + r_n[j])
            P[i, j] = i
            if V[i] < r_n[j]:
                V[i] = r_n[j]
                P[i, j] = argmax
            V[i] *= e[np.int64(np.equal(H[i, j], s[0, j])), j]

    ll = np.sum(np.log10(c)) + np.log10(np.max(V))

    return V, P, ll


@nb.jit
def backwards_viterbi_hap(m, V_last, P):
    """
    Backwards pass to determine the most likely path
    """

    # Initialise
    path = np.zeros(m).astype(np.int64)
    path[m - 1] = np.argmax(V_last)

    for j in range(m - 2, -1, -1):
        path[j] = P[path[j + 1], j + 1]

    return path
