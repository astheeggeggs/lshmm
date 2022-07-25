"""Collection of functions to run Viterbi algorithms on haploid genotype data, where the data is structured as variants x samples."""
import numba as nb
import numpy as np

# Speedier version, variants x samples

# def viterbi_naive_init(n, m, H, s, e, r):
#     '''
#     Initialisation portion of initial naive implementation of LS viterbi to avoid
#     lots of code duplication
#     '''

#     V = np.zeros((m,n))
#     P = np.zeros((m,n)).astype(np.int64)
#     V[0,:] = 1/n * e[0,np.equal(H[0,:], s[0,0]).astype(np.int64)]
#     P[0,:] = 0 # Reminder
#     r_n = r/n

#     return V, P, r_n


@nb.jit
def viterbi_naive_init(n, m, H, s, e, r):
    """Initialise naive implementation of LS viterbi."""
    V = np.zeros((m, n))
    P = np.zeros((m, n)).astype(np.int64)
    r_n = r / n
    for i in range(n):
        V[0, i] = 1 / n * e[0, np.int64(np.equal(H[0, i], s[0, 0]))]

    return V, P, r_n


# def viterbi_init(n, m, H, s, e, r):
#     '''
#     Initialisation portion of initial naive, but more space memory efficient implementation
#     of LS viterbi to avoid lots of code duplication
#     '''

#     V_previous = 1/n * e[0,np.equal(H[0,:], s[0,0]).astype(np.int64)]
#     V = np.zeros(n)
#     P = np.zeros((m,n)).astype(np.int64)
#     P[0,:] = 0 # Reminder
#     r_n = r/n

#     return V, V_previous, P, r_n


@nb.jit
def viterbi_init(n, m, H, s, e, r):
    """Initialise naive, but more space memory efficient implementation of LS viterbi."""
    V_previous = np.zeros(n)
    V = np.zeros(n)
    P = np.zeros((m, n)).astype(np.int64)
    r_n = r / n

    for i in range(n):
        V_previous[i] = 1 / n * e[0, np.int64(np.equal(H[0, i], s[0, 0]))]

    return V, V_previous, P, r_n


# def forwards_viterbi_hap_naive(n, m, H, s, e, r):
#     '''
#     Simple naive LS forward Viterbi algorithm.
#     '''

#     # Initialise
#     V, P, r_n = viterbi_naive_init(n, m, H, s, e, r)

#     for j in range(1,m):
#         for i in range(n):
#             # Get the vector to maximise over
#             v = np.zeros(n)
#             for k in range(n):
#                 # v[k] = e[j,np.equal(H[j,i], s[0,j]).astype(np.int64)] * V[j-1,k]
#                 v[k] = e[j,np.int64(np.equal(H[j,i], s[0,j]))] * V[j-1,k]
#                 if k == i:
#                     v[k] *= 1 - r[j] + r_n[j]
#                 else:
#                     v[k] *= r_n[j]
#             P[j,i] = np.argmax(v)
#             V[j,i] = v[P[j,i]]

#     ll = np.log10(np.amax(V[m-1,:]))

#     return V, P, ll


@nb.jit
def forwards_viterbi_hap_naive(n, m, H, s, e, r):
    """Naive implementation of LS haploid Viterbi algorithm."""
    # Initialise
    V, P, r_n = viterbi_naive_init(n, m, H, s, e, r)

    for j in range(1, m):
        for i in range(n):
            # Get the vector to maximise over
            v = np.zeros(n)
            for k in range(n):
                # v[k] = e[j,np.equal(H[j,i], s[0,j]).astype(np.int64)] * V[j-1,k]
                v[k] = e[j, np.int64(np.equal(H[j, i], s[0, j]))] * V[j - 1, k]
                if k == i:
                    v[k] *= 1 - r[j] + r_n[j]
                else:
                    v[k] *= r_n[j]
            P[j, i] = np.argmax(v)
            V[j, i] = v[P[j, i]]

    ll = np.log10(np.amax(V[m - 1, :]))

    return V, P, ll


# def forwards_viterbi_hap_naive_vec(n, m, H, s, e, r):
#     '''
#     Simple matrix based method naive LS forward Viterbi algorithm. Vectorised things
#      - I jumped the gun!
#     '''

#     # Initialise
#     V, P, r_n = viterbi_naive_init(n, m, H, s, e, r)

#     for j in range(1,m):
#         v_tmp = V[j-1,:] * r_n[j]
#         for i in range(n):
#             v = np.copy(v_tmp)
#             v[i] += V[j-1,i] * (1 - r[j])
#             v *= e[j,np.int64(np.equal(H[j,i], s[0,j]))]
#             P[j,i] = np.argmax(v)
#             V[j,i] = v[P[j,i]]

#     ll = np.log10(np.amax(V[m-1,:]))

#     return V, P, ll


@nb.jit
def forwards_viterbi_hap_naive_vec(n, m, H, s, e, r):
    """Naive matrix based implementation of LS haploid forward Viterbi algorithm using numpy."""
    # Initialise
    V, P, r_n = viterbi_naive_init(n, m, H, s, e, r)

    for j in range(1, m):
        v_tmp = V[j - 1, :] * r_n[j]
        for i in range(n):
            v = np.copy(v_tmp)
            v[i] += V[j - 1, i] * (1 - r[j])
            v *= e[j, np.int64(np.equal(H[j, i], s[0, j]))]
            P[j, i] = np.argmax(v)
            V[j, i] = v[P[j, i]]

    ll = np.log10(np.amax(V[m - 1, :]))

    return V, P, ll


# def forwards_viterbi_hap_naive_low_mem(n, m, H, s, e, r):
#     '''
#     Simple naive LS forward Viterbi algorithm. More memory efficient.
#     '''

#     # Initialise
#     V, V_previous, P, r_n = viterbi_init(n, m, H, s, e, r)

#     for j in range(1,m):
#         for i in range(n):
#             # Get the vector to maximise over
#             v = np.zeros(n)
#             for k in range(n):
#                 v[k] = e[j,np.int64(np.equal(H[j,i], s[0,j]))] * V_previous[k]
#                 if k == i:
#                     v[k] *= 1 - r[j] + r_n[j]
#                 else:
#                     v[k] *= r_n[j]
#             P[j,i] = np.argmax(v)
#             V[i] = v[P[j,i]]
#         V_previous = np.copy(V)

#     ll = np.log10(np.amax(V))

#     return V, P, ll


@nb.jit
def forwards_viterbi_hap_naive_low_mem(n, m, H, s, e, r):
    """Naive implementation of LS haploid Viterbi algorithm, with reduced memory."""
    # Initialise
    V, V_previous, P, r_n = viterbi_init(n, m, H, s, e, r)

    for j in range(1, m):
        for i in range(n):
            # Get the vector to maximise over
            v = np.zeros(n)
            for k in range(n):
                v[k] = e[j, np.int64(np.equal(H[j, i], s[0, j]))] * V_previous[k]
                if k == i:
                    v[k] *= 1 - r[j] + r_n[j]
                else:
                    v[k] *= r_n[j]
            P[j, i] = np.argmax(v)
            V[i] = v[P[j, i]]
        V_previous = np.copy(V)

    ll = np.log10(np.amax(V))

    return V, P, ll


# def forwards_viterbi_hap_naive_low_mem_rescaling(n, m, H, s, e, r):
#     '''
#     Simple naive LS forward Viterbi algorithm. More memory efficient, and with
#     a rescaling to avoid underflow problems
#     '''

#     # Initialise
#     V, V_previous, P, r_n = viterbi_init(n, m, H, s, e, r)
#     c = np.ones(m)

#     for j in range(1,m):
#         c[j] =  np.amax(V_previous)
#         V_previous *= 1/c[j]
#         for i in range(n):
#             # Get the vector to maximise over
#             v = np.zeros(n)
#             for k in range(n):
#                 v[k] = e[j,np.int64(np.equal(H[j,i], s[0,j]))] * V_previous[k]
#                 if k == i:
#                     v[k] *= 1 - r[j] + r_n[j]
#                 else:
#                     v[k] *= r_n[j]
#             P[j,i] = np.argmax(v)
#             V[i] = v[P[j,i]]

#         V_previous = np.copy(V)

#     ll = np.sum(np.log10(c)) + np.log10(np.amax(V))

#     return V, P, ll


@nb.jit
def forwards_viterbi_hap_naive_low_mem_rescaling(n, m, H, s, e, r):
    """Naive implementation of LS haploid Viterbi algorithm, with reduced memory and rescaling."""
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
                v[k] = e[j, np.int64(np.equal(H[j, i], s[0, j]))] * V_previous[k]
                if k == i:
                    v[k] *= 1 - r[j] + r_n[j]
                else:
                    v[k] *= r_n[j]
            P[j, i] = np.argmax(v)
            V[i] = v[P[j, i]]

        V_previous = np.copy(V)

    ll = np.sum(np.log10(c)) + np.log10(np.amax(V))

    return V, P, ll


# def forwards_viterbi_hap_low_mem_rescaling(n, m, H, s, e, r):
#     '''
#     Simple LS forward Viterbi algorithm. Smaller memory footprint and rescaling,
#     and considers the structure of the Markov process.
#     '''

#     # Initialise
#     V, V_previous, P, r_n = viterbi_init(n, m, H, s, e, r)
#     c = np.ones(m)

#     for j in range(1,m):
#         argmax = np.argmax(V_previous)
#         c[j] =  V_previous[argmax]
#         V_previous *= 1/c[j]
#         V = np.zeros(n)
#         for i in range(n):
#             V[i] = V_previous[i] * (1 - r[j] + r_n[j])
#             P[j,i] = i
#             if V[i] < r_n[j]:
#                 V[i] = r_n[j]
#                 P[j,i] = argmax
#             V[i] *= e[j,np.equal(H[j,i], s[0,j]).astype(np.int64)]
#         V_previous = np.copy(V)

#     ll = np.sum(np.log10(c)) + np.log10(np.max(V))

#     return V, P, ll


@nb.jit
def forwards_viterbi_hap_low_mem_rescaling(n, m, H, s, e, r):
    """LS haploid Viterbi algorithm, with reduced memory and exploits the Markov process structure."""
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
            P[j, i] = i
            if V[i] < r_n[j]:
                V[i] = r_n[j]
                P[j, i] = argmax
            V[i] *= e[j, np.int64(np.equal(H[j, i], s[0, j]))]
        V_previous = np.copy(V)

    ll = np.sum(np.log10(c)) + np.log10(np.max(V))

    return V, P, ll


# def forwards_viterbi_hap_lower_mem_rescaling(n, m, H, s, e, r):
#     '''
#     Simple LS forward Viterbi algorithm. Even smaller memory footprint and rescaling,
#     and considers the structure of the Markov process.
#     '''

#     # Initialise
#     V = 1/n * e[0, np.equal(H[0,:], s[0,0]).astype(np.int64)]
#     P = np.zeros((m,n)).astype(np.int64)
#     # P[0,:] = 0
#     r_n = r/n
#     c = np.ones(m)

#     for j in range(1,m):
#         argmax = np.argmax(V)
#         c[j] =  V[argmax]
#         V *= 1/c[j]
#         for i in range(n):
#             V[i] = V[i] * (1 - r[j] + r_n[j])
#             P[j,i] = i
#             if V[i] < r_n[j]:
#                 V[i] = r_n[j]
#                 P[j,i] = argmax
#             V[i] *= e[j,np.int64(np.equal(H[j,i], s[0,j]))]

#     ll = np.sum(np.log10(c)) + np.log10(np.max(V))

#     return V, P, ll


@nb.jit
def forwards_viterbi_hap_lower_mem_rescaling(n, m, H, s, e, r):
    """LS haploid Viterbi algorithm with even smaller memory footprint and exploits the Markov process structure."""
    # Initialise
    V = np.zeros(n)
    for i in range(n):
        V[i] = 1 / n * e[0, np.int64(np.equal(H[0, i], s[0, 0]))]
    P = np.zeros((m, n)).astype(np.int64)
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
            V[i] *= e[j, np.int64(np.equal(H[j, i], s[0, j]))]

    ll = np.sum(np.log10(c)) + np.log10(np.max(V))

    return V, P, ll


@nb.jit
def forwards_viterbi_hap_lower_mem_rescaling_no_pointer(n, m, H, s, e, r):
    """LS haploid Viterbi algorithm with even smaller memory footprint and exploits the Markov process structure."""
    # Initialise
    V = np.zeros(n)
    for i in range(n):
        V[i] = 1 / n * e[0, np.int64(np.equal(H[0, i], s[0, 0]))]
    r_n = r / n
    c = np.ones(m)
    recombs = [
        set() for _ in range(m)
    ]  # This is going to be filled with the templates we can recombine to that have higher prob than staying where we are.
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
                recombs[j].add(
                    i
                )  # We add template i as a potential template to recombine to at site j.
            V[i] *= e[j, np.int64(np.equal(H[j, i], s[0, j]))]

    V_argmaxes[m - 1] = np.argmax(V)
    ll = np.sum(np.log10(c)) + np.log10(np.max(V))

    return V, V_argmaxes, recombs, ll


# Speedier version, variants x samples
@nb.jit
def backwards_viterbi_hap(m, V_last, P):
    """Run a backwards pass to determine the most likely path."""
    # Initialise
    assert len(V_last.shape) == 1
    path = np.zeros(m).astype(np.int64)
    path[m - 1] = np.argmax(V_last)

    for j in range(m - 2, -1, -1):
        path[j] = P[j + 1, path[j + 1]]

    return path


@nb.jit
def backwards_viterbi_hap_no_pointer(m, V_argmaxes, recombs):
    """Run a backwards pass to determine the most likely path."""
    # Initialise
    path = np.zeros(m).astype(np.int64)
    path[m - 1] = V_argmaxes[m - 1]

    for j in range(m - 2, -1, -1):
        current_best_template = path[j + 1]
        if current_best_template in recombs[j + 1]:
            current_best_template = V_argmaxes[j]
        path[j] = current_best_template

    return path


@nb.jit
def path_ll_hap(n, m, H, path, s, e, r):
    """Evaluate log-likelihood path through a reference panel which results in sequence s."""
    index = np.int64(np.equal(H[0, path[0]], s[0, 0]))
    log_prob_path = np.log10((1 / n) * e[0, index])
    old = path[0]
    r_n = r / n

    for l in range(1, m):
        index = np.int64(np.equal(H[l, path[l]], s[0, l]))
        current = path[l]
        same = old == current

        if same:
            log_prob_path += np.log10((1 - r[l]) + r_n[l])
        else:
            log_prob_path += np.log10(r_n[l])

        log_prob_path += np.log10(e[l, index])
        old = current

    return log_prob_path
