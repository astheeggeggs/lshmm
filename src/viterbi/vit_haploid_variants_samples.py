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
    """
    Initialisation portion of initial naive implementation of LS viterbi to avoid
    lots of code duplication
    """

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
    """
    Initialisation portion of initial naive, but more space memory efficient implementation
    of LS viterbi to avoid lots of code duplication
    """

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
    """
    Simple matrix based method naive LS forward Viterbi algorithm. Vectorised things
     - I jumped the gun!
    """

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
    """
    Simple LS forward Viterbi algorithm. Even smaller memory footprint and rescaling,
    and considers the structure of the Markov process.
    """

    # Initialise
    V = np.zeros(n)
    for i in range(n):
        V[i] = 1 / n * e[0, np.int(np.equal(H[0, i], s[0, 0]))]
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


# Speedier version, variants x samples
@nb.jit
def backwards_viterbi_hap(m, V_last, P):
    """
    Backwards pass to determine the most likely path
    """

    # Initialise
    path = np.zeros(m).astype(np.int64)
    path[m - 1] = np.argmax(V_last)

    for j in range(m - 2, -1, -1):
        path[j] = P[j + 1, path[j + 1]]

    return path
