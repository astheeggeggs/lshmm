# """Collection of functions to run Viterbi algorithms on haploid genotype data, where the data is structured as variants x samples."""
# import numba as nb
# import numpy as np

# def check_alleles(alleles, m):
#     """
#     Checks the specified allele list and returns a list of lists
#     of alleles of length num_sites.
#     If alleles is a 1D list of strings, assume that this list is used
#     for each site and return num_sites copies of this list.
#     Otherwise, raise a ValueError if alleles is not a list of length
#     num_sites.
#     """
#     if isinstance(alleles[0], str):
#         return np.int8([len(alleles) for _ in range(m)])
#     if len(alleles) != m:
#         raise ValueError("Malformed alleles list")
#     n_alleles = np.int8([(len(alleles_site)) for alleles_site in alleles])
#     return n_alleles

# # Speedier version, variants x samples

# @nb.jit
# def viterbi_naive_init(n, m, H, s, e, r):
#     """Initialise naive implementation of LS viterbi."""
#     V = np.zeros((m, n))
#     P = np.zeros((m, n)).astype(np.int64)
#     r_n = r / n
#     for i in range(n):
#         V[0, i] = 1 / n * e[0, np.int64(np.equal(H[0, i], s[0, 0]))]

#     return V, P, r_n

# @nb.jit
# def viterbi_init(n, m, H, s, e, r):
#     """Initialise naive, but more space memory efficient implementation of LS viterbi."""
#     V_previous = np.zeros(n)
#     V = np.zeros(n)
#     P = np.zeros((m, n)).astype(np.int64)
#     r_n = r / n

#     for i in range(n):
#         V_previous[i] = 1 / n * e[0, np.int64(np.equal(H[0, i], s[0, 0]))]

#     return V, V_previous, P, r_n


# @nb.jit
# def forwards_viterbi_hap_naive(n, m, n_alleles, H, s, e, r):
#     """Naive implementation of LS haploid Viterbi algorithm."""
#     # Initialise
#     V, P, r_n = viterbi_naive_init(n, m, H, s, e, r)

#     for j in range(1, m):
#         for i in range(n):
#             # Get the vector to maximise over
#             v = np.zeros(n)
#             for k in range(n):
#                 # v[k] = e[j,np.equal(H[j,i], s[0,j]).astype(np.int64)] * V[j-1,k]
#                 v[k] = e[j, np.int64(np.equal(H[j, i], s[0, j]))] * V[j - 1, k]
#                 if k == i:
#                     v[k] *= 1 - r[j] + r_n[j]
#                 else:
#                     v[k] *= r_n[j]
#             P[j, i] = np.argmax(v)
#             V[j, i] = v[P[j, i]]

#     ll = np.log10(np.amax(V[m - 1, :]))

#     return V, P, ll


# @nb.jit
# def forwards_viterbi_hap_lower_mem_rescaling(n, m, n_alleles, H, s, e, r):
#     """LS haploid Viterbi algorithm with even smaller memory footprint and exploits the Markov process structure."""
#     # Initialise
#     V = np.zeros(n)
#     for i in range(n):
#         V[i] = 1 / n * e[0, np.int64(np.equal(H[0, i], s[0, 0]))]
#     P = np.zeros((m, n)).astype(np.int64)
#     r_n = r / n
#     c = np.ones(m)

#     for j in range(1, m):
#         argmax = np.argmax(V)
#         c[j] = V[argmax]
#         V *= 1 / c[j]
#         for i in range(n):
#             V[i] = V[i] * (1 - r[j] + r_n[j])
#             P[j, i] = i
#             if V[i] < r_n[j]:
#                 V[i] = r_n[j]
#                 P[j, i] = argmax
#             V[i] *= e[j, np.int64(np.equal(H[j, i], s[0, j]))]

#     ll = np.sum(np.log10(c)) + np.log10(np.max(V))

#     return V, P, ll


# def forwards_viterbi_hap_naive_wrapper(n, m, alleles, H, s, e, r):
#     n_alleles = check_alleles(alleles, m)
#     V, P, ll = forwards_viterbi_hap_naive(n, m, n_alleles, H, s, e, r)
#     return V, P, ll

# def forwards_viterbi_hap_lower_mem_rescaling_wrapper(n, m, alleles, H, s, e, r):
#     n_alleles = check_alleles(alleles, m)
#     V, P, ll = forwards_viterbi_hap_lower_mem_rescaling(n, m, n_alleles, H, s, e, r)
#     return V, P, ll

# # Speedier version, variants x samples
# @nb.jit
# def backwards_viterbi_hap(m, V_last, P):
#     """Run a backwards pass to determine the most likely path."""
#     # Initialise
#     assert len(V_last.shape) == 1
#     path = np.zeros(m).astype(np.int64)
#     path[m - 1] = np.argmax(V_last)

#     for j in range(m - 2, -1, -1):
#         path[j] = P[j + 1, path[j + 1]]

#     return path

# # DEV: This might need some work
# @nb.jit
# def path_ll_hap(n, m, H, path, s, e, r):
#     """Evaluate log-likelihood path through a reference panel which results in sequence s."""
#     index = np.int64(np.equal(H[0, path[0]], s[0, 0]))
#     log_prob_path = np.log10((1 / n) * e[0, index])
#     old = path[0]
#     r_n = r / n

#     for l in range(1, m):
#         index = np.int64(np.equal(H[l, path[l]], s[0, l]))
#         current = path[l]
#         same = old == current

#         if same:
#             log_prob_path += np.log10((1 - r[l]) + r_n[l])
#         else:
#             log_prob_path += np.log10(r_n[l])

#         log_prob_path += np.log10(e[l, index])
#         old = current

#     return log_prob_path
