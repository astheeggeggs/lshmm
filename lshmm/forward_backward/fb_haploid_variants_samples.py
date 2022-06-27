"""Collection of functions to run forwards and backwards algorithms on haploid genotype data, where the data is structured as variants x samples."""
import numba as nb
import numpy as np

# def forwards_ls_hap(n, m, H, s, e, r, norm=True):
#     '''
#     Simple matrix based method for LS forward algorithm using numpy vectorisation.
#     '''

#     # Initialise
#     F = np.zeros((m,n))
#     c = np.ones(m)
#     F[0,:] = 1/n * e[0, np.equal(H[0, :], s[0,0]).astype(np.int64)]
#     r_n = r/n

#     if norm:

#         c[0] = np.sum(F[0,:])
#         F[0,:] *= 1/c[0]

#         # Forwards pass
#         for l in range(1,m):
#             F[l,:] = F[l-1,:] * (1 - r[l]) + r_n[l]  # Don't need to multiply by r_n[l] F[:,l-1] as we've normalised.
#             F[l,:] *= e[l,np.equal(H[l,:], s[0,l]).astype(np.int64)]
#             c[l] = np.sum(F[l,:])
#             F[l,:] *= 1/c[l]

#         ll = np.sum(np.log10(c))

#     else:
#         # Forwards pass
#         for l in range(1,m):
#             F[l,:] = F[l-1,:] * (1 - r[l]) + np.sum(F[l-1,:]) * r_n[l]
#             F[l,:] *= e[l, np.equal(H[l,:], s[0,l]).astype(np.int64)]

#         ll = np.log10(np.sum(F[m-1,:]))

#     return F, c, ll


@nb.jit
def forwards_ls_hap(n, m, H, s, e, r, norm=True):
    """Matrix based haploid LS forward algorithm using numpy vectorisation."""
    # Initialise
    F = np.zeros((m, n))
    r_n = r / n
    print("running")

    if norm:

        c = np.zeros(m)
        for i in range(n):
            print(e[0, np.int64(np.equal(H[0, i], s[0, 0]))])
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


# def backwards_ls_hap(n, m, H, s, e, c, r):
#     '''
#     Simple matrix based method for LS backwards algorithm using numpy vectorisation.
#     '''

#     # Initialise
#     B = np.zeros((m,n))
#     B[m-1,:] = 1
#     r_n = r/n

#     # Backwards pass
#     for l in range(m-2, -1, -1):
#         B[l,:] = r_n[l+1] * np.sum(e[l+1, np.equal(H[l+1,:], s[0,l+1]).astype(np.int64)] * B[l+1,:])
#         B[l,:] += (1 - r[l+1]) * e[l+1, np.equal(H[l+1,:], s[0,l+1]).astype(np.int64)] * B[l+1,:]
#         B[l,:] *= 1/c[l+1]

#     return B


@nb.jit
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
            tmp_B[i] = (
                e[l + 1, np.int64(np.equal(H[l + 1, i], s[0, l + 1]))] * B[l + 1, i]
            )
            tmp_B_sum += tmp_B[i]
        for i in range(n):
            B[l, i] = r_n[l + 1] * tmp_B_sum
            B[l, i] += (1 - r[l + 1]) * tmp_B[i]
            B[l, i] *= 1 / c[l + 1]

    return B
