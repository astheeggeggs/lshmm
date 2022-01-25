"""Collection of functions to run forwards and backwards algorithms on haploid genotype data, where the data is structured as variants x samples."""
import numba as nb
import numpy as np

EQUAL_BOTH_HOM = 4
UNEQUAL_BOTH_HOM = 0
BOTH_HET = 7
REF_HOM_OBS_HET = 1
REF_HET_OBS_HOM = 2

# https://github.com/numba/numba/issues/1269
@nb.njit
def np_apply_along_axis(func1d, axis, arr):
    """Create numpy-like functions for max, sum etc."""
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@nb.njit
def np_amax(array, axis):
    """Numba implementation of numpy vectorised maximum."""
    return np_apply_along_axis(np.amax, axis, array)


@nb.njit
def np_sum(array, axis):
    """Numba implementation of numpy vectorised sum."""
    return np_apply_along_axis(np.sum, axis, array)


@nb.njit
def np_argmax(array, axis):
    """Numba implementation of numpy vectorised argmax."""
    return np_apply_along_axis(np.argmax, axis, array)


def forwards_ls_dip(n, m, G, s, e, r):
    """Matrix based diploid LS forward algorithm using numpy vectorisation."""
    # Initialise the forward tensor
    F = np.zeros((m, n, n))
    F[0, :, :] = 1 / (n ** 2)
    index = (
        4 * np.equal(G[0, :, :], s[0, 0]).astype(np.int64)
        + 2 * (G[0, :, :] == 1).astype(np.int64)
        + np.int64(s[0, 0] == 1)
    )
    F[0, :, :] *= e[0, index]
    c = np.ones(m)
    r_n = r / n

    # Forwards
    for l in range(1, m):

        index = (
            4 * np.equal(G[l, :, :], s[0, l]).astype(np.int64)
            + 2 * (G[l, :, :] == 1).astype(np.int64)
            + np.int64(s[0, l] == 1)
        )

        # No change in both
        F[l, :, :] = (1 - r[l]) ** 2 * F[l - 1, :, :]

        # Both change
        F[l, :, :] += (r_n[l]) ** 2 * np.sum(F[l - 1, :, :])

        # One changes
        # sum_j1 = np.tile(np.sum(F[l-1,:,:], 0, keepdims=True), (n,1))
        sum_j1 = np_sum(F[l - 1, :, :], 0).repeat(n).reshape((-1, n)).T
        # sum_j2 = np.tile(np.sum(F[l-1,:,:], 1, keepdims=True), (1,n))
        sum_j2 = np_sum(F[l - 1, :, :], 1).repeat(n).reshape((-1, n))
        F[l, :, :] += ((1 - r[l]) * r_n[l]) * (sum_j1 + sum_j2)

        # Emission
        F[l, :, :] *= e[l, index]
        c[l] = np.sum(F[l, :, :])
        F[l, :, :] *= 1 / c[l]

    ll = np.sum(np.log10(c))
    return F, c, ll


def backwards_ls_dip(n, m, G, s, e, c, r):
    """Matrix based diploid LS backward algorithm using numpy vectorisation."""
    # Initialise the backward tensor
    B = np.zeros((m, n, n))

    # Initialise
    B[m - 1, :, :] = 1
    r_n = r / n

    # Backwards
    for l in range(m - 2, -1, -1):

        index = (
            4 * np.equal(G[l + 1, :, :], s[0, l + 1]).astype(np.int64)
            + 2 * (G[l + 1, :, :] == 1).astype(np.int64)
            + np.int64(s[0, l + 1] == 1)
        )

        # No change in both
        B[l, :, :] = r_n[l + 1] ** 2 * np.sum(
            e[l + 1, index.reshape((n, n))] * B[l + 1, :, :]
        )

        # Both change
        B[l, :, :] += (
            (1 - r[l + 1]) ** 2 * B[l + 1, :, :] * e[l + 1, index.reshape((n, n))]
        )

        # One changes
        # sum_j1 = np.tile(np.sum(B[l+1,:,:] * e[l+1, index], 0, keepdims=True), (n,1))
        sum_j1 = np_sum(B[l + 1, :, :], 0).repeat(n).reshape((-1, n)).T
        # sum_j2 = np.tile(np.sum(B[l+1,:,:] * e[l+1, index], 1, keepdims=True), (1,n))
        sum_j2 = np_sum(B[l + 1, :, :], 1).repeat(n).reshape((-1, n))
        B[l, :, :] += ((1 - r[l + 1]) * r_n[l + 1]) * (sum_j1 + sum_j2)
        B[l, :, :] *= 1 / c[l + 1]

    return B


# def forward_ls_dip_starting_point(n, m, G, s, e, r):
#     '''
#     Unbelievably naive implementation of LS diploid forwards. Just to get something down
#     that works.
#     '''

#     # Initialise the forward tensor
#     F = np.zeros((m,n,n))
#     F[0,:,:] = 1/(n**2)
#     index = (
#         4*np.equal(G[0,:,:], s[0,0]).astype(np.int64) +
#         2*(G[0,:,:] == 1).astype(np.int64) +
#         np.int64(s[0,0] == 1)
#         )
#     F[0,:,:] *= e[0, index]
#     r_n = r/n

#     for l in range(1,m):

#         # Determine the various components
#         F_no_change = np.zeros((n,n))
#         F_j1_change = np.zeros(n)
#         F_j2_change = np.zeros(n)
#         F_both_change = 0

#         for j1 in range(n):
#             for j2 in range(n):
#                 F_no_change[j1, j2] = (1-r[l])**2 * F[l-1, j1, j2]

#         for j1 in range(n):
#             for j2 in range(n):
#                 F_both_change += r_n[l]**2 * F[l-1, j1, j2]

#         for j1 in range(n):
#             for j2 in range(n): # This is the variable to sum over - it changes
#                 F_j2_change[j1] += (1 - r[l]) * r_n[l] * F[l-1, j1, j2]

#         for j2 in range(n):
#             for j1 in range(n): # This is the variable to sum over - it changes
#                 F_j1_change[j2] += (1 - r[l]) * r_n[l] * F[l-1, j1, j2]

#         F[l,:,:] = F_both_change

#         for j1 in range(n):
#             F[l, j1, :] += F_j2_change

#         for j2 in range(n):
#             F[l, :, j2] += F_j1_change

#         for j1 in range(n):
#             for j2 in range(n):
#                 F[l, j1, j2] += F_no_change[j1, j2]

#         for j1 in range(n):
#             for j2 in range(n):
#                 # What is the emission?
#                 if s[0,l] == 1:
#                     # OBS is het
#                     if G[l, j1, j2] == 1: # REF is het
#                         F[l, j1, j2] *= e[l,BOTH_HET]
#                     else: # REF is hom
#                         F[l, j1, j2] *= e[l,REF_HOM_OBS_HET]
#                 else:
#                     # OBS is hom
#                     if G[l, j1, j2] == 1: # REF is het
#                         F[l, j1, j2] *= e[l,REF_HET_OBS_HOM]
#                     else: # REF is hom
#                         if G[l, j1, j2] == s[0,l]: # Equal
#                             F[l, j1, j2] *= e[l,EQUAL_BOTH_HOM]
#                         else: # Unequal
#                             F[l, j1, j2] *= e[l,UNEQUAL_BOTH_HOM]

#     ll = np.log10(np.sum(F[l,:,:]))

#     return F, ll


@nb.njit
def forward_ls_dip_starting_point(n, m, G, s, e, r):
    """Naive implementation of LS diploid forwards algorithm."""
    # Initialise the forward tensor
    F = np.zeros((m, n, n))
    r_n = r / n
    for j1 in range(n):
        for j2 in range(n):
            F[0, j1, j2] = 1 / (n ** 2)
            index_tmp = (
                4 * np.int64(np.equal(G[0, j1, j2], s[0, 0]))
                + 2 * np.int64((G[0, j1, j2] == 1))
                + np.int64(s[0, 0] == 1)
            )
            F[0, j1, j2] *= e[0, index_tmp]

    for l in range(1, m):

        # Determine the various components
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
                # What is the emission?
                if s[0, l] == 1:
                    # OBS is het
                    if G[l, j1, j2] == 1:  # REF is het
                        F[l, j1, j2] *= e[l, BOTH_HET]
                    else:  # REF is hom
                        F[l, j1, j2] *= e[l, REF_HOM_OBS_HET]
                else:
                    # OBS is hom
                    if G[l, j1, j2] == 1:  # REF is het
                        F[l, j1, j2] *= e[l, REF_HET_OBS_HOM]
                    else:  # REF is hom
                        if G[l, j1, j2] == s[0, l]:  # Equal
                            F[l, j1, j2] *= e[l, EQUAL_BOTH_HOM]
                        else:  # Unequal
                            F[l, j1, j2] *= e[l, UNEQUAL_BOTH_HOM]

    ll = np.log10(np.sum(F[l, :, :]))

    return F, ll


# def backward_ls_dip_starting_point(n, m, G, s, e, r):
#     '''
#     Unbelievably naive implementation of LS diploid backwards. Just to get something down
#     that works.
#     '''

#     # Backwards
#     B = np.zeros((m,n,n))

#     # Initialise
#     B[m-1, :, :] = 1
#     r_n = r/n

#     for l in range(m-2, -1, -1):

#         # Determine the various components
#         B_no_change = np.zeros((n,n))
#         B_j1_change = np.zeros(n)
#         B_j2_change = np.zeros(n)
#         B_both_change = 0

#         # Evaluate the emission matrix at this site, for all pairs
#         e_tmp = np.zeros((n,n))
#         for j1 in range(n):
#             for j2 in range(n):
#                 # What is the emission?
#                 if s[0,l+1] == 1:
#                     # OBS is het
#                     if G[l+1, j1, j2] == 1: # REF is het
#                         e_tmp[j1, j2] = e[l+1, BOTH_HET]
#                     else: # REF is hom
#                         e_tmp[j1, j2] = e[l+1, REF_HOM_OBS_HET]
#                 else:
#                     # OBS is hom
#                     if G[l+1, j1, j2] == 1: # REF is het
#                         e_tmp[j1, j2] = e[l+1,REF_HET_OBS_HOM]
#                     else: # REF is hom
#                         if G[l+1, j1, j2] == s[0,l+1]: # Equal
#                             e_tmp[j1, j2] = e[l+1,EQUAL_BOTH_HOM]
#                         else: # Unequal
#                             e_tmp[j1, j2] = e[l+1,UNEQUAL_BOTH_HOM]

#         for j1 in range(n):
#             for j2 in range(n):
#                 B_no_change[j1, j2] = (1-r[l+1])**2 * B[l+1,j1,j2] * e_tmp[j1, j2]

#         for j1 in range(n):
#             for j2 in range(n):
#                 B_both_change += r_n[l+1]**2 * e_tmp[j1, j2] * B[l+1,j1,j2]

#         for j1 in range(n):
#             for j2 in range(n): # This is the variable to sum over - it changes
#                 B_j2_change[j1] += (1 - r[l+1]) * r_n[l+1] * B[l+1,j1,j2] * e_tmp[j1, j2]

#         for j2 in range(n):
#             for j1 in range(n): # This is the variable to sum over - it changes
#                 B_j1_change[j2] += (1 - r[l+1]) * r_n[l+1] * B[l+1,j1,j2] * e_tmp[j1, j2]

#         B[l,:,:] = B_both_change

#         for j1 in range(n):
#             B[l, j1, :] += B_j2_change

#         for j2 in range(n):
#             B[l, :, j2] += B_j1_change

#         for j1 in range(n):
#             for j2 in range(n):
#                 B[l, j1, j2] += B_no_change[j1, j2]

#     return B


@nb.njit
def backward_ls_dip_starting_point(n, m, G, s, e, r):
    """Naive implementation of LS diploid backwards algorithm."""
    # Backwards
    B = np.zeros((m, n, n))

    # Initialise
    B[m - 1, :, :] = 1
    r_n = r / n

    for l in range(m - 2, -1, -1):

        # Determine the various components
        B_no_change = np.zeros((n, n))
        B_j1_change = np.zeros(n)
        B_j2_change = np.zeros(n)
        B_both_change = 0

        # Evaluate the emission matrix at this site, for all pairs
        e_tmp = np.zeros((n, n))
        for j1 in range(n):
            for j2 in range(n):
                # What is the emission?
                if s[0, l + 1] == 1:
                    # OBS is het
                    if G[l + 1, j1, j2] == 1:  # REF is het
                        e_tmp[j1, j2] = e[l + 1, BOTH_HET]
                    else:  # REF is hom
                        e_tmp[j1, j2] = e[l + 1, REF_HOM_OBS_HET]
                else:
                    # OBS is hom
                    if G[l + 1, j1, j2] == 1:  # REF is het
                        e_tmp[j1, j2] = e[l + 1, REF_HET_OBS_HOM]
                    else:  # REF is hom
                        if G[l + 1, j1, j2] == s[0, l + 1]:  # Equal
                            e_tmp[j1, j2] = e[l + 1, EQUAL_BOTH_HOM]
                        else:  # Unequal
                            e_tmp[j1, j2] = e[l + 1, UNEQUAL_BOTH_HOM]

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


# def forward_ls_dip_loop(n, m, G, s, e, r):
#     '''
#     LS diploid forwards with lots of loops.
#     '''

#     # Initialise the forward tensor
#     F = np.zeros((m,n,n))
#     F[0,:,:] = 1/(n**2)
#     index = (
#         4*np.equal(G[0,:,:], s[0,0]).astype(np.int64) +
#         2*(G[0,:,:] == 1).astype(np.int64) +
#         np.int64(s[0,0] == 1)
#         )
#     F[0,:,:] *= e[0, index]
#     r_n = r/n

#     for l in range(1,m):

#         # Determine the various components
#         F_no_change = np.zeros((n,n))
#         F_j1_change = np.zeros(n)
#         F_j2_change = np.zeros(n)
#         F_both_change = 0

#         for j1 in range(n):
#             for j2 in range(n):
#                 F_no_change[j1, j2] = (1-r[l])**2 * F[l-1,j1,j2]
#                 F_j1_change[j1] += (1 - r[l]) * r_n[l] * F[l-1,j2,j1]
#                 F_j2_change[j1] += (1 - r[l]) * r_n[l] * F[l-1,j1,j2]
#                 F_both_change += r_n[l]**2 * F[l-1,j1,j2]

#         F[l,:,:] = F_both_change

#         for j1 in range(n):
#             F[l, j1, :] += F_j2_change
#             F[l, :, j1] += F_j1_change
#             for j2 in range(n):
#                 F[l, j1, j2] += F_no_change[j1, j2]

#         for j1 in range(n):
#             for j2 in range(n):
#                 # What is the emission?
#                 if s[0,l] == 1:
#                     # OBS is het
#                     if G[l, j1, j2] == 1: # REF is het
#                         F[l, j1, j2] *= e[l, BOTH_HET]
#                     else: # REF is hom
#                         F[l, j1, j2] *= e[l, REF_HOM_OBS_HET]
#                 else:
#                     # OBS is hom
#                     if G[l, j1, j2] == 1: # REF is het
#                         F[l, j1, j2] *= e[l, REF_HET_OBS_HOM]
#                     else: # REF is hom
#                         if G[l, j1, j2] == s[0,l]: # Equal
#                             F[l, j1, j2] *= e[l, EQUAL_BOTH_HOM]
#                         else: # Unequal
#                             F[l, j1, j2] *= e[l, UNEQUAL_BOTH_HOM]

#     ll = np.log10(np.sum(F[l,:,:]))
#     return F, ll


@nb.njit
def forward_ls_dip_loop(n, m, G, s, e, r):
    """LS diploid forwards algoritm without vectorisation."""
    # Initialise the forward tensor
    F = np.zeros((m, n, n))
    r_n = r / n
    for j1 in range(n):
        for j2 in range(n):
            F[0, j1, j2] = 1 / (n ** 2)
            index_tmp = (
                4 * np.int64(np.equal(G[0, j1, j2], s[0, 0]))
                + 2 * np.int64((G[0, j1, j2] == 1))
                + np.int64(s[0, 0] == 1)
            )
            F[0, j1, j2] *= e[0, index_tmp]

    for l in range(1, m):

        # Determine the various components
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
                # What is the emission?
                if s[0, l] == 1:
                    # OBS is het
                    if G[l, j1, j2] == 1:  # REF is het
                        F[l, j1, j2] *= e[l, BOTH_HET]
                    else:  # REF is hom
                        F[l, j1, j2] *= e[l, REF_HOM_OBS_HET]
                else:
                    # OBS is hom
                    if G[l, j1, j2] == 1:  # REF is het
                        F[l, j1, j2] *= e[l, REF_HET_OBS_HOM]
                    else:  # REF is hom
                        if G[l, j1, j2] == s[0, l]:  # Equal
                            F[l, j1, j2] *= e[l, EQUAL_BOTH_HOM]
                        else:  # Unequal
                            F[l, j1, j2] *= e[l, UNEQUAL_BOTH_HOM]

    ll = np.log10(np.sum(F[l, :, :]))
    return F, ll


# def backward_ls_dip_loop(n, m, G, s, e, r):
#     '''
#     LS diploid backwards with lots of loops.
#     '''

#     # Initialise the backward tensor
#     B = np.zeros((m,n,n))
#     B[m-1, :, :] = 1
#     r_n = r/n

#     for l in range(m-2, -1, -1):

#         # Determine the various components
#         B_no_change = np.zeros((n,n))
#         B_j1_change = np.zeros(n)
#         B_j2_change = np.zeros(n)
#         B_both_change = 0

#         # Evaluate the emission matrix at this site, for all pairs
#         e_tmp = np.zeros((n,n))
#         for j1 in range(n):
#             for j2 in range(n):
#                 # What is the emission?
#                 if s[0,l+1] == 1:
#                     # OBS is het
#                     if G[l+1, j1, j2] == 1: # REF is het
#                         e_tmp[j1, j2] = e[l+1, BOTH_HET]
#                     else: # REF is hom
#                         e_tmp[j1, j2] = e[l+1, REF_HOM_OBS_HET]
#                 else:
#                     # OBS is hom
#                     if G[l+1, j1, j2] == 1: # REF is het
#                         e_tmp[j1, j2] = e[l+1, REF_HET_OBS_HOM]
#                     else: # REF is hom
#                         if G[l+1, j1, j2] == s[0,l+1]: # Equal
#                             e_tmp[j1, j2] = e[l+1, EQUAL_BOTH_HOM]
#                         else: # Unequal
#                             e_tmp[j1, j2] = e[l+1, UNEQUAL_BOTH_HOM]

#         for j1 in range(n):
#             for j2 in range(n):
#                 B_no_change[j1, j2] = (1-r[l+1])**2 * B[l+1,j1,j2] * e_tmp[j1, j2]
#                 B_j2_change[j1] += (1 - r[l+1]) * r_n[l+1] * B[l+1,j1,j2] * e_tmp[j1, j2]
#                 B_j1_change[j1] += (1 - r[l+1]) * r_n[l+1] * B[l+1,j2,j1] * e_tmp[j2, j1]
#                 B_both_change += r_n[l+1]**2 * e_tmp[j1, j2] * B[l+1,j1,j2]

#         B[l,:,:] = B_both_change

#         for j1 in range(n):
#             B[l, j1, :] += B_j2_change
#             B[l, :, j1] += B_j1_change
#             for j2 in range(n):
#                 B[l, j1, j2] += B_no_change[j1, j2]

#     return B


@nb.njit
def backward_ls_dip_loop(n, m, G, s, e, r):
    """LS diploid backwards algoritm without vectorisation."""
    # Initialise the backward tensor
    B = np.zeros((m, n, n))
    B[m - 1, :, :] = 1
    r_n = r / n

    for l in range(m - 2, -1, -1):

        # Determine the various components
        B_no_change = np.zeros((n, n))
        B_j1_change = np.zeros(n)
        B_j2_change = np.zeros(n)
        B_both_change = 0

        # Evaluate the emission matrix at this site, for all pairs
        e_tmp = np.zeros((n, n))
        for j1 in range(n):
            for j2 in range(n):
                # What is the emission?
                if s[0, l + 1] == 1:
                    # OBS is het
                    if G[l + 1, j1, j2] == 1:  # REF is het
                        e_tmp[j1, j2] = e[l + 1, BOTH_HET]
                    else:  # REF is hom
                        e_tmp[j1, j2] = e[l + 1, REF_HOM_OBS_HET]
                else:
                    # OBS is hom
                    if G[l + 1, j1, j2] == 1:  # REF is het
                        e_tmp[j1, j2] = e[l + 1, REF_HET_OBS_HOM]
                    else:  # REF is hom
                        if G[l + 1, j1, j2] == s[0, l + 1]:  # Equal
                            e_tmp[j1, j2] = e[l + 1, EQUAL_BOTH_HOM]
                        else:  # Unequal
                            e_tmp[j1, j2] = e[l + 1, UNEQUAL_BOTH_HOM]

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

    return B
