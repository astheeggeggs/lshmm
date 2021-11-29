import numpy as np
import numba as nb
import time

# https://github.com/numba/numba/issues/1269
@nb.njit
def np_apply_along_axis(func1d, axis, arr):
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
  return np_apply_along_axis(np.amax, axis, array)

@nb.njit
def np_sum(array, axis):
  return np_apply_along_axis(np.sum, axis, array)

@nb.njit
def np_argmax(array, axis):
  return np_apply_along_axis(np.argmax, axis, array)

# Haploid Li and Stephen's
# DEV: add numba to the c-like functions.
# DEV: don't use argmax and amax
# DEV: make sure matrices are structured so the outer most loop corresponds to the first dimension etc.

# Speedier version, variants x samples
def rand_for_testing_hap(n, m, mean_r=1e-5, mean_mu=1e-5):
    '''
    Create a simple random haplotype matrix with no LD, a random haplotype
    vector to use as an observation, set a recombination probability vector,
    and up and random mutation vector for the emissions.
    '''

    # sites x samples
    H = np.random.randint(2, size = (m,n))

    # Recombination probability
    r = mean_r * np.ones(m) * ((np.random.rand(m) + 0.5)/2)
    r[0] = 0

    # Error probability
    mu = mean_mu * np.ones(m) * ((np.random.rand(m) + 0.5)/2)

    # New sequence
    s = np.random.randint(2, size = (1, m))

    # Define the emission probability matrix
    e  = np.zeros((m,2))
    e[:,0] = mu
    e[:,1] = 1 - mu

    return H, s, r, mu, e

# Speedier version, variants x samples
# @nb.jit
def forwards_ls_hap(n, m, H, s, e, r):
    '''
    Simple matrix based method for LS forward algorithm using numpy vectorisation.
    '''

    # Initialise
    F = np.zeros((m,n))
    c = np.ones(m)
    F[0,:] = 1/n * e[0, np.equal(H[0, :], s[0,0]).astype(np.int64)]
    r_n = r/n

    # Forwards pass
    for l in range(1,m):
        F[l,:] = F[l-1,:] * (1 - r[l]) + np.sum(F[l-1,:]) * r_n[l]
        F[l,:] *= e[l, np.equal(H[l,:], s[0,l]).astype(np.int64)]
        c[l] = np.sum(F[l,:])
        F[l,:] *= 1/c[l]

    ll = np.sum(np.log10(c))

    return F, c, ll

# Speedier version, variants x samples
# @nb.jit
def backwards_ls_hap(n, m, H, s, e, c, r):
    '''
    Simple matrix based method for LS backwards algorithm using numpy vectorisation.
    '''

    # Initialise 
    B = np.zeros((m,n))
    B[m-1,:] = 1
    r_n = r/n

    # Backwards pass
    for l in range(m-2, -1, -1):
        B[l,:] = r_n[l+1] * np.sum(e[l+1, np.equal(H[l+1,:], s[0,l+1]).astype(np.int64)] * B[l+1,:])
        B[l,:] += (1 - r[l+1]) * e[l+1, np.equal(H[l+1,:], s[0,l+1]).astype(np.int64)] * B[l+1,:]
        B[l,:] *= 1/c[l+1]

    return B

# Speedier version, variants x samples
# @nb.jit
def viterbi_naive_init(n, m, H, s, e, r):
    '''
    Initialisation portion of initial naive implementation of LS viterbi to avoid 
    lots of code duplication
    '''

    V = np.zeros((m,n))
    P = np.zeros((m,n)).astype(np.int64)
    V[0,:] = 1/n * e[0,np.equal(H[0,:], s[0,0]).astype(np.int64)]
    P[0,:] = 0 # Reminder
    r_n = r/n

    return V, P, r_n

# Speedier version, variants x samples
# @nb.jit
def viterbi_init(n, m, H, s, e, r):
    '''
    Initialisation portion of initial naive, but more space memory efficient implementation 
    of LS viterbi to avoid lots of code duplication
    '''

    V_previous = 1/n * e[0,np.equal(H[0,:], s[0,0]).astype(np.int64)]
    V = np.zeros(n)
    P = np.zeros((m,n)).astype(np.int64)
    P[0,:] = 0 # Reminder
    r_n = r/n

    return V, V_previous, P, r_n

# Speedier version, variants x samples
# @nb.jit
def forwards_viterbi_hap_naive(n, m, H, s, e, r):
    '''
    Simple naive LS forward Viterbi algorithm.
    '''

    # Initialise
    V, P, r_n = viterbi_naive_init(n, m, H, s, e, r)

    for j in range(1,m):
        for i in range(n):
            # Get the vector to maximise over
            v = np.zeros(n)
            for k in range(n):
                # v[k] = e[j,np.equal(H[j,i], s[0,j]).astype(np.int64)] * V[j-1,k]
                v[k] = e[j,np.int64(np.equal(H[j,i], s[0,j]))] * V[j-1,k]
                if k == i:
                    v[k] *= 1 - r[j] + r_n[j]
                else:
                    v[k] *= r_n[j]
            V[j,i] = np.amax(v)
            P[j,i] = np.argmax(v)

    ll = np.log10(np.amax(V[m-1,:]))

    return V, P, ll

# Speedier version, variants x samples
# @nb.jit
def forwards_viterbi_hap_naive_vec(n, m, H, s, e, r):
    '''
    Simple matrix based method naive LS forward Viterbi algorithm. Vectorised things
     - I jumped the gun!
    '''
    
    # Initialise
    V, P, r_n = viterbi_naive_init(n, m, H, s, e, r)

    for j in range(1,m):
        v_tmp = V[j-1,:] * r_n[j]
        for i in range(n):
            v = np.copy(v_tmp)
            v[i] += V[j-1,i] * (1 - r[j])
            v *= e[j,np.int64(np.equal(H[j,i], s[0,j]))]
            P[j,i] = np.argmax(v)
            V[j,i] = v[P[j,i]]

    ll = np.log10(np.amax(V[m-1,:]))

    return V, P, ll

# Speedier version, variants x samples
# @nb.jit
def forwards_viterbi_hap_naive_low_mem(n, m, H, s, e, r):
    '''
    Simple naive LS forward Viterbi algorithm. More memory efficient.
    '''

    # Initialise
    V, V_previous, P, r_n = viterbi_init(n, m, H, s, e, r)

    for j in range(1,m):
        for i in range(n):
            # Get the vector to maximise over
            v = np.zeros(n)
            for k in range(n):
                v[k] = e[j,np.int64(np.equal(H[j,i], s[0,j]))] * V_previous[k]
                if k == i:
                    v[k] *= 1 - r[j] + r_n[j]
                else:
                    v[k] *= r_n[j]
            V[i] = np.amax(v)
            P[j,i] = np.argmax(v)
        V_previous = np.copy(V)

    ll = np.log10(np.amax(V))

    return V, P, ll

# Speedier version, variants x samples
# @nb.jit
def forwards_viterbi_hap_naive_low_mem_rescaling(n, m, H, s, e, r):
    '''
    Simple naive LS forward Viterbi algorithm. More memory efficient, and with
    a rescaling to avoid underflow problems
    '''

    # Initialise
    V, V_previous, P, r_n = viterbi_init(n, m, H, s, e, r)
    c = np.ones(m)

    for j in range(1,m):
        c[j] =  np.amax(V_previous)
        V_previous *= 1/c[j]
        for i in range(n):
            # Get the vector to maximise over
            v = np.zeros(n)
            for k in range(n):
                v[k] = e[j,np.int64(np.equal(H[j,i], s[0,j]))] * V_previous[k]
                if k == i:
                    v[k] *= 1 - r[j] + r_n[j]
                else:
                    v[k] *= r_n[j]
            V[i] = np.amax(v)
            P[j,i] = np.argmax(v)
        V_previous = np.copy(V)

    ll = np.sum(np.log10(c)) + np.log10(np.amax(V))

    return V, P, ll

# Speedier version, variants x samples
# @nb.jit
def forwards_viterbi_hap_low_mem_rescaling(n, m, H, s, e, r):
    '''
    Simple LS forward Viterbi algorithm. Smaller memory footprint and rescaling,
    and considers the structure of the Markov process.
    '''
    
    # Initialise
    V, V_previous, P, r_n = viterbi_init(n, m, H, s, e, r)
    c = np.ones(m)

    for j in range(1,m):
        c[j] =  np.amax(V_previous)
        argmax = np.argmax(V_previous)
        V_previous *= 1/c[j]
        V = np.zeros(n)
        for i in range(n):
            V[i] = V_previous[i] * (1 - r[j] + r_n[j])
            P[j,i] = i
            if V[i] < r_n[j]:
                V[i] = r_n[j]
                P[j,i] = argmax
            V[i] *= e[j,np.equal(H[j,i], s[0,j]).astype(np.int64)]
        V_previous = np.copy(V)

    ll = np.sum(np.log10(c)) + np.log10(np.max(V))

    return V, P, ll

# Speedier version, variants x samples
# @nb.jit
def forwards_viterbi_hap_lower_mem_rescaling(n, m, H, s, e, r):
    '''
    Simple LS forward Viterbi algorithm. Even smaller memory footprint and rescaling,
    and considers the structure of the Markov process.
    '''

    # Initialise
    V = 1/n * e[0, np.equal(H[0,:], s[0,0]).astype(np.int64)]
    P = np.zeros((m,n)).astype(np.int64)
    # P[0,:] = 0
    r_n = r/n
    c = np.ones(m)

    for j in range(1,m):
        c[j] =  np.amax(V)
        argmax = np.argmax(V)
        V *= 1/c[j]
        for i in range(n):
            V[i] = V[i] * (1 - r[j] + r_n[j])
            P[j,i] = i
            if V[i] < r_n[j]:
                V[i] = r_n[j]
                P[j,i] = argmax
            V[i] *= e[j,np.int64(np.equal(H[j,i], s[0,j]))]

    ll = np.sum(np.log10(c)) + np.log10(np.max(V))

    return V, P, ll

# Speedier version, variants x samples
@nb.jit
def backwards_viterbi_hap(m, V_last, P):
    '''
    Backwards pass to determine the most likely path
    '''

    # Initialise
    path = np.zeros(m).astype(np.int64)
    path[m-1] = np.argmax(V_last)

    for j in range(m-2, -1, -1):
        path[j] = P[j+1, path[j+1]]

    return(path)

# Diploid Li and Stephens

def rand_for_testing_dip(n, m, mean_r=1e-5, mean_mu=1e-5):
    '''
    Create a simple random haplotype matrix with no LD, use this to define
    all possible pairs of haplotypes, G, a random unphased genotype vector 
    to use as an observation, set a recombination probability vector, and 
    random mutation vector for the emissions.
    '''

    # Set up random genetic data
    H = np.random.randint(2, size = (m,n))
    G = np.zeros((m, n, n))
    for i in range(m):
        G[i,:,:] = np.add.outer(H[i,:], H[i,:])

    # Recombination probability
    r = mean_r * np.ones(m) * ((np.random.rand(m) + 0.5)/2)
    r[0] = 0

    # Error probability
    mu = mean_mu * np.ones(m) * ((np.random.rand(m) + 0.5)/2)

    # New sequence
    s = np.random.randint(2, size = (1, m)) + np.random.randint(2, size = (1, m))

    e  = np.zeros((m,8))
    e[:,EQUAL_BOTH_HOM] = (1 - mu) **2
    e[:,UNEQUAL_BOTH_HOM] = mu**2
    e[:,BOTH_HET] = (1-mu)
    e[:,REF_HOM_OBS_HET] = 2 * mu * (1 - mu)
    e[:,REF_HET_OBS_HOM] = mu * (1 - mu)

    return H, G, s, r, mu, e

@nb.jit
def path_ll_dip(G, phased_path, s, e):
    '''
    Simple evaluation of the log likelihood of a path through the reference panel resulting
    in the observed sequence.
    '''
    
    index = (
        4*np.int64(np.equal(G[0, phased_path[0][0], phased_path[1][0]], s[0,0])) +
        2*np.int64(G[0, phased_path[0][0], phased_path[1][0]] == 1) +
        np.int64(s[0,0] == 1)
        )
    log_prob_path = np.log10(1/(n**2) * e[0,index])
    old_phase = np.array([phased_path[0][0], phased_path[1][0]])
    r_n = r/n

    for l in range(1,m):

        index = (
            4*np.int64(np.equal(G[l, phased_path[0][l], phased_path[1][l]], s[0,l])) +
            2*np.int64(G[l, phased_path[0][l], phased_path[1][l]] == 1) +
            np.int64(s[0,l] == 1)
            )

        current_phase = np.array([phased_path[0][l], phased_path[1][l]])
        phase_diff = np.sum(~np.equal(current_phase, old_phase))
        
        if phase_diff == 0:
            log_prob_path += np.log10((1 - r[l])**2 + 2*(r_n[l]*(1 - r[l])) + r_n[l]**2)
        elif phase_diff == 1:
            log_prob_path += np.log10(r_n[l]*(1 - r[l]) + r_n[l]**2)
        else:
            log_prob_path += np.log10(r_n[l]**2)

        log_prob_path += np.log10(e[l,index])
        old_phase = current_phase

    return log_prob_path

def forwards_ls_dip(n, m, G, s, e):
    '''
    Simple matrix based method for diploid LS forward algorithm using numpy vectorisation.
    '''

    # Initialise the forward tensor
    F = np.zeros((m,n,n))
    F[0,:,:] = 1/(n**2)
    index = (
        4*np.equal(G[0,:,:], s[0,0]).astype(np.int64) + 
        2*(G[0,:,:] == 1).astype(np.int64) + 
        np.int64(s[0,0] == 1)
        )
    F[0,:,:] *= e[0, index]
    c = np.ones(m)
    r_n = r/n

    # Forwards
    for l in range(1,m):
        
        index = (
            4*np.equal(G[l,:,:], s[0,l]).astype(np.int64) + 
            2*(G[l,:,:] == 1).astype(np.int64) + 
            np.int64(s[0,l] == 1)
            )

        # No change in both
        F[l,:,:] = (1 - r[l])**2 * F[l-1,:,:]
        
        # Both change
        F[l,:,:] += (r_n[l])**2 * np.sum(F[l-1,:,:])
        
        # One changes
        # sum_j1 = np.tile(np.sum(F[l-1,:,:], 0, keepdims=True), (n,1))
        sum_j1 = np_sum(F[l-1,:,:], 0).repeat(n).reshape((-1, n)).T
        # sum_j2 = np.tile(np.sum(F[l-1,:,:], 1, keepdims=True), (1,n))
        sum_j2 = np_sum(F[l-1,:,:], 1).repeat(n).reshape((-1, n))
        F[l,:,:] += ((1 - r[l]) * r_n[l]) * (sum_j1 + sum_j2)
        
        # Emission
        F[l,:,:] *= e[l, index]
        c[l] = np.sum(F[l,:,:])
        F[l,:,:] *= 1/c[l]

    ll = np.sum(np.log10(c))
    return F, c, ll

def backwards_ls_dip(n, m, G, s, e, c):
    '''
    Simple matrix based method for diploid LS backward algorithm using numpy vectorisation.
    '''

    # Initialise the backward tensor
    B = np.zeros((m,n,n))

    # Initialise
    B[m-1,:,:] = 1
    r_n = r/n

    # Backwards
    for l in range(m-2, -1, -1):
        
        index = (
            4*np.equal(G[l+1,:,:], s[0,l+1]).astype(np.int64) +
            2*(G[l+1,:,:] == 1).astype(np.int64) +
            (s[0,l+1] == 1).astype(np.int64)
            )

        # No change in both
        B[l,:,:] = r_n[l+1]**2 * np.sum(e[l+1, index.reshape((n,n))] * B[l+1,:,:])
        
        # Both change
        B[l,:,:] += (1-r[l+1])**2 * B[l+1,:,:] * e[l+1, index.reshape((n,n))]
        
        # One changes
        # sum_j1 = np.tile(np.sum(B[l+1,:,:] * e[l+1, index], 0, keepdims=True), (n,1))
        sum_j1 = np_sum(B[l+1,:,:], 0).repeat(n).reshape((-1, n)).T
        # sum_j2 = np.tile(np.sum(B[l+1,:,:] * e[l+1, index], 1, keepdims=True), (1,n))
        sum_j2 = np_sum(B[l+1,:,:], 1).repeat(n).reshape((-1, n))
        B[l,:,:] += ((1 - r[l+1]) * r_n[l+1]) * (sum_j1 + sum_j2)
        B[l,:,:] *= 1/c[l+1]

    return B

def forward_ls_dip_starting_point(n, m, G, s, e):
    '''
    Unbelievably naive implementation of LS diploid forwards. Just to get something down
    that works.
    '''

    # Initialise the forward tensor
    F = np.zeros((m,n,n))
    F[0,:,:] = 1/(n**2)
    index = (
        4*np.equal(G[0,:,:], s[0,0]).astype(np.int64) + 
        2*(G[0,:,:] == 1).astype(np.int64) + 
        np.int64(s[0,0] == 1)
        )
    F[0,:,:] *= e[0, index]
    r_n = r/n

    for l in range(1,m):

        # Determine the various components
        F_no_change = np.zeros((n,n))
        F_j1_change = np.zeros(n)
        F_j2_change = np.zeros(n)
        F_both_change = 0

        for j1 in range(n):
            for j2 in range(n):
                F_no_change[j1, j2] = (1-r[l])**2 * F[l-1, j1, j2]

        for j1 in range(n):
            for j2 in range(n):
                F_both_change += r_n[l]**2 * F[l-1, j1, j2]

        for j1 in range(n):
            for j2 in range(n): # This is the variable to sum over - it changes
                F_j2_change[j1] += (1 - r[l]) * r_n[l] * F[l-1, j1, j2]

        for j2 in range(n):
            for j1 in range(n): # This is the variable to sum over - it changes
                F_j1_change[j2] += (1 - r[l]) * r_n[l] * F[l-1, j1, j2]

        F[l,:,:] = F_both_change

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
                if s[0,l] == 1:
                    # OBS is het
                    if G[l, j1, j2] == 1: # REF is het
                        F[l, j1, j2] *= e[l,BOTH_HET]
                    else: # REF is hom
                        F[l, j1, j2] *= e[l,REF_HOM_OBS_HET]
                else:
                    # OBS is hom
                    if G[l, j1, j2] == 1: # REF is het
                        F[l, j1, j2] *= e[l,REF_HET_OBS_HOM]
                    else: # REF is hom
                        if G[l, j1, j2] == s[0,l]: # Equal
                            F[l, j1, j2] *= e[l,EQUAL_BOTH_HOM]
                        else: # Unequal
                            F[l, j1, j2] *= e[l,UNEQUAL_BOTH_HOM]

    ll = np.log10(np.sum(F[l,:,:]))

    return F, ll

def backward_ls_dip_starting_point(n, m, G, s, e):
    '''
    Unbelievably naive implementation of LS diploid backwards. Just to get something down
    that works.
    '''

    # Backwards
    B = np.zeros((m,n,n))

    # Initialise
    B[m-1, :, :] = 1
    r_n = r/n

    for l in range(m-2, -1, -1):

        # Determine the various components
        B_no_change = np.zeros((n,n))
        B_j1_change = np.zeros(n)
        B_j2_change = np.zeros(n)
        B_both_change = 0

        # Evaluate the emission matrix at this site, for all pairs
        e_tmp = np.zeros((n,n))
        for j1 in range(n):
            for j2 in range(n):
                # What is the emission?
                if s[0,l+1] == 1:
                    # OBS is het
                    if G[l+1, j1, j2] == 1: # REF is het
                        e_tmp[j1, j2] = e[l+1, BOTH_HET]
                    else: # REF is hom
                        e_tmp[j1, j2] = e[l+1, REF_HOM_OBS_HET]
                else:
                    # OBS is hom
                    if G[l+1, j1, j2] == 1: # REF is het
                        e_tmp[j1, j2] = e[l+1,REF_HET_OBS_HOM]
                    else: # REF is hom
                        if G[l+1, j1, j2] == s[0,l+1]: # Equal
                            e_tmp[j1, j2] = e[l+1,EQUAL_BOTH_HOM]
                        else: # Unequal
                            e_tmp[j1, j2] = e[l+1,UNEQUAL_BOTH_HOM]

        for j1 in range(n):
            for j2 in range(n):
                B_no_change[j1, j2] = (1-r[l+1])**2 * B[l+1,j1,j2] * e_tmp[j1, j2]

        for j1 in range(n):
            for j2 in range(n):
                B_both_change += r_n[l+1]**2 * e_tmp[j1, j2] * B[l+1,j1,j2]

        for j1 in range(n):
            for j2 in range(n): # This is the variable to sum over - it changes
                B_j2_change[j1] += (1 - r[l+1]) * r_n[l+1] * B[l+1,j1,j2] * e_tmp[j1, j2]

        for j2 in range(n):
            for j1 in range(n): # This is the variable to sum over - it changes
                B_j1_change[j2] += (1 - r[l+1]) * r_n[l+1] * B[l+1,j1,j2] * e_tmp[j1, j2]

        B[l,:,:] = B_both_change

        for j1 in range(n):
            B[l, j1, :] += B_j2_change

        for j2 in range(n):
            B[l, :, j2] += B_j1_change

        for j1 in range(n):
            for j2 in range(n):
                B[l, j1, j2] += B_no_change[j1, j2]

    return B

def forward_ls_dip_loop(n, m, G, s, e):
    '''
    LS diploid forwards with lots of loops.
    '''

    # Initialise the forward tensor
    F = np.zeros((m,n,n))
    F[0,:,:] = 1/(n**2)
    index = (
        4*np.equal(G[0,:,:], s[0,0]).astype(np.int64) + 
        2*(G[0,:,:] == 1).astype(np.int64) + 
        np.int64(s[0,0] == 1)
        )
    F[0,:,:] *= e[0, index]
    r_n = r/n

    for l in range(1,m):

        # Determine the various components
        F_no_change = np.zeros((n,n))
        F_j1_change = np.zeros(n)
        F_j2_change = np.zeros(n)
        F_both_change = 0

        for j1 in range(n):
            for j2 in range(n):
                F_no_change[j1, j2] = (1-r[l])**2 * F[l-1,j1,j2]
                F_j1_change[j1] += (1 - r[l]) * r_n[l] * F[l-1,j2,j1]
                F_j2_change[j1] += (1 - r[l]) * r_n[l] * F[l-1,j1,j2]
                F_both_change += r_n[l]**2 * F[l-1,j1,j2]     

        F[l,:,:] = F_both_change  

        for j1 in range(n):
            F[l, j1, :] += F_j2_change
            F[l, :, j1] += F_j1_change
            for j2 in range(n):
                F[l, j1, j2] += F_no_change[j1, j2]

        for j1 in range(n):
            for j2 in range(n):
                # What is the emission?
                if s[0,l] == 1:
                    # OBS is het
                    if G[l, j1, j2] == 1: # REF is het
                        F[l, j1, j2] *= e[l, BOTH_HET]
                    else: # REF is hom
                        F[l, j1, j2] *= e[l, REF_HOM_OBS_HET]
                else:
                    # OBS is hom
                    if G[l, j1, j2] == 1: # REF is het
                        F[l, j1, j2] *= e[l, REF_HET_OBS_HOM]
                    else: # REF is hom
                        if G[l, j1, j2] == s[0,l]: # Equal
                            F[l, j1, j2] *= e[l, EQUAL_BOTH_HOM]
                        else: # Unequal
                            F[l, j1, j2] *= e[l, UNEQUAL_BOTH_HOM]

    ll = np.log10(np.sum(F[l,:,:]))
    return F, ll

def backward_ls_dip_loop(n, m, G, s, e):
    '''
    LS diploid backwards with lots of loops.
    '''

    # Initialise the backward tensor
    B = np.zeros((m,n,n))
    B[m-1, :, :] = 1
    r_n = r/n
    
    for l in range(m-2, -1, -1):

        # Determine the various components
        B_no_change = np.zeros((n,n))
        B_j1_change = np.zeros(n)
        B_j2_change = np.zeros(n)
        B_both_change = 0

        # Evaluate the emission matrix at this site, for all pairs
        e_tmp = np.zeros((n,n))
        for j1 in range(n):
            for j2 in range(n):
                # What is the emission?
                if s[0,l+1] == 1:
                    # OBS is het
                    if G[l+1, j1, j2] == 1: # REF is het
                        e_tmp[j1, j2] = e[l+1, BOTH_HET]
                    else: # REF is hom
                        e_tmp[j1, j2] = e[l+1, REF_HOM_OBS_HET]
                else:
                    # OBS is hom
                    if G[l+1, j1, j2] == 1: # REF is het
                        e_tmp[j1, j2] = e[l+1, REF_HET_OBS_HOM]
                    else: # REF is hom
                        if G[l+1, j1, j2] == s[0,l+1]: # Equal
                            e_tmp[j1, j2] = e[l+1, EQUAL_BOTH_HOM]
                        else: # Unequal
                            e_tmp[j1, j2] = e[l+1, UNEQUAL_BOTH_HOM]

        for j1 in range(n):
            for j2 in range(n):
                B_no_change[j1, j2] = (1-r[l+1])**2 * B[l+1,j1,j2] * e_tmp[j1, j2]
                B_j2_change[j1] += (1 - r[l+1]) * r_n[l+1] * B[l+1,j1,j2] * e_tmp[j1, j2]
                B_j1_change[j1] += (1 - r[l+1]) * r_n[l+1] * B[l+1,j2,j1] * e_tmp[j2, j1]
                B_both_change += r_n[l+1]**2 * e_tmp[j1, j2] * B[l+1,j1,j2]

        B[l,:,:] = B_both_change

        for j1 in range(n):
            B[l, j1, :] += B_j2_change
            B[l, :, j1] += B_j1_change
            for j2 in range(n):
                B[l, j1, j2] += B_no_change[j1, j2]

    return B

def forwards_viterbi_dip_naive(n, m, G, s, e, r):
    
    # Initialise
    V = np.zeros((m, n, n))
    P = np.zeros((m, n, n)).astype(np.int64)
    c = np.ones(m)
    index = (
        4*np.equal(G[0,:,:], s[0,0]).astype(np.int64) +
        2*(G[0,:,:] == 1).astype(np.int64) +
        np.int64(s[0,0] == 1)
        )
    V[0,:,:] = 1/(n**2) * e[0,index]
    r_n = r/n

    for l in range(1,m):
        index = (
            4*np.equal(G[l,:,:], s[0,l]).astype(np.int64) +
            2*(G[l,:,:] == 1).astype(np.int64) +
            np.int64(s[0,l] == 1)
            )
        for j1 in range(n):
            for j2 in range(n):
                # Get the vector to maximise over
                v = np.zeros((n,n))
                for k1 in range(n):
                    for k2 in range(n):
                        v[k1, k2] = V[l-1,k1, k2]
                        if ((k1 == j1) and (k2 == j2)):
                            v[k1, k2] *= ((1 - r[l])**2 + 2*(1-r[l]) * r_n[l] + r_n[l]**2)
                        elif ((k1 == j1) or (k2 == j2)):
                            v[k1, k2] *= (r_n[l] * (1 - r[l]) + r_n[l]**2)
                        else:
                            v[k1, k2] *= r_n[l]**2
                V[l,j1,j2] = np.amax(v) * e[l, index[j1, j2]]
                P[l,j1,j2] = np.argmax(v)
        c[l] = np.amax(V[l,:,:])
        V[l,:,:] *= 1/c[l]

    ll = np.sum(np.log10(c))
    
    return V, P, ll

def forwards_viterbi_dip_naive_low_mem(n, m, G, s, e, r):

    # Initialise
    V = np.zeros((n,n))
    P = np.zeros((m,n,n)).astype(np.int64)
    c = np.ones(m)
    index = (
        4*np.equal(G[0,:,:], s[0,0]).astype(np.int64) +
        2*(G[0,:,:] == 1).astype(np.int64) +
        np.int64(s[0,0] == 1)
        )
    V_previous = 1/(n**2) * e[0,index]
    r_n = r/n

    # Take a look at Haploid Viterbi implementation in Jeromes code and see if we can pinch some ideas.
    # Diploid Viterbi, with smaller memory footprint.
    for l in range(1,m):
        index = (
            4*np.equal(G[l,:,:], s[0,l]).astype(np.int64) +
            2*(G[l,:,:] == 1).astype(np.int64) +
            np.int64(s[0,l] == 1)
            )
        for j1 in range(n):
            for j2 in range(n):
                # Get the vector to maximise over
                v = np.zeros((n,n))
                for k1 in range(n):
                    for k2 in range(n):
                        v[k1, k2] = V_previous[k1, k2]
                        if ((k1 == j1) and (k2 == j2)):
                            v[k1, k2] *= ((1 - r[l])**2 + 2*(1-r[l]) * r_n[l] + r_n[l]**2)
                        elif ((k1 == j1) or (k2 == j2)):
                            v[k1, k2] *= (r_n[l] * (1 - r[l]) + r_n[l]**2)
                        else:
                            v[k1, k2] *= r_n[l]**2
                V[j1,j2] = np.amax(v) * e[l,index[j1, j2]]
                P[l,j1,j2] = np.argmax(v)
        c[l] = np.amax(V)
        V_previous = np.copy(V) / c[l]

    ll = np.sum(np.log10(c))

    return V, P, ll

def forwards_viterbi_dip_low_mem(n, m, G, s, e, r):

    # Initialise
    V = np.zeros((n, n))
    index = (
        4*np.equal(G[0,:,:], s[0,0]).astype(np.int64) +
        2*(G[0,:,:] == 1).astype(np.int64) +
        np.int64(s[0,0] == 1)
        )
    V_previous = 1/(n**2) * e[0,index]
    c = np.ones(m)
    r_n = r/n

    # Diploid Viterbi, with smaller memory footprint, rescaling, and using the structure of the HMM.
    for l in range(1,m):
        
        index = (
            4*np.equal(G[l,:,:], s[0,l]).astype(np.int64) +
            2*(G[l,:,:] == 1).astype(np.int64) +
            np.int64(s[0,l] == 1)
            )
        
        c[l] = np.amax(V_previous)
        argmax = np.argmax(V_previous)

        V_previous *= 1/c[l]
        V_rowcol_max = np_amax(V_previous, 0)
        arg_rowcol_max = np_argmax(V_previous, 0)

        no_switch = (1 - r[l])**2 + 2*(r_n[l]*(1 - r[l])) + r_n[l]**2
        single_switch = r_n[l]*(1 - r[l]) + r_n[l]**2
        double_switch = r_n[l]**2

        j1_j2 = 0

        for j1 in range(n):
            for j2 in range(n):

                V_single_switch = max(V_rowcol_max[j1], V_rowcol_max[j2])
                P_single_switch = np.argmax(np.array([V_rowcol_max[j1], V_rowcol_max[j2]]))

                if P_single_switch == 0:
                    template_single_switch = j1*n + arg_rowcol_max[j1]
                else:
                    template_single_switch = arg_rowcol_max[j2]*n + j2
                
                V[j1,j2] = V_previous[j1,j2] * no_switch # No switch in either
                P[l, j1, j2] = j1_j2

                # Single or double switch?
                single_switch_tmp = single_switch * V_single_switch
                if (single_switch_tmp > double_switch):
                    # Then single switch is the alternative
                    if (V[j1,j2] < single_switch * V_single_switch):
                        V[j1,j2] = single_switch * V_single_switch
                        P[l, j1, j2] = template_single_switch
                else:
                    # Double switch is the alternative
                    if V[j1, j2] < double_switch:
                        V[j1, j2] = double_switch
                        P[l, j1, j2] = argmax

                V[j1,j2] *= e[l, index[j1, j2]]
                j1_j2 += 1
        V_previous = np.copy(V)

    ll = np.sum(np.log10(c)) + np.log10(np.amax(V))

    return V, P, ll

def forwards_viterbi_dip_naive_vec(n, m, G, s, e, r):
    
    # Initialise
    V = np.zeros((m,n,n))
    P = np.zeros((m,n,n)).astype(np.int64)
    c = np.ones(m)
    index = (
        4*np.equal(G[0,:,:], s[0,0]).astype(np.int64) +
        2*(G[0,:,:] == 1).astype(np.int64) +
        np.int64(s[0,0] == 1)
        )
    V[0,:,:] = 1/(n**2) * e[0,index]
    r_n = r/n

    # Jumped the gun - vectorising.
    for l in range(1,m):
        
        index = (
            4*np.equal(G[l,:,:], s[0,l]).astype(np.int64) +
            2*(G[l,:,:] == 1).astype(np.int64) +
            np.int64(s[0,l] == 1)
            )

        for j1 in range(n):
            for j2 in range(n):
                v = (r_n[l]**2) * np.ones((n,n))
                v[j1,j2] += (1-r[l])**2
                v[j1, :] += (r_n[l] * (1 - r[l]))
                v[:, j2] += (r_n[l] * (1 - r[l]))
                v *= V[l-1,:,:]
                V[l,j1,j2] = np.amax(v) * e[l,index[j1, j2]]
                P[l,j1,j2] = np.argmax(v)
        
        c[l] = np.amax(V[l,:,:])
        V[l,:,:] *= 1/c[l]

    ll = np.sum(np.log10(c))
    
    return V, P, ll

def forwards_viterbi_dip_naive_full_vec(n, m, G, s, e, r):

    char_both = np.eye(n*n).ravel().reshape((n,n,n,n))
    char_col = np.tile(np.sum(np.eye(n*n).reshape((n,n,n,n)), 3), (n,1,1,1))  
    char_row = np.copy(char_col).T
    rows, cols = np.ogrid[:n, :n]

    # Initialise
    V = np.zeros((m,n,n))
    P = np.zeros((m,n,n)).astype(np.int64)
    c = np.ones(m)
    index = (
        4*np.equal(G[0,:,:], s[0,0]).astype(np.int64) +
        2*(G[0,:,:] == 1).astype(np.int64) +
        np.int64(s[0,0] == 1)
        )
    V[0,:,:] = 1/(n**2) * e[0,index]
    r_n = r/n

    for l in range(1,m):
        index = (
            4*np.equal(G[l,:,:], s[0,l]).astype(np.int64) +
            2*(G[l,:,:] == 1).astype(np.int64) +
            np.int64(s[0,l] == 1)
            )
        v = (r_n[l]**2) + (1-r[l])**2 * char_both + (r_n[l] * (1 - r[l])) * (char_col +  char_row)
        v *= V[l-1,:,:]
        P[l,:,:] = np.argmax(v.reshape(n,n,-1), 2) # Have to flatten to use argmax
        V[l,:,:] = v.reshape(n,n,-1)[rows, cols, P[l,:,:]] * e[l,index]
        c[l] = np.amax(V[l,:,:])
        V[l,:,:] *= 1/c[l]

    ll = np.sum(np.log10(c))

    return V, P, ll

def backwards_viterbi_dip(n, m, V_last, P):
    '''
    Backwards pass to determine the most likely path
    '''

    # Initialisation
    path = np.zeros(m).astype(np.int64)
    path[m-1] = np.argmax(V_last)

    # Backtrace
    for j in range(m-2, -1, -1):
        path[j] = P[j+1,:,:].ravel()[path[j+1]]

    return path

def get_phased_path(n, path):
    # Obtain the phased path
    return np.unravel_index(path, (n,n))

n = 10
m = 100

H, s, r, mu, e = rand_for_testing_hap(n, m)

tic = time.perf_counter()
F, c, ll = forwards_ls_hap(n, m, H, s, e, r)
B = backwards_ls_hap(n, m, H, s, e, c, r)
ll = np.sum(np.log10(c))
print(f"log-likelihood: {ll}")
toc = time.perf_counter()
print(f"forwards backwards in {toc - tic:0.4f} seconds")

tic = time.perf_counter()
V, P, ll = forwards_viterbi_hap_naive(n, m, H, s, e, r)
path = backwards_viterbi_hap(m, V[m-1, :], P)
print(f"log-likelihood: {ll}")
toc = time.perf_counter()
print(f"naive viterbi in in {toc - tic:0.4f} seconds")

tic = time.perf_counter()
V, P, ll = forwards_viterbi_hap_naive_low_mem(n, m, H, s, e, r)
path = backwards_viterbi_hap(m, V, P)
print(f"log-likelihood: {ll}")
toc = time.perf_counter()
print(f"naive low mem viterbi in {toc - tic:0.4f} seconds")

tic = time.perf_counter()
V, P, ll = forwards_viterbi_hap_naive_low_mem_rescaling(n, m, H, s, e, r)
path = backwards_viterbi_hap(m, V, P)
print(f"log-likelihood: {ll}")
toc = time.perf_counter()
print(f"naive low mem rescaling viterbi in {toc - tic:0.4f} seconds")

tic = time.perf_counter()
V, P, ll = forwards_viterbi_hap_lower_mem_rescaling(n, m, H, s, e, r)
path = backwards_viterbi_hap(m, V, P)
print(f"log-likelihood: {ll}")
toc = time.perf_counter()
print(f"final viterbi in {toc - tic:0.4f} seconds")

tic = time.perf_counter()
V, P, ll = forwards_viterbi_hap_naive_vec(n, m, H, s, e, r)
path = backwards_viterbi_hap(m, V[m-1,:], P)
print(f"log-likelihood: {ll}")
toc = time.perf_counter()
print(f"naive vector viterbi in {toc - tic:0.4f} seconds")

# Diploid Li and Stephens
# Yes, I know there's a factor of two that we can squeeze out of this.

EQUAL_BOTH_HOM = 4
UNEQUAL_BOTH_HOM = 0
BOTH_HET = 7
REF_HOM_OBS_HET = 1
REF_HET_OBS_HOM= 2

H, G, s, r, mu, e = rand_for_testing_dip(n, m)

tic = time.perf_counter()
F, c, ll = forwards_ls_dip(n, m, G, s, e)
print(f"log-likelihood: {ll}")
B = backwards_ls_dip(n, m, G, s, e, c)
toc = time.perf_counter()
print(f"forwards backwards in {toc - tic:0.4f} seconds")

tic = time.perf_counter()
F, ll = forward_ls_dip_starting_point(n, m, G, s, e)
print(f"log-likelihood: {ll}")
B = backward_ls_dip_starting_point(n, m, G, s, e)
toc = time.perf_counter()
print(f"forwards backwards in {toc - tic:0.4f} seconds")

tic = time.perf_counter()
F, ll = forward_ls_dip_loop(n, m, G, s, e)
print(f"log-likelihood: {ll}")
B = backward_ls_dip_loop(n, m, G, s, e)
toc = time.perf_counter()
print(f"forwards backwards in {toc - tic:0.4f} seconds")

tic = time.perf_counter()
V, P, ll = forwards_viterbi_dip_naive(n, m, G, s, e, r)
print(f"log-likelihood: {ll}")
path = backwards_viterbi_dip(n, m, V[m-1,:,:], P)
phased_path = get_phased_path(n, path)
toc = time.perf_counter()
print(f"viterbi in {toc - tic:0.4f} seconds")
path_ll = path_ll_dip(G, phased_path, s, e)
print(path_ll)

tic = time.perf_counter()
V, P, ll = forwards_viterbi_dip_naive_low_mem(n, m, G, s, e, r)
print(f"log-likelihood: {ll}")
path = backwards_viterbi_dip(n, m, V, P)
phased_path = get_phased_path(n, path)
toc = time.perf_counter()
print(f"viterbi in {toc - tic:0.4f} seconds")
path_ll = path_ll_dip(G, phased_path, s, e)
print(path_ll)

tic = time.perf_counter()
V, P, ll = forwards_viterbi_dip_low_mem(n, m, G, s, e, r)
print(f"log-likelihood: {ll}")
path = backwards_viterbi_dip(n, m, V, P)
phased_path = get_phased_path(n, path)
toc = time.perf_counter()
print(f"viterbi in {toc - tic:0.4f} seconds")
path_ll = path_ll_dip(G, phased_path, s, e)
print(path_ll)

tic = time.perf_counter()
V, P, ll = forwards_viterbi_dip_naive_vec(n, m, G, s, e, r)
print(f"log-likelihood: {ll}")
path = backwards_viterbi_dip(n, m, V[m-1,:,:], P)
toc = time.perf_counter()
print(f"viterbi in {toc - tic:0.4f} seconds")
path_ll = path_ll_dip(G, phased_path, s, e)
print(path_ll)

tic = time.perf_counter()
V, P, ll = forwards_viterbi_dip_naive_full_vec(n, m, G, s, e, r)
print(f"log-likelihood: {ll}")
path = backwards_viterbi_dip(n, m, V[m-1,:,:], P)
toc = time.perf_counter()
print(f"viterbi in {toc - tic:0.4f} seconds")
path_ll = path_ll_dip(G, phased_path, s, e)
print(path_ll)

class LiStephensBase:
    """
    Superclass of Li and Stephens tests.
    """

    def assertCompressedMatricesEqual(self, cm1, cm2):
        """
        Checks that the specified compressed matrices contain the same data.
        """
        A1 = cm1.decode()
        A2 = cm2.decode()
        assert np.allclose(A1, A2)
        assert A1.shape == A2.shape
        assert cm1.num_sites == cm2.num_sites
        nf1 = cm1.normalisation_factor
        nf2 = cm1.normalisation_factor
        assert np.allclose(nf1, nf2)
        assert nf1.shape == nf2.shape
        # It seems that we can't rely on the number of transitions in the two
        # implementations being equal, which seems odd given that we should
        # be doing things identically. Still, once the decoded matrices are the
        # same then it seems highly likely to be correct.

        # if not np.array_equal(cm1.num_transitions, cm2.num_transitions):
        #     print()
        #     print(cm1.num_transitions)
        #     print(cm2.num_transitions)
        # self.assertTrue(np.array_equal(cm1.num_transitions, cm2.num_transitions))
        # for j in range(cm1.num_sites):
        #     s1 = dict(cm1.get_site(j))
        #     s2 = dict(cm2.get_site(j))
        #     self.assertEqual(set(s1.keys()), set(s2.keys()))
        #     for key in s1.keys():
        #         self.assertAlmostEqual(s1[key], s2[key])

    def example_haplotypes(self, ts, alleles, num_random=10, seed=2):
        rng = np.random.RandomState(seed)
        H = ts.genotype_matrix(alleles=alleles).T
        haplotypes = [H[0], H[-1]]
        for _ in range(num_random):
            # Choose a random path through H
            p = rng.randint(0, ts.num_samples, ts.num_sites)
            h = H[p, np.arange(ts.num_sites)]
            haplotypes.append(h)
        h = H[0].copy()
        h[-1] = tskit.MISSING_DATA
        haplotypes.append(h)
        h = H[0].copy()
        h[ts.num_sites // 2] = tskit.MISSING_DATA
        haplotypes.append(h)
        # All missing is OK tool
        h = H[0].copy()
        h[:] = tskit.MISSING_DATA
        haplotypes.append(h)
        return haplotypes

    def example_parameters(self, ts, alleles, seed=1):
        """
        Returns an iterator over combinations of haplotype, recombination and mutation
        rates.
        """
        rng = np.random.RandomState(seed)
        haplotypes = self.example_haplotypes(ts, alleles, seed=seed)

        # This is the exact matching limit.
        rho = np.zeros(ts.num_sites) + 0.01
        mu = np.zeros(ts.num_sites)
        rho[0] = 0
        for h in haplotypes:
            yield h, rho, mu

        # Here we have equal mutation and recombination
        rho = np.zeros(ts.num_sites) + 0.01
        mu = np.zeros(ts.num_sites) + 0.01
        rho[0] = 0
        for h in haplotypes:
            yield h, rho, mu

        # Mixture of random and extremes
        rhos = [
            np.zeros(ts.num_sites) + 0.999,
            np.zeros(ts.num_sites) + 1e-6,
            rng.uniform(0, 1, ts.num_sites),
        ]
        # mu can't be more than 1 / 3 if we have 4 alleles
        mus = [
            np.zeros(ts.num_sites) + 0.33,
            np.zeros(ts.num_sites) + 1e-6,
            rng.uniform(0, 0.33, ts.num_sites),
        ]
        for h, rho, mu in itertools.product(haplotypes, rhos, mus):
            rho[0] = 0
            yield h, rho, mu

    def assertAllClose(self, A, B):
        assert np.allclose(A, B)

    def test_simple_n_4_no_recombination(self):
        ts = msprime.simulate(4, recombination_rate=0, mutation_rate=0.5, random_seed=1)
        assert ts.num_sites > 3
        self.verify(ts)

    def test_simple_n_3(self):
        ts = msprime.simulate(3, recombination_rate=2, mutation_rate=7, random_seed=2)
        assert ts.num_sites > 5
        self.verify(ts)

    def test_simple_n_7(self):
        ts = msprime.simulate(7, recombination_rate=2, mutation_rate=5, random_seed=2)
        assert ts.num_sites > 5
        self.verify(ts)

    def test_simple_n_8_high_recombination(self):
        ts = msprime.simulate(8, recombination_rate=20, mutation_rate=5, random_seed=2)
        assert ts.num_trees > 15
        assert ts.num_sites > 5
        self.verify(ts)

    def test_simple_n_15(self):
        ts = msprime.simulate(15, recombination_rate=2, mutation_rate=5, random_seed=2)
        assert ts.num_sites > 5
        self.verify(ts)

    def test_jukes_cantor_n_3(self):
        ts = msprime.simulate(3, mutation_rate=2, random_seed=2)
        ts = tsutil.jukes_cantor(ts, num_sites=10, mu=10, seed=4)
        self.verify(ts, tskit.ALLELES_ACGT)

    def test_jukes_cantor_n_8_high_recombination(self):
        ts = msprime.simulate(8, recombination_rate=20, random_seed=2)
        ts = tsutil.jukes_cantor(ts, num_sites=20, mu=5, seed=4)
        self.verify(ts, tskit.ALLELES_ACGT)

    def test_jukes_cantor_n_15(self):
        ts = msprime.simulate(15, mutation_rate=2, random_seed=2)
        ts = tsutil.jukes_cantor(ts, num_sites=10, mu=0.1, seed=10)
        self.verify(ts, tskit.ALLELES_ACGT)

    def test_jukes_cantor_balanced_ternary(self):
        ts = tskit.Tree.generate_balanced(27, arity=3).tree_sequence
        ts = tsutil.jukes_cantor(ts, num_sites=10, mu=0.1, seed=10)
        self.verify(ts, tskit.ALLELES_ACGT)

    @pytest.mark.skip(reason="Not supporting internal samples yet")
    def test_ancestors_n_3(self):
        ts = msprime.simulate(3, recombination_rate=2, mutation_rate=7, random_seed=2)
        assert ts.num_sites > 5
        tables = ts.dump_tables()
        print(tables.nodes)
        tables.nodes.flags = np.ones_like(tables.nodes.flags)
        print(tables.nodes)
        ts = tables.tree_sequence()
        self.verify(ts)


@pytest.mark.slow
class ForwardAlgorithmBase(LiStephensBase):
    """
    Base for forward algorithm tests.
    """


class TestNumpyMatrixMethod(ForwardAlgorithmBase):
    """
    Tests that we compute the same values from the numpy matrix method as
    the naive algorithm.
    """

    def verify(self, ts, alleles=tskit.ALLELES_01):
        G = ts.genotype_matrix(alleles=alleles)
        for h, rho, mu in self.example_parameters(ts, alleles):
            F1, S1 = ls_forward_matrix(h, alleles, G, rho, mu)
            F2, S2 = ls_forward_matrix_naive(h, alleles, G, rho, mu)
            self.assertAllClose(F1, F2)
            self.assertAllClose(S1, S2)


class ViterbiAlgorithmBase(LiStephensBase):
    """
    Base for viterbi algoritm tests.
    """


class TestExactMatchViterbi(ViterbiAlgorithmBase):
    def verify(self, ts, alleles=tskit.ALLELES_01):
        G = ts.genotype_matrix(alleles=alleles)
        H = G.T
        # print(H)
        rho = np.zeros(ts.num_sites) + 0.1
        mu = np.zeros(ts.num_sites)
        rho[0] = 0
        for h in H:
            p1 = ls_viterbi_naive(h, alleles, G, rho, mu)
            p2 = ls_viterbi_vectorised(h, alleles, G, rho, mu)
            cm1 = ls_viterbi_tree(h, alleles, ts, rho, mu, use_lib=True)
            p3 = cm1.traceback()
            cm2 = ls_viterbi_tree(h, alleles, ts, rho, mu, use_lib=False)
            p4 = cm1.traceback()
            self.assertCompressedMatricesEqual(cm1, cm2)

            assert len(np.unique(p1)) == 1
            assert len(np.unique(p2)) == 1
            assert len(np.unique(p3)) == 1
            assert len(np.unique(p4)) == 1
            m1 = H[p1, np.arange(H.shape[1])]
            assert np.array_equal(m1, h)
            m2 = H[p2, np.arange(H.shape[1])]
            assert np.array_equal(m2, h)
            m3 = H[p3, np.arange(H.shape[1])]
            assert np.array_equal(m3, h)
            m4 = H[p3, np.arange(H.shape[1])]
            assert np.array_equal(m4, h)


@pytest.mark.slow
class TestGeneralViterbi(ViterbiAlgorithmBase, unittest.TestCase):
    def verify(self, ts, alleles=tskit.ALLELES_01):
        # np.set_printoptions(linewidth=20000)
        # np.set_printoptions(threshold=20000000)
        G = ts.genotype_matrix(alleles=alleles)
        # m, n = G.shape
        for h, rho, mu in self.example_parameters(ts, alleles):
            # print("h = ", h)
            # print("rho=", rho)
            # print("mu = ", mu)
            p1 = ls_viterbi_vectorised(h, alleles, G, rho, mu)
            p2 = ls_viterbi_naive(h, alleles, G, rho, mu)
            cm1 = ls_viterbi_tree(h, alleles, ts, rho, mu, use_lib=True)
            p3 = cm1.traceback()
            cm2 = ls_viterbi_tree(h, alleles, ts, rho, mu, use_lib=False)
            p4 = cm1.traceback()
            self.assertCompressedMatricesEqual(cm1, cm2)
            # print()
            # m1 = H[p1, np.arange(m)]
            # m2 = H[p2, np.arange(m)]
            # m3 = H[p3, np.arange(m)]
            # count = np.unique(p1).shape[0]
            # print()
            # print("\tp1 = ", p1)
            # print("\tp2 = ", p2)
            # print("\tp3 = ", p3)
            # print("\tm1 = ", m1)
            # print("\tm2 = ", m2)
            # print("\t h = ", h)
            proba1 = ls_path_log_probability(h, p1, alleles, G, rho, mu)
            proba2 = ls_path_log_probability(h, p2, alleles, G, rho, mu)
            proba3 = ls_path_log_probability(h, p3, alleles, G, rho, mu)
            proba4 = ls_path_log_probability(h, p4, alleles, G, rho, mu)
            # print("\t P = ", proba1, proba2)
            self.assertAlmostEqual(proba1, proba2, places=6)
            self.assertAlmostEqual(proba1, proba3, places=6)
            self.assertAlmostEqual(proba1, proba4, places=6)


class TestMissingHaplotypes(LiStephensBase):
    def verify(self, ts, alleles=tskit.ALLELES_01):
        G = ts.genotype_matrix(alleles=alleles)
        H = G.T

        rho = np.zeros(ts.num_sites) + 0.1
        rho[0] = 0
        mu = np.zeros(ts.num_sites) + 0.001

        # When everything is missing data we should have no recombinations.
        h = H[0].copy()
        h[:] = tskit.MISSING_DATA
        path = ls_viterbi_vectorised(h, alleles, G, rho, mu)
        assert np.all(path == 0)
        cm = ls_viterbi_tree(h, alleles, ts, rho, mu, use_lib=True)
        # For the tree base algorithm it's not simple which particular sample
        # gets chosen.
        path = cm.traceback()
        assert len(set(path)) == 1

        # TODO Not clear what else we can check about missing data.


class TestForwardMatrixScaling(ForwardAlgorithmBase, unittest.TestCase):
    """
    Tests that we get the correct values from scaling version of the matrix
    algorithm works correctly.
    """

    def verify(self, ts, alleles=tskit.ALLELES_01):
        G = ts.genotype_matrix(alleles=alleles)
        computed_log_proba = False
        for h, rho, mu in self.example_parameters(ts, alleles):
            F_unscaled = ls_forward_matrix_unscaled(h, alleles, G, rho, mu)
            F, S = ls_forward_matrix(h, alleles, G, rho, mu)
            column = np.atleast_2d(np.cumprod(S)).T
            F_scaled = F * column
            self.assertAllClose(F_scaled, F_unscaled)
            log_proba1 = forward_matrix_log_proba(F, S)
            psum = np.sum(F_unscaled[-1])
            # If the computed probability is close to zero, there's no point in
            # computing.
            if psum > 1e-20:
                computed_log_proba = True
                log_proba2 = np.log(psum)
                self.assertAlmostEqual(log_proba1, log_proba2)
        assert computed_log_proba


class TestForwardTree(ForwardAlgorithmBase):
    """
    Tests that the tree algorithm computes the same forward matrix as the
    simple method.
    """

    def verify(self, ts, alleles=tskit.ALLELES_01):
        G = ts.genotype_matrix(alleles=alleles)
        for h, rho, mu in self.example_parameters(ts, alleles):
            F, S = ls_forward_matrix(h, alleles, G, rho, mu)
            cm1 = ls_forward_tree(h, alleles, ts, rho, mu, use_lib=True)
            cm2 = ls_forward_tree(h, alleles, ts, rho, mu, use_lib=False)
            self.assertCompressedMatricesEqual(cm1, cm2)
            Ft = cm1.decode()
            self.assertAllClose(S, cm1.normalisation_factor)
            self.assertAllClose(F, Ft)


class TestAllPaths(unittest.TestCase):
    """
    Tests that we compute the correct forward probablities if we sum over all
    possible paths through the genotype matrix.
    """

    def verify(self, G, h):
        m, n = G.shape
        rho = np.zeros(m) + 0.1
        mu = np.zeros(m) + 0.01
        rho[0] = 0
        proba = 0
        for path in itertools.product(range(n), repeat=m):
            proba += ls_path_probability(h, path, G, rho, mu)

        alleles = [["0", "1"] for _ in range(m)]
        F = ls_forward_matrix_unscaled(h, alleles, G, rho, mu)
        forward_proba = np.sum(F[-1])
        self.assertAlmostEqual(proba, forward_proba)

    def test_n3_m4(self):
        G = np.array(
            [
                # fmt: off
                [1, 0, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                # fmt: on
            ]
        )
        self.verify(G, [0, 0, 0, 0])
        self.verify(G, [1, 1, 1, 1])
        self.verify(G, [1, 1, 0, 0])

    def test_n4_m5(self):
        G = np.array(
            [
                # fmt: off
                [1, 0, 0, 0],
                [0, 0, 1, 1],
                [1, 0, 1, 1],
                [0, 1, 1, 0],
                # fmt: on
            ]
        )
        self.verify(G, [0, 0, 0, 0, 0])
        self.verify(G, [1, 1, 1, 1, 1])
        self.verify(G, [1, 1, 0, 0, 0])

    def test_n5_m5(self):
        G = np.zeros((5, 5), dtype=int)
        np.fill_diagonal(G, 1)
        self.verify(G, [0, 0, 0, 0, 0])
        self.verify(G, [1, 1, 1, 1, 1])
        self.verify(G, [1, 1, 0, 0, 0])


class TestBasicViterbi:
    """
    Very simple tests of the Viterbi algorithm.
    """

    def verify_exact_match(self, G, h, path):
        m, n = G.shape
        rho = np.zeros(m) + 1e-9
        mu = np.zeros(m)  # Set mu to zero exact match
        rho[0] = 0
        alleles = [["0", "1"] for _ in range(m)]
        path1 = ls_viterbi_naive(h, alleles, G, rho, mu)
        path2 = ls_viterbi_vectorised(h, alleles, G, rho, mu)
        assert list(path1) == path
        assert list(path2) == path

    def test_n2_m6_exact(self):
        G = np.array(
            [
                # fmt: off
                [1, 0],
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [0, 1],
                # fmt: on
            ]
        )
        self.verify_exact_match(G, [1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1])
        self.verify_exact_match(G, [0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0])
        self.verify_exact_match(G, [0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1])
        self.verify_exact_match(G, [0, 0, 0, 1, 1, 0], [1, 1, 1, 1, 1, 0])
        self.verify_exact_match(G, [0, 0, 0, 0, 1, 0], [1, 1, 1, 0, 1, 0])

    def test_n3_m6_exact(self):
        G = np.array(
            [
                # fmt: off
                [1, 0, 1],
                [1, 0, 0],
                [1, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 0],
                # fmt: on
            ]
        )
        self.verify_exact_match(G, [1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1])
        self.verify_exact_match(G, [0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0])
        self.verify_exact_match(G, [0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1])
        self.verify_exact_match(G, [1, 0, 1, 0, 1, 0], [2, 2, 2, 2, 2, 2])

    def test_n3_m6(self):
        G = np.array(
            [
                # fmt: off
                [1, 0, 1],
                [1, 0, 0],
                [1, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 0],
                # fmt: on
            ]
        )

        m, n = G.shape
        rho = np.zeros(m) + 1e-2
        mu = np.zeros(m)
        rho[0] = 0
        alleles = [["0", "1"] for _ in range(m)]
        h = np.ones(m, dtype=int)
        path1 = ls_viterbi_naive(h, alleles, G, rho, mu)

        # Add in mutation at a very low rate.
        mu[:] = 1e-8
        path2 = ls_viterbi_naive(h, alleles, G, rho, mu)
        path3 = ls_viterbi_vectorised(h, alleles, G, rho, mu)
        assert np.array_equal(path1, path2)
        assert np.array_equal(path2, path3)




