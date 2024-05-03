import numpy as np

from lshmm import jit


EQUAL_BOTH_HOM = 4
UNEQUAL_BOTH_HOM = 0
BOTH_HET = 7
REF_HOM_OBS_HET = 1
REF_HET_OBS_HOM = 2
MISSING_INDEX = 3

MISSING = -1


""" Helper functions. """


# https://github.com/numba/numba/issues/1269
@jit.numba_njit
def np_apply_along_axis(func1d, axis, arr):
    """Create Numpy-like functions for max, sum, etc."""
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


@jit.numba_njit
def np_amax(array, axis):
    """Numba implementation of Numpy-vectorised max."""
    return np_apply_along_axis(np.amax, axis, array)


@jit.numba_njit
def np_sum(array, axis):
    """Numba implementation of Numpy-vectorised sum."""
    return np_apply_along_axis(np.sum, axis, array)


@jit.numba_njit
def np_argmax(array, axis):
    """Numba implementation of Numpy-vectorised argmax."""
    return np_apply_along_axis(np.argmax, axis, array)


""" Functions used across different implementations of the LS HMM. """


def get_num_alleles(ref_panel, query):
    assert ref_panel.shape[0] == query.shape[1]
    num_sites = ref_panel.shape[0]
    allele_lists = []
    for i in range(num_sites):
        all_alleles = np.append(ref_panel[i, :], query[:, i])
        allele_lists.append(all_alleles)
    num_alleles = check_alleles(allele_lists, num_sites)
    return num_alleles


def check_alleles(alleles, num_sites):
    """
    Check a list of allele lists (or strings representing alleles) at m sites, and
    return a list of counts of distinct alleles at the m sites.

    If alleles is a list of strings, then each string represents distinct alleles
    at a site, and each character in a string represents a distinct allele.
    It is assumed that MISSING is not encoded in these strings.

    Note MISSING values in the allele lists are excluded from the counts.

    :param list alleles: A list of lists of alleles (or strings).
    :param int num_sites: Number of sites.
    :return: An array of counts of distinct alleles at each site.
    :rtype: numpy.ndarray
    """
    if len(alleles) != num_sites:
        err_msg = "Number of allele lists (or strings) is not equal to number of sites."
        raise ValueError(err_msg)
    # Process string encoding of distinct alleles.
    if isinstance(alleles[0], str):
        return np.int8([len(alleles) for _ in range(num_sites)])
    # Otherwise, process allele lists.
    exclusion_set = np.array([MISSING])
    num_alleles = np.zeros(num_sites, dtype=np.int32)
    for i in range(num_sites):
        uniq_alleles = np.unique(alleles[i])
        num_alleles[i] = np.sum(~np.isin(uniq_alleles, exclusion_set))
    return num_alleles


@jit.numba_njit
def get_index_in_emission_matrix_haploid(ref_allele, query_allele):
    is_allele_match = ref_allele == query_allele
    is_query_missing = query_allele == MISSING
    if is_allele_match or is_query_missing:
        return 1
    return 0


@jit.numba_njit
def get_index_in_emission_matrix_diploid(ref_allele, query_allele):
    if query_allele == MISSING:
        return MISSING_INDEX
    else:
        is_match = ref_allele == query_allele
        is_ref_one = ref_allele == 1
        is_query_one = query_allele == 1
        return 4 * is_match + 2 * is_ref_one + is_query_one


@jit.numba_njit
def get_index_in_emission_matrix_diploid_G(ref_G, query_allele, n):
    if query_allele == MISSING:
        return MISSING_INDEX * np.ones((n, n), dtype=np.int64)
    else:
        is_match = ref_G == query_allele
        is_ref_one = ref_G == 1
        is_query_one = query_allele == 1
        return 4 * is_match + 2 * is_ref_one + is_query_one


def get_emission_matrix_haploid(mu, num_sites, num_alleles, scale_mutation_rate):
    e = np.zeros((num_sites, 2))
    if isinstance(mu, float):
        mu = mu * np.ones(num_sites)
    if scale_mutation_rate:
        # Scale mutation based on the number of alleles,
        # so p_mutation is probability of mutation any given one of the alleles.
        # The overall mutation probability is then (n_alleles - 1) * p_mutation.
        e[:, 0] = mu - mu * np.equal(
            num_alleles, np.ones(num_sites)
        )  # Add boolean in case we're at an invariant site
        e[:, 1] = 1 - (num_alleles - 1) * mu
    else:
        # No scaling based on the number of alleles,
        # so p_mutation is the probability of mutation to anything
        # (summing over the states we can switch to).
        # This means that we must rescale the probability of mutation to
        # a different allele by the number of alleles at the site.
        for j in range(num_sites):
            if num_alleles[j] == 1:
                # In case we're at an invariant site
                e[j, 0] = 0
                e[j, 1] = 1
            else:
                e[j, 0] = mu[j] / (num_alleles[j] - 1)
                e[j, 1] = 1 - mu[j]
    return e


def get_emission_matrix_diploid(mu, num_sites):
    e = np.zeros((num_sites, 8))
    e[:, EQUAL_BOTH_HOM] = (1 - mu) ** 2
    e[:, UNEQUAL_BOTH_HOM] = mu**2
    e[:, BOTH_HET] = (1 - mu) ** 2 + mu**2
    e[:, REF_HOM_OBS_HET] = 2 * mu * (1 - mu)
    e[:, REF_HET_OBS_HOM] = mu * (1 - mu)
    e[:, MISSING_INDEX] = 1
    return e
