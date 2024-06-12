import numpy as np

from lshmm import jit


EQUAL_BOTH_HOM = 4
UNEQUAL_BOTH_HOM = 0
BOTH_HET = 7
REF_HOM_OBS_HET = 1
REF_HET_OBS_HOM = 2
MISSING_INDEX = 3

MISSING = -1
NONCOPY = -2


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


def convert_haplotypes_to_phased_genotypes(ref_panel):
    """
    Convert a set of haplotypes into a matrix of diploid genotypes encoded as allele dosages,
    and return the genotypes.

    It is assumed all sites are biallelic and alleles are encoded as ancestral/derived.
    The only allowable allele states are 0, 1, and NONCOPY (for partial ancestral haplotypes).
    TODO: Handle multiallelic sites.

    Allowable genotype values are 0, 1, 2, and NONCOPY. If either one haplotype is NONCOPY
    at a site, then the genotype at the site is assigned NONCOPY.

    The input reference haplotypes is of size (m, n), and the output genotypes is of size (m, n, n),
    where m = number of sites and n = number of reference haplotypes.

    :param numpy.ndarray ref_panel: An array of reference haplotypes.
    :return: An array of reference genotypes.
    :rtype: numpy.ndarray
    """
    ALLOWED_ALLELE_STATES = np.array([0, 1, NONCOPY], dtype=np.int32)
    assert np.all(
        np.isin(np.unique(ref_panel), ALLOWED_ALLELE_STATES)
    ), f"Reference haplotypes contain illegal allele states."
    num_sites = ref_panel.shape[0]
    num_haps = ref_panel.shape[1]
    genotypes = np.zeros((num_sites, num_haps, num_haps), dtype=np.int32) - np.inf
    for i in range(num_sites):
        site_alleles = ref_panel[i, :]
        genotypes[i, :, :] = np.add.outer(site_alleles, site_alleles)
        genotypes[i, site_alleles == NONCOPY, :] = NONCOPY
        genotypes[i, :, site_alleles == NONCOPY] = NONCOPY
    return genotypes


def convert_haplotypes_to_unphased_genotypes(query):
    """
    Convert an array of two haplotypes into an array of genotypes encoded as allele dosages,
    and return the genotypes.

    It is assumed all sites are biallelic and alleles are encoded as ancestral/derived.
    The only allowable allele states are 0, 1, and MISSING.
    TODO: Handle multiallelic sites.

    Allowable genotype values are 0, 1, 2, and MISSING. If either one haplotype is MISSING
    at a site, then the genotype at the site is assigned MISSING.

    The input query haplotypes is of size (2, m), and the output genotypes is of size (1, m),
    where m = number of sites.

    :param numpy.ndarray query: An array of two query haplotypes.
    :return: An array of query genotypes.
    :rtype: numpy.ndarray
    """
    ALLOWED_ALLELE_STATES = np.array([0, 1, MISSING], dtype=np.int32)
    assert np.all(
        np.isin(np.unique(query), ALLOWED_ALLELE_STATES)
    ), f"Query haplotypes contain illegal allele states."
    num_sites = query.shape[1]
    num_haps = query.shape[0]
    assert num_haps == 2, "Two haplotypes are expected in a diploid query."
    genotypes = np.zeros((1, num_sites), dtype=np.int32) - np.inf
    genotypes[0, :] = np.sum(query, axis=0)
    genotypes[0, np.any(query == MISSING, axis=0)] = MISSING
    return genotypes


@jit.numba_njit
def get_num_copiable_entries(ref_panel):
    assert ref_panel.ndim in [2, 3], "Reference panel array has incorrect dimensions."
    assert np.all(
        ref_panel != MISSING
    ), "Reference panel cannot contain any MISSING values."
    if ref_panel.ndim == 2:
        num_copiable_entries = np.sum(ref_panel != NONCOPY, axis=1)
    else:
        num_sites = ref_panel.shape[0]
        num_copiable_entries = np.zeros(num_sites, dtype=np.int32)
        for i in range(num_sites):
            num_copiable_entries[i] = np.sum(ref_panel[i, :, :] != NONCOPY)
    assert np.all(
        num_copiable_entries > 0
    ), "Number of copiable entries must be greater than zero at all sites."
    return num_copiable_entries


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
    It is assumed that MISSING and NONCOPY are not encoded in these strings.

    Note that MISSING and NONCOPY values are excluded from the counts.

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
    exclusion_set = np.array([MISSING, NONCOPY])
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
def get_index_in_emission_matrix_diploid(ref_genotype, query_genotype):
    """
    Compare the implied unphased genotypes (allele dosages) of
    the reference and query to get the index of the entry
    in the emission probability matrix, and return the index.
    """
    if query_genotype == MISSING:
        return MISSING_INDEX
    else:
        is_match = ref_genotype == query_genotype
        is_ref_het = ref_genotype == 1
        is_query_het = query_genotype == 1
        return 4 * is_match + 2 * is_ref_het + is_query_het


@jit.numba_njit
def get_index_in_emission_matrix_diploid_genotypes(
    ref_genotypes, query_genotype, num_ref_haps
):
    if query_genotype == MISSING:
        return MISSING_INDEX * np.ones((num_ref_haps, num_ref_haps), dtype=np.int64)
    else:
        is_match = ref_genotypes == query_genotype
        is_ref_het = ref_genotypes == 1
        is_query_het = query_genotype == 1
        return 4 * is_match + 2 * is_ref_het + is_query_het


@jit.numba_njit
def get_emission_matrix_haploid(mu, num_sites, num_alleles, scale_mutation_rate):
    """
    Compute an emission probability matrix for the haploid case, and return it.

    The emission probability matrix is of size (num_sites, 2). The first entry
    of each row corresponds to an emission when alleles do not match, and the second
    entry to an emission when alleles match.

    By default, there is no scaling of mutation rates based on the number of alleles,
    so that mutation probability is the probability of mutation to **any allele**
    (therefore, summing over all the states that can be switched to).

    This means that we must rescale the probability of mutation to a different allele
    by the number of alleles at the site.

    Optionally, scale mutation based on the number of alleles, so that mutation
    probability is the probability of mutation **any given one** of the alleles.
    The overall mutation probability is then (num_alleles - 1) * mutation probability.

    :param float/numpy.ndarray(dtype=np.float64) mu: Probability of mutation.
    :param int num_sites: Number of sites.
    :param numpy.ndarray(dtype=np.int8): Number of distinct alleles per site.
    :param bool scale_mutation_rate: Scale mutation rate based on the number of alleles if True (default).
    """
    assert len(mu) == len(
        num_alleles
    ), "Arrays of mutation probability and number of alleles are unequal in length."
    if isinstance(mu, float):
        mu = np.zeros(num_sites, dtype=np.float64) + mu
    emission_matrix = np.zeros((num_sites, 2), np.float64) - np.inf
    for i in range(num_sites):
        if num_alleles[i] == 1:
            # Set probabilities at invariant sites.
            prob_mutation = 0
            prob_no_mutation = 1
        else:
            if scale_mutation_rate:
                prob_mutation = mu[i]
                prob_no_mutation = 1 - (num_alleles[i] - 1) * mu[i]
            else:
                prob_mutation = mu[i] / (num_alleles[i] - 1)
                prob_no_mutation = 1 - mu[i]
        emission_matrix[i, 0] = prob_mutation
        emission_matrix[i, 1] = prob_no_mutation
    return emission_matrix


@jit.numba_njit
def get_emission_matrix_diploid(mu, num_sites, num_alleles, scale_mutation_rate):
    assert len(mu) == len(
        num_alleles
    ), "Arrays of mutation probability and number of alleles are unequal in length."
    if isinstance(mu, float):
        mu = np.zeros(num_sites, dtype=np.float64) + mu
    prob_mutation = np.zeros(num_sites, dtype=np.float64) - np.inf
    prob_no_mutation = np.zeros(num_sites, dtype=np.float64) - np.inf
    emission_matrix = np.zeros((num_sites, 8), dtype=np.float64) - np.inf
    for i in range(num_sites):
        if num_alleles[i] == 1:
            # Set probabilities at invariant sites.
            prob_mutation[i] = 0
            prob_no_mutation[i] = 1
        else:
            if scale_mutation_rate:
                prob_mutation[i] = mu[i]
                prob_no_mutation[i] = 1 - (num_alleles[i] - 1) * mu[i]
            else:
                prob_mutation[i] = mu[i] / (num_alleles[i] - 1)
                prob_no_mutation[i] = 1 - mu[i]
    for i in range(num_sites):
        emission_matrix[i, EQUAL_BOTH_HOM] = prob_no_mutation[i] ** 2
        emission_matrix[i, UNEQUAL_BOTH_HOM] = prob_mutation[i] ** 2
        emission_matrix[i, BOTH_HET] = prob_no_mutation[i] ** 2 + prob_mutation[i] ** 2
        emission_matrix[i, REF_HOM_OBS_HET] = 2 * prob_mutation[i] * prob_no_mutation[i]
        emission_matrix[i, REF_HET_OBS_HOM] = prob_mutation[i] * prob_no_mutation[i]
        emission_matrix[i, MISSING_INDEX] = 1
    return emission_matrix


@jit.numba_njit
def get_emission_probability_haploid(ref_allele, query_allele, site, emission_matrix):
    if ref_allele == NONCOPY:
        return 0.0
    else:
        emission_index = get_index_in_emission_matrix_haploid(ref_allele, query_allele)
        return emission_matrix[site, emission_index]


@jit.numba_njit
def get_emission_probability_diploid(
    ref_genotype, query_genotype, site, emission_matrix
):
    if ref_genotype == NONCOPY:
        return 0.0
    else:
        emission_index = get_index_in_emission_matrix_diploid(
            ref_genotype, query_genotype
        )
        return emission_matrix[site, emission_index]


@jit.numba_njit
def get_emission_probability_diploid_genotypes(
    ref_genotypes, query_genotype, site, emission_matrix
):
    assert ref_genotypes.shape[0] == ref_genotypes.shape[1]
    num_ref_haps = len(ref_genotypes)
    emission_probs = np.zeros((num_ref_haps, num_ref_haps), dtype=np.float64)
    for i in range(num_ref_haps):
        for j in range(num_ref_haps):
            if ref_genotypes[i, j] == NONCOPY:
                emission_probs[i, j] = 0.0
            else:
                emission_index = get_index_in_emission_matrix_diploid(
                    ref_genotypes[i, j], query_genotype
                )
                emission_probs[i, j] = emission_matrix[site, emission_index]
    return emission_probs
