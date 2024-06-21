"""External API definitions."""

import warnings

import numpy as np

from . import core
from .fb_diploid import (
    backward_ls_dip_loop,
    forward_ls_dip_loop,
)
from .fb_haploid import (
    backwards_ls_hap,
    forwards_ls_hap,
)
from .vit_diploid import (
    backwards_viterbi_dip,
    forwards_viterbi_dip_low_mem,
    get_phased_path,
    path_ll_dip,
)
from .vit_haploid import (
    backwards_viterbi_hap,
    forwards_viterbi_hap_lower_mem_rescaling,
    path_ll_hap,
)


def check_inputs(
    reference_panel,
    query,
    ploidy,
    prob_recombination,
    prob_mutation,
    scale_mutation_rate,
):
    """
    Check that the input data and parameters are valid, and return data to run
    the HMM algorithms.

    The reference panel and query are arrays of size (m, n) and (k, m), respectively,
    where:
        m = number of sites.
        n = number of samples in the reference panel (haplotypes, not individuals).
        k = number of samples in the query (haplotypes, not individuals).

    TODO: Support running on multiple queries. Currently, only k = 1 or 2 is supported.

    The mutation rate can be scaled according to the set of alleles
    that can be mutated to based on the number of distinct alleles at each site.

    :param numpy.ndarray reference_panel: A panel of reference sequences.
    :param numpy.ndarray query: A query sequence.
    :param numpy.ndarray ploidy: Ploidy (only 1 or 2 are supported).
    :param numpy.ndarray prob_recombination: Recombination probability.
    :param numpy.ndarray prob_mutation: Mutation probability.
    :param bool scale_mutation_rate: Scale mutation rate or not.
    :return: Num. ref. hap., num. sites, checked ref. panel, checked query, emission prob. matrix.
    :rtype: tuple
    """
    # Check ploidy.
    if not ploidy in [1, 2]:
        err_msg = "Only ploidy levels 1 and 2 are supported."
        raise ValueError(err_msg)

    # Check the reference panel.
    if not len(reference_panel.shape) == 2:
        err_msg = "Reference panel array has incorrect dimensions."
        raise ValueError(err_msg)

    if np.any(reference_panel == core.MISSING):
        err_msg = "Reference panel cannot have any MISSING values."
        raise ValueError(err_msg)

    if ploidy == 2:
        if not np.all(np.isin(reference_panel, [0, 1, core.NONCOPY])):
            err_msg = "Reference panel has illegal alleles. "
            err_msg += "Only 0/1 encoding is supported in diploid mode."
            raise ValueError(err_msg)

    num_sites, num_ref_haps = reference_panel.shape

    # Check the queries.
    if query.shape[0] != ploidy:
        err_msg = "Query array has incorrect dimensions."
        raise ValueError(err_msg)

    if query.shape[1] != num_sites:
        err_msg = "Number of sites in the query and reference panel don't match."
        raise ValueError(err_msg)

    if np.any(query == core.NONCOPY):
        err_msg = "Query cannot have any NONCOPY values."
        raise ValueError(err_msg)

    if ploidy == 2:
        if not np.all(np.isin(query, [0, 1, core.MISSING])):
            err_msg = "Query has illegal alleles. "
            err_msg += "Only 0/1 encoding is supported in diploid mode."
            raise ValueError(err_msg)

    # Check the recombination probability.
    if isinstance(prob_recombination, (int, float)):
        prob_recombination = np.zeros(num_sites, dtype=np.float64) + prob_recombination
        prob_recombination[0] = 0.0
    elif (
        isinstance(prob_recombination, np.ndarray)
        and len(prob_recombination) == num_sites
    ):
        if prob_recombination[0] != 0:
            err_msg = "First value in the recombination probability array must be zero."
            raise ValueError(err_msg)
    else:
        err_msg = (
            "Recombination probability is not a scalar or an array of expected length."
        )
        raise ValueError(err_msg)

    # Set whether to scale mutation rates if not set already.
    if scale_mutation_rate is None:
        scale_mutation_rate = True

    # Check the mutation probability.
    if prob_mutation is None:
        warn_msg = "No mutation probability is passed; setting it as per Li & Stephens (2003) eqn. A2 and A3."
        warnings.warn(warn_msg)
        prob_mutation = core.estimate_mutation_probability(num_ref_haps)
        prob_mutation = np.zeros(num_sites, dtype=np.float64) + prob_mutation
    elif isinstance(prob_mutation, (int, float)):
        if not scale_mutation_rate:
            warn_msg = "A scalar mutation probability is passed, but not rescaling it."
            warnings.warn(warn_msg)
        prob_mutation = np.zeros(num_sites, dtype=np.float64) + prob_mutation
    elif isinstance(prob_mutation, np.ndarray) and len(prob_mutation) == num_sites:
        if scale_mutation_rate:
            warn_msg = "Rescaling an array of mutation probabilities."
            warnings.warn(warn_msg)
    else:
        err_msg = "Mutation probability is not a scalar or an array of expected length."
        raise ValueError(err_msg)

    # Calculate the emission probability matrix.
    num_alleles = core.get_num_alleles(reference_panel, query)
    if ploidy == 1:
        emission_matrix = core.get_emission_matrix_haploid(
            mu=prob_mutation,
            num_sites=num_sites,
            num_alleles=num_alleles,
            scale_mutation_rate=scale_mutation_rate,
        )
    else:
        emission_matrix = core.get_emission_matrix_diploid(
            mu=prob_mutation,
            num_sites=num_sites,
            num_alleles=num_alleles,
            scale_mutation_rate=scale_mutation_rate,
        )

    if ploidy == 1:
        return (
            num_ref_haps,
            num_sites,
            reference_panel,
            query,
            emission_matrix,
        )
    else:
        ref_panel_genotypes = core.convert_haplotypes_to_phased_genotypes(
            reference_panel
        )
        query_genotypes = core.convert_haplotypes_to_unphased_genotypes(query)
        return (
            num_ref_haps,
            num_sites,
            ref_panel_genotypes,
            query_genotypes,
            emission_matrix,
        )


def forwards(
    reference_panel,
    query,
    ploidy,
    prob_recombination,
    *,
    prob_mutation=None,
    scale_mutation_rate=None,
    normalise=None,
):
    """Run the forwards algorithm on haploid or diploid genotype data."""
    if normalise is None:
        normalise = True

    num_ref_haps, num_sites, ref_panel_checked, query_checked, emission_matrix = (
        check_inputs(
            reference_panel=reference_panel,
            query=query,
            ploidy=ploidy,
            prob_recombination=prob_recombination,
            prob_mutation=prob_mutation,
            scale_mutation_rate=scale_mutation_rate,
        )
    )

    if ploidy == 1:
        forward_function = forwards_ls_hap
    else:
        forward_function = forward_ls_dip_loop

    (
        forward_array,
        normalisation_factor_from_forward,
        log_lik,
    ) = forward_function(
        num_ref_haps,
        num_sites,
        ref_panel_checked,
        query_checked,
        emission_matrix,
        prob_recombination,
        norm=normalise,
    )

    return forward_array, normalisation_factor_from_forward, log_lik


def backwards(
    reference_panel,
    query,
    ploidy,
    normalisation_factor_from_forward,
    prob_recombination,
    *,
    prob_mutation=None,
    scale_mutation_rate=None,
):
    """Run the backwards algorithm on haploid or diploid genotype data."""
    num_ref_haps, num_sites, ref_panel_checked, query_checked, emission_matrix = (
        check_inputs(
            reference_panel=reference_panel,
            query=query,
            ploidy=ploidy,
            prob_recombination=prob_recombination,
            prob_mutation=prob_mutation,
            scale_mutation_rate=scale_mutation_rate,
        )
    )

    if ploidy == 1:
        backward_function = backwards_ls_hap
    else:
        backward_function = backward_ls_dip_loop

    backwards_array = backward_function(
        num_ref_haps,
        num_sites,
        ref_panel_checked,
        query_checked,
        emission_matrix,
        normalisation_factor_from_forward,
        prob_recombination,
    )

    return backwards_array


def viterbi(
    reference_panel,
    query,
    ploidy,
    prob_recombination,
    *,
    prob_mutation=None,
    scale_mutation_rate=None,
):
    """Run the Viterbi algorithm on haploid or diploid genotype data."""
    num_ref_haps, num_sites, ref_panel_checked, query_checked, emission_matrix = (
        check_inputs(
            reference_panel=reference_panel,
            query=query,
            ploidy=ploidy,
            prob_recombination=prob_recombination,
            prob_mutation=prob_mutation,
            scale_mutation_rate=scale_mutation_rate,
        )
    )

    if ploidy == 1:
        V, P, log_lik = forwards_viterbi_hap_lower_mem_rescaling(
            num_ref_haps,
            num_sites,
            ref_panel_checked,
            query_checked,
            emission_matrix,
            prob_recombination,
        )
        best_path = backwards_viterbi_hap(num_sites, V, P)
    else:
        V, P, log_lik = forwards_viterbi_dip_low_mem(
            num_ref_haps,
            num_sites,
            ref_panel_checked,
            query_checked,
            emission_matrix,
            prob_recombination,
        )
        unphased_path = backwards_viterbi_dip(num_sites, V, P)
        best_path = get_phased_path(num_ref_haps, unphased_path)

    return best_path, log_lik


def path_loglik(
    reference_panel,
    query,
    ploidy,
    path,
    prob_recombination,
    *,
    prob_mutation=None,
    scale_mutation_rate=None,
):
    """Evaluate the log-likelihood of a copying path for a query through a reference panel."""
    num_ref_haps, num_sites, ref_panel_checked, query_checked, emission_matrix = (
        check_inputs(
            reference_panel=reference_panel,
            query=query,
            ploidy=ploidy,
            prob_recombination=prob_recombination,
            prob_mutation=prob_mutation,
            scale_mutation_rate=scale_mutation_rate,
        )
    )

    if ploidy == 1:
        path_ll_function = path_ll_hap
    else:
        path_ll_function = path_ll_dip

    log_lik = path_ll_function(
        num_ref_haps,
        num_sites,
        ref_panel_checked,
        path,
        query_checked,
        emission_matrix,
        prob_recombination,
    )

    return log_lik
