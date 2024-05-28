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
    prob_recombination,
    prob_mutation,
    scale_mutation_rate,
):
    """
    Check that the input data and parameters are valid.

    The reference panel must be a matrix of size (m, n) or (m, n, n).
    The query must be a matrix of size (k, m) or (k, m, 2).

    m = number of sites.
    n = number of samples in the reference panel (haplotypes, not individuals).
    k = number of samples in the query (haplotypes, not individuals).

    The mutation rate can be scaled according to the set of alleles
    that can be mutated to based on the number of distinct alleles at each site.

    :param numpy.ndarray(dtype=int) reference_panel: Matrix of size (m, n) or (m, n, n).
    :param numpy.ndarray(dtype=int) query: Matrix of size (k, m) or (k, m, 2).
    :param numpy.ndarray(dtype=float) prob_recombination: Scalar or vector of length m.
    :param numpy.ndarray(dtype=float) prob_mutation: Scalar or vector of length m.
    :param bool scale_mutation_rate: Scale mutation rate or not.
    :return: Number of reference haplotypes, Number of sites, ploidy
    :rtype: tuple
    """
    # Check reference panel
    if not len(reference_panel.shape) in (2, 3):
        err_msg = "Reference panel array must have 2 or 3 dimensions."
        raise ValueError(err_msg)

    if len(reference_panel.shape) == 2:
        num_sites, num_ref_haps = reference_panel.shape
        ploidy = 1
    else:
        num_sites, num_ref_haps, _ = reference_panel.shape
        ploidy = 2

    if ploidy == 2 and (reference_panel.shape[1] != reference_panel.shape[2]):
        err_msg = (
            "Reference_panel dimensions are incorrect, "
            "perhaps a sample x sample x variant matrix was passed. "
            "Expected sites x samples x samples."
        )
        raise ValueError(err_msg)

    # Check query sequence(s)
    if query.shape[1] != num_sites:
        err_msg = (
            "Number of sites in query does not match reference panel. "
            "If haploid, ensure a sites x samples matrix is passed."
        )
        raise ValueError(err_msg)

    # Ensure the mutation probability is a scalar or a vector of length m.
    if isinstance(prob_mutation, (int, float)):
        if not scale_mutation_rate:
            warn_msg = "Passed a scalar mutation probability, but not rescaling it."
            warnings.warn(warn_msg)
    elif isinstance(prob_mutation, np.ndarray) and prob_mutation.shape[0] == num_sites:
        if scale_mutation_rate:
            warn_msg = "Passed a vector of mutation probabilities. Rescaling them."
            warnings.warn(warn_msg)
    elif prob_mutation is None:
        warn_msg = (
            "No mutation probability passed. "
            "Setting it based on Li & Stephens (2003) equations A2 and A3."
        )
        warnings.warn(warn_msg)
    else:
        err_msg = f"Mutation probability is not None, a scalar, or vector of length m."
        raise ValueError(err_msg)

    # Ensure the recombination probability is a scalar or a vector of length m.
    if not (
        isinstance(prob_recombination, (int, float))
        or (
            isinstance(prob_recombination, np.ndarray)
            and prob_recombination.shape[0] == num_sites
        )
    ):
        err_msg = f"Recombination probability is not a scalar or vector of length m."
        raise ValueError(err_msg)

    return (num_ref_haps, num_sites, ploidy)


def set_emission_probabilities(
    num_ref_haps,
    num_sites,
    ploidy,
    num_alleles,
    prob_mutation,
    scale_mutation_rate,
):
    if prob_mutation is None:
        # Set the mutation probability to be that proposed in Li & Stephens (2003).
        theta_tilde = 1 / np.sum([1 / k for k in range(1, num_ref_haps - 1)])
        prob_mutation = 0.5 * (theta_tilde / (num_ref_haps + theta_tilde))

    if isinstance(prob_mutation, float):
        prob_mutation = prob_mutation * np.ones(num_sites)

    if ploidy == 1:
        emission_probs = core.get_emission_matrix_haploid(
            mu=prob_mutation,
            num_sites=num_sites,
            num_alleles=num_alleles,
            scale_mutation_rate=scale_mutation_rate,
        )
    else:
        emission_probs = core.get_emission_matrix_diploid(
            mu=prob_mutation, num_sites=num_sites
        )

    return emission_probs


def forwards(
    reference_panel,
    query,
    num_alleles,
    prob_recombination,
    *,
    prob_mutation=None,
    scale_mutation_rate=None,
    normalise=None,
):
    """Run the forwards algorithm on haplotype or unphased genotype data."""
    if scale_mutation_rate is None:
        scale_mutation_rate = True

    if normalise is None:
        normalise = True

    num_ref_haps, num_sites, ploidy = check_inputs(
        reference_panel=reference_panel,
        query=query,
        prob_recombination=prob_recombination,
        prob_mutation=prob_mutation,
        scale_mutation_rate=scale_mutation_rate,
    )

    emission_probs = set_emission_probabilities(
        num_ref_haps=num_ref_haps,
        num_sites=num_sites,
        ploidy=ploidy,
        num_alleles=num_alleles,
        prob_mutation=prob_mutation,
        scale_mutation_rate=scale_mutation_rate,
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
        reference_panel,
        query,
        emission_probs,
        prob_recombination,
        norm=normalise,
    )

    return forward_array, normalisation_factor_from_forward, log_lik


def backwards(
    reference_panel,
    query,
    num_alleles,
    normalisation_factor_from_forward,
    prob_recombination,
    *,
    prob_mutation=None,
    scale_mutation_rate=None,
):
    """Run the backwards algorithm on haplotype or unphased genotype data."""
    if scale_mutation_rate is None:
        scale_mutation_rate = True

    num_ref_haps, num_sites, ploidy = check_inputs(
        reference_panel=reference_panel,
        query=query,
        prob_recombination=prob_recombination,
        prob_mutation=prob_mutation,
        scale_mutation_rate=scale_mutation_rate,
    )

    emission_probs = set_emission_probabilities(
        num_ref_haps=num_ref_haps,
        num_sites=num_sites,
        ploidy=ploidy,
        num_alleles=num_alleles,
        prob_mutation=prob_mutation,
        scale_mutation_rate=scale_mutation_rate,
    )

    if ploidy == 1:
        backward_function = backwards_ls_hap
    else:
        backward_function = backward_ls_dip_loop

    backwards_array = backward_function(
        num_ref_haps,
        num_sites,
        reference_panel,
        query,
        emission_probs,
        normalisation_factor_from_forward,
        prob_recombination,
    )

    return backwards_array


def viterbi(
    reference_panel,
    query,
    num_alleles,
    prob_recombination,
    *,
    prob_mutation=None,
    scale_mutation_rate=None,
):
    """Run the Viterbi algorithm on haplotype or unphased genotype data."""
    if scale_mutation_rate is None:
        scale_mutation_rate = True

    num_ref_haps, num_sites, ploidy = check_inputs(
        reference_panel=reference_panel,
        query=query,
        prob_recombination=prob_recombination,
        prob_mutation=prob_mutation,
        scale_mutation_rate=scale_mutation_rate,
    )

    emission_probs = set_emission_probabilities(
        num_ref_haps=num_ref_haps,
        num_sites=num_sites,
        ploidy=ploidy,
        num_alleles=num_alleles,
        prob_mutation=prob_mutation,
        scale_mutation_rate=scale_mutation_rate,
    )

    if ploidy == 1:
        V, P, log_lik = forwards_viterbi_hap_lower_mem_rescaling(
            num_ref_haps,
            num_sites,
            reference_panel,
            query,
            emission_probs,
            prob_recombination,
        )
        best_path = backwards_viterbi_hap(num_sites, V, P)
    else:
        V, P, log_lik = forwards_viterbi_dip_low_mem(
            num_ref_haps,
            num_sites,
            reference_panel,
            query,
            emission_probs,
            prob_recombination,
        )
        unphased_path = backwards_viterbi_dip(num_sites, V, P)
        best_path = get_phased_path(num_ref_haps, unphased_path)

    return best_path, log_lik


def path_loglik(
    reference_panel,
    query,
    num_alleles,
    path,
    prob_recombination,
    *,
    prob_mutation=None,
    scale_mutation_rate=None,
):
    """Evaluate the log-likelihood of a copying path for a query through a reference panel."""
    if scale_mutation_rate is None:
        scale_mutation_rate = True

    num_ref_haps, num_sites, ploidy = check_inputs(
        reference_panel=reference_panel,
        query=query,
        prob_recombination=prob_recombination,
        prob_mutation=prob_mutation,
        scale_mutation_rate=scale_mutation_rate,
    )

    emission_probs = set_emission_probabilities(
        num_ref_haps=num_ref_haps,
        num_sites=num_sites,
        ploidy=ploidy,
        num_alleles=num_alleles,
        prob_mutation=prob_mutation,
        scale_mutation_rate=scale_mutation_rate,
    )

    if ploidy == 1:
        path_ll_function = path_ll_hap
    else:
        path_ll_function = path_ll_dip

    log_lik = path_ll_function(
        num_ref_haps,
        num_sites,
        reference_panel,
        path,
        query,
        emission_probs,
        prob_recombination,
    )

    return log_lik
