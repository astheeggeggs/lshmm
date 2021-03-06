"""External API definitions."""
import warnings

import numpy as np

from .forward_backward.fb_diploid_variants_samples import (
    backward_ls_dip_loop,
    forward_ls_dip_loop,
)
from .forward_backward.fb_haploid_variants_samples import (
    backwards_ls_hap,
    forwards_ls_hap,
)
from .vit_diploid_variants_samples import (
    backwards_viterbi_dip,
    forwards_viterbi_dip_low_mem,
    get_phased_path,
)
from .vit_haploid_variants_samples import (
    backwards_viterbi_hap,
    forwards_viterbi_hap_lower_mem_rescaling,
)

EQUAL_BOTH_HOM = 4
UNEQUAL_BOTH_HOM = 0
BOTH_HET = 7
REF_HOM_OBS_HET = 1
REF_HET_OBS_HOM = 2


def check_alleles(alleles, m):
    """
    Checks the specified allele list and returns a list of lists
    of alleles of length num_sites.
    If alleles is a 1D list of strings, assume that this list is used
    for each site and return num_sites copies of this list.
    Otherwise, raise a ValueError if alleles is not a list of length
    num_sites.
    """
    if isinstance(alleles[0], str):
        return np.int8([len(alleles) for _ in range(m)])
    if len(alleles) != m:
        raise ValueError("Malformed alleles list")
    n_alleles = np.int8([(len(alleles_site)) for alleles_site in alleles])
    return n_alleles


def checks(
    reference_panel,
    query,
    mutation_rate,
    recombination_rate,
    scale_mutation_based_on_n_alleles,
):

    ref_shape = reference_panel.shape
    ploidy = len(ref_shape) - 1

    if ploidy not in (1, 2):
        raise ValueError("Ploidy not supported.")

    if not (query.shape[1] == ref_shape[0]):
        raise ValueError(
            "Number of variants in query does not match reference_panel. If haploid, ensure variant x sample matrices are passed."
        )

    if (ploidy == 2) and (not (ref_shape[1] == ref_shape[2])):
        raise ValueError(
            "reference_panel dimensions incorrect, perhaps a sample x sample x variant matrix was passed. Expected variant x sample x sample."
        )

    m = ref_shape[0]
    n = ref_shape[1]

    print(f"Number of sites: {m}, number of samples: {n}\n")

    # Ensure that the mutation rate is either a scalar or vector of length m
    if not isinstance(mutation_rate, float) and (mutation_rate is not None):
        if type(mutation_rate is np.ndarray):
            if mutation_rate.shape[0] is not m:
                raise ValueError(
                    f"mutation_rate is not a scalar or vector of length m: {m}"
                )
        else:
            raise ValueError(
                f"mutation_rate is not a scalar or vector of length m: {m}"
            )

    # Ensure that the recombination probabilities is either a scalar or a vector of length m
    if recombination_rate.shape[0] is not m:
        raise ValueError(f"recombination_rate is not a vector of length m: {m}")

    if isinstance(mutation_rate, float) and not (scale_mutation_based_on_n_alleles):
        warnings.warn(
            "Passed a scalar mutation rate, but not rescaling this mutation rate conditional on the number of alleles at the site"
        )

    if type(mutation_rate is np.ndarray) and (scale_mutation_based_on_n_alleles):
        warnings.warn(
            "Passed a vector of mutation rates, but rescaling each mutation rate conditional on the number of alleles at each site"
        )

    return n, m, ploidy


# def forwards(n, m, G_or_H, s, e, r):
#     """
#     Run the Li and Stephens forwards algorithm on haplotype or
#     unphased genotype data.
#     """
#     template_dimensions = G_or_H.shape
#     assert len(template_dimensions) in [2, 3]

#     if len(template_dimensions) == 2:
#         # Haploid
#         assert (G_or_H.shape == np.array([m, n])).all()
#         F, c, ll = forwards_ls_hap(n, m, G_or_H, s, e, r, norm=True)
#     else:
#         # Diploid
#         assert (G_or_H.shape == np.array([m, n, n])).all()
#         F, c, ll = forward_ls_dip_loop(n, m, G_or_H, s, e, r, norm=True)

#     return F, c, ll


def forwards(
    reference_panel,
    query,
    recombination_rate,
    alleles=None,
    mutation_rate=None,
    scale_mutation_based_on_n_alleles=True,
):
    """
    Run the Li and Stephens forwards algorithm on haplotype or
    unphased genotype data.
    """
    n, m, ploidy = checks(
        reference_panel,
        query,
        mutation_rate,
        recombination_rate,
        scale_mutation_based_on_n_alleles,
    )
    # Check alleles should go in here, and mofify e before passing to the algorithm
    # If alleles is not passed, we don't perform a test of alleles, but set n_alleles based on the reference_panel.

    if alleles is None:
        n_alleles = np.int8(
            [
                len(np.unique(np.append(reference_panel[j, :], query[:, j])))
                for j in range(reference_panel.shape[0])
            ]
        )
    else:
        n_alleles = check_alleles(alleles, m)

    if mutation_rate is None:
        # Set the mutation rate to be the proposed mutation rate in Li and Stephens (2003).
        theta_tilde = 1 / np.sum([1 / k for k in range(1, n - 1)])
        mutation_rate = 0.5 * (theta_tilde / (n + theta_tilde))

    if isinstance(mutation_rate, float):
        mutation_rate = mutation_rate * np.ones(m)

    if ploidy == 1:
        # Haploid
        # Evaluate emission probabilities here, using the mutation rate - this can take a scalar or vector.
        e = np.zeros((m, 2))

        if scale_mutation_based_on_n_alleles:
            # Scale mutation based on the number of alleles - so the mutation rate is the mutation rate to one of the alleles.
            # The overall mutation rate is then (n_alleles - 1) * mutation_rate.
            e[:, 0] = mutation_rate - mutation_rate * np.equal(
                n_alleles, np.ones(m)
            )  # Added boolean in case we're at an invariant site
            e[:, 1] = 1 - (n_alleles - 1) * mutation_rate
        else:
            # No scaling based on the number of alleles - so the mutation rate is the mutation rate to anything.
            # Which means that we must rescale the mutation rate to a different allele, by the number of alleles.
            for j in range(m):
                if n_alleles[j] == 1:  # In case we're at an invariant site
                    e[j, 0] = 0
                    e[j, 1] = 1
                else:
                    e[j, 0] = mutation_rate[j] / (n_alleles[j] - 1)
                    e[j, 1] = 1 - mutation_rate[j]

        (
            forward_array,
            normalisation_factor_from_forward,
            log_likelihood,
        ) = forwards_ls_hap(
            n, m, reference_panel, query, e, recombination_rate, norm=True
        )
    else:
        # Diploid
        # Evaluate emission probabilities here, using the mutation rate - this can take a scalar or vector.
        # DEV: there's a wrinkle here.
        e = np.zeros((m, 8))
        e[:, EQUAL_BOTH_HOM] = (1 - mutation_rate) ** 2
        e[:, UNEQUAL_BOTH_HOM] = mutation_rate ** 2
        e[:, BOTH_HET] = 1 - mutation_rate
        e[:, REF_HOM_OBS_HET] = 2 * mutation_rate * (1 - mutation_rate)
        e[:, REF_HET_OBS_HOM] = mutation_rate * (1 - mutation_rate)

        (
            forward_array,
            normalisation_factor_from_forward,
            log_likelihood,
        ) = forward_ls_dip_loop(
            n, m, reference_panel, query, e, recombination_rate, norm=True
        )

    return forward_array, normalisation_factor_from_forward, log_likelihood


# def backwards(n, m, G_or_H, s, e, c, r):
#     """
#     Run the Li and Stephens backwards algorithm on haplotype or
#     unphased genotype data.
#     """
#     template_dimensions = G_or_H.shape
#     assert len(template_dimensions) in [2, 3]

#     if len(template_dimensions) == 2:
#         # Haploid
#         assert (G_or_H.shape == np.array([m, n])).all()
#         B = backwards_ls_hap(n, m, G_or_H, s, e, c, r)
#     else:
#         # Diploid
#         assert (G_or_H.shape == np.array([m, n, n])).all()
#         B = backward_ls_dip_loop(n, m, G_or_H, s, e, c, r)

#     return B


def backwards(
    reference_panel,
    query,
    normalisation_factor_from_forward,
    recombination_rate,
    alleles=None,
    mutation_rate=None,
    scale_mutation_based_on_n_alleles=True,
):
    """
    Run the Li and Stephens backwards algorithm on haplotype or
    unphased genotype data.
    """
    n, m, ploidy = checks(
        reference_panel,
        query,
        mutation_rate,
        recombination_rate,
        scale_mutation_based_on_n_alleles,
    )
    # Check alleles should go in here, and mofify e before passing to the algorithm
    # If alleles is not passed, we don't perform a test of alleles, but set n_alleles based on the reference_panel.
    if alleles is None:
        n_alleles = np.int8(
            [
                len(np.unique(np.append(reference_panel[j, :], query[:, j])))
                for j in range(reference_panel.shape[0])
            ]
        )
    else:
        n_alleles = check_alleles(alleles, m)

    if mutation_rate is None:
        # Set the mutation rate to be the proposed mutation rate in Li and Stephens (2003).
        theta_tilde = 1 / np.sum([1 / k for k in range(1, n - 1)])
        mutation_rate = 0.5 * (theta_tilde / (n + theta_tilde))

    if ploidy == 1:
        # Haploid
        # Evaluate emission probabilities here, using the mutation rate - this can take a scalar or vector.
        e = np.zeros((m, 2))

        if scale_mutation_based_on_n_alleles:
            # Scale mutation based on the number of alleles - so the mutation rate is the mutation rate to one of the alleles.
            # The overall mutation rate is then (n_alleles - 1) * mutation_rate.
            e[:, 0] = mutation_rate - mutation_rate * np.equal(
                n_alleles, np.ones(m)
            )  # Added boolean in case we're at an invariant site
            e[:, 1] = 1 - (n_alleles - 1) * mutation_rate
        else:
            # No scaling based on the number of alleles - so the mutation rate is the mutation rate to anything.
            # Which means that we must rescale the mutation rate to a different allele, by the number of alleles.
            for j in range(m):
                if n_alleles[j] == 1:  # In case we're at an invariant site
                    e[j, 0] = 0
                    e[j, 1] = 1
                else:
                    e[j, 0] = mutation_rate[j] / (n_alleles[j] - 1)
                    e[j, 1] = 1 - mutation_rate[j]

        backwards_array = backwards_ls_hap(
            n,
            m,
            reference_panel,
            query,
            e,
            normalisation_factor_from_forward,
            recombination_rate,
        )
    else:
        # Diploid
        # Evaluate emission probabilities here, using the mutation rate - this can take a scalar or vector.
        # DEV: there's a wrinkle here.
        e = np.zeros((m, 8))
        e[:, EQUAL_BOTH_HOM] = (1 - mutation_rate) ** 2
        e[:, UNEQUAL_BOTH_HOM] = mutation_rate ** 2
        e[:, BOTH_HET] = 1 - mutation_rate
        e[:, REF_HOM_OBS_HET] = 2 * mutation_rate * (1 - mutation_rate)
        e[:, REF_HET_OBS_HOM] = mutation_rate * (1 - mutation_rate)

        backwards_array = backward_ls_dip_loop(
            n,
            m,
            reference_panel,
            query,
            e,
            normalisation_factor_from_forward,
            recombination_rate,
        )

    return backwards_array


# def viterbi(n, m, G_or_H, s, e, r):
#     """
#     Run the Li and Stephens Viterbi algorithm on haplotype or
#     unphased genotype data.
#     """
#     template_dimensions = G_or_H.shape
#     assert len(template_dimensions) in [2, 3]

#     if len(template_dimensions) == 2:
#         # Haploid
#         assert (G_or_H.shape == np.array([m, n])).all()
#         V, P, ll = forwards_viterbi_hap_lower_mem_rescaling(n, m, G_or_H, s, e, r)
#         path = backwards_viterbi_hap(m, V, P)
#     else:
#         # Diploid
#         assert (G_or_H.shape == np.array([m, n, n])).all()
#         V, P, ll = forwards_viterbi_dip_low_mem(n, m, G_or_H, s, e, r)
#         unphased_path = backwards_viterbi_dip(m, V, P)
#         path = get_phased_path(n, unphased_path)

#     return path, ll


def viterbi(
    reference_panel,
    query,
    recombination_rate,
    alleles=None,
    mutation_rate=None,
    scale_mutation_based_on_n_alleles=True,
):
    """
    Run the Li and Stephens Viterbi algorithm on haplotype or
    unphased genotype data.
    """
    n, m, ploidy = checks(
        reference_panel,
        query,
        mutation_rate,
        recombination_rate,
        scale_mutation_based_on_n_alleles,
    )

    if alleles is None:
        n_alleles = np.int8(
            [
                len(np.unique(np.append(reference_panel[j, :], query[:, j])))
                for j in range(reference_panel.shape[0])
            ]
        )
    else:
        n_alleles = check_alleles(alleles, m)

    if mutation_rate is None:
        # Set the mutation rate to be the proposed mutation rate in Li and Stephens (2003).
        theta_tilde = 1 / np.sum([1 / k for k in range(1, n - 1)])
        mutation_rate = 0.5 * (theta_tilde / (n + theta_tilde))

    if ploidy == 1:
        # Haploid
        # Evaluate emission probabilities here, using the mutation rate - this can take a scalar or vector.
        e = np.zeros((m, 2))

        if scale_mutation_based_on_n_alleles:
            # Scale mutation based on the number of alleles - so the mutation rate is the mutation rate to one of the alleles.
            # The overall mutation rate is then (n_alleles - 1) * mutation_rate.
            e[:, 0] = mutation_rate - mutation_rate * np.equal(
                n_alleles, np.ones(m)
            )  # Added boolean in case we're at an invariant site
            e[:, 1] = 1 - (n_alleles - 1) * mutation_rate
        else:
            # No scaling based on the number of alleles - so the mutation rate is the mutation rate to anything.
            # Which means that we must rescale the mutation rate to a different allele, by the number of alleles.
            for j in range(m):
                if n_alleles[j] == 1:  # In case we're at an invariant site
                    e[j, 0] = 0
                    e[j, 1] = 1
                else:
                    e[j, 0] = mutation_rate[j] / (n_alleles[j] - 1)
                    e[j, 1] = 1 - mutation_rate[j]

        V, P, log_likelihood = forwards_viterbi_hap_lower_mem_rescaling(
            n, m, reference_panel, query, e, recombination_rate
        )
        most_likely_path = backwards_viterbi_hap(m, V, P)
    else:
        # Diploid
        # Evaluate emission probabilities here, using the mutation rate - this can take a scalar or vector.
        # DEV: there's a wrinkle here.
        e = np.zeros((m, 8))
        e[:, EQUAL_BOTH_HOM] = (1 - mutation_rate) ** 2
        e[:, UNEQUAL_BOTH_HOM] = mutation_rate ** 2
        e[:, BOTH_HET] = 1 - mutation_rate
        e[:, REF_HOM_OBS_HET] = 2 * mutation_rate * (1 - mutation_rate)
        e[:, REF_HET_OBS_HOM] = mutation_rate * (1 - mutation_rate)

        V, P, log_likelihood = forwards_viterbi_dip_low_mem(
            n, m, reference_panel, query, e, recombination_rate
        )
        unphased_path = backwards_viterbi_dip(m, V, P)
        most_likely_path = get_phased_path(n, unphased_path)

    return most_likely_path, log_likelihood
