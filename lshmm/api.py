"""External API definitions."""
import numpy as np

from .forward_backward.fb_diploid_variants_samples import (
    backward_ls_dip_loop,
    forward_ls_dip_loop,
)
from .forward_backward.fb_haploid_variants_samples import (
    backwards_ls_hap,
    forwards_ls_hap,
)
from .viterbi.vit_diploid_variants_samples import (
    backwards_viterbi_dip,
    forwards_viterbi_dip_low_mem,
    get_phased_path,
)
from .viterbi.vit_haploid_variants_samples import (
    backwards_viterbi_hap,
    forwards_viterbi_hap_lower_mem_rescaling,
)

EQUAL_BOTH_HOM = 4
UNEQUAL_BOTH_HOM = 0
BOTH_HET = 7
REF_HOM_OBS_HET = 1
REF_HET_OBS_HOM = 2


def forwards(reference_panel, query, recombination_rate, mutation_rate):
    """
    Run the Li and Stephens forwards algorithm on haplotype or
    unphased genotype data.
    """
    ref_shape = reference_panel.shape
    ploidy = len(ref_shape) - 1

    if ploidy not in (1, 2):
        raise ValueError("Ploidy not supported.")

    if not (query.shape[0] == ref_shape[0]):
        raise ValueError(
            "Number of variants in query does not match reference_panel. If haploid, ensure variant x sample matrices are passed."
        )

    if (ploidy == 2) and (not (ref_shape[1] == ref_shape[2])):
        raise ValueError(
            "reference_panel dimensions incorrect, perhaps a sample x sample x variant matrix was passed. Expected variant x sample x sample."
        )

    m = ref_shape[0]
    n = ref_shape[1]

    if ploidy == 1:
        # Haploid
        # Evaluate emission probabilities here, using the mutation rate - this can take a scalar or vector.
        e = np.zeros((m, 2))
        e[:, 0] = mutation_rate
        e[:, 1] = 1 - mutation_rate

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
        e[:, UNEQUAL_BOTH_HOM] = mutations_rate ** 2
        e[:, BOTH_HET] = 1 - mutations_rate
        e[:, REF_HOM_OBS_HET] = 2 * mutations_rate * (1 - mutations_rate)
        e[:, REF_HET_OBS_HOM] = mutations_rate * (1 - mutations_rate)

        (
            forward_array,
            normalisation_factor_from_forward,
            log_likelihood,
        ) = forward_ls_dip_loop(
            n, m, reference_panel, query, e, recombination_rate, norm=True
        )

    return forward_array, normalisation_factor_from_forward, log_likelihood


def backwards(
    reference_panel,
    query,
    recombination_rate,
    mutation_rate,
    normalisation_factor_from_forward,
):
    """
    Run the Li and Stephens backwards algorithm on haplotype or
    unphased genotype data.
    """
    ref_shape = reference_panel.shape
    ploidy = len(ref_shape) - 1

    if ploidy not in (1, 2):
        raise ValueError("Ploidy not supported.")

    if not (query.shape[0] == ref_shape[0]):
        raise ValueError(
            "Number of variants in query does not match reference_panel. If haploid, ensure variant x sample matrices are passed."
        )

    if (ploidy == 2) and (not (ref_shape[1] == ref_shape[2])):
        raise ValueError(
            "reference_panel dimensions incorrect, perhaps a sample x sample x variant matrix was passed. Expected variant x sample x sample."
        )

    m = ref_shape[0]
    n = ref_shape[1]

    if ploidy == 1:
        # Haploid
        # Evaluate emission probabilities here, using the mutation rate - this can take a scalar or vector.
        e = np.zeros((m, 2))
        e[:, 0] = mutation_rate
        e[:, 1] = 1 - mutation_rate

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
        e[:, UNEQUAL_BOTH_HOM] = mutations_rate ** 2
        e[:, BOTH_HET] = 1 - mutations_rate
        e[:, REF_HOM_OBS_HET] = 2 * mutations_rate * (1 - mutations_rate)
        e[:, REF_HET_OBS_HOM] = mutations_rate * (1 - mutations_rate)

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


def viterbi(reference_panel, query, mutation_rate, recombination_rate):
    """
    Run the Li and Stephens Viterbi algorithm on haplotype or
    unphased genotype data.
    """
    ref_shape = reference_panel.shape
    ploidy = len(ref_shape) - 1

    if ploidy not in (1, 2):
        raise ValueError("Ploidy not supported.")

    if not (query.shape[0] == ref_shape[0]):
        raise ValueError(
            "Number of variants in query does not match reference_panel. If haploid, ensure variant x sample matrices are passed."
        )

    if (ploidy == 2) and (not (ref_shape[1] == ref_shape[2])):
        raise ValueError(
            "reference_panel dimensions incorrect, perhaps a sample x sample x variant matrix was passed. Expected variant x sample x sample."
        )

    m = ref_shape[0]
    n = ref_shape[1]

    if ploidy == 1:
        # Haploid
        # Evaluate emission probabilities here, using the mutation rate - this can take a scalar or vector.
        # DEV: there's a wrinkle here.
        e = np.zeros((m, 2))
        e[:, 0] = mutation_rate
        e[:, 1] = 1 - mutation_rate

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
        e[:, UNEQUAL_BOTH_HOM] = mutations_rate ** 2
        e[:, BOTH_HET] = 1 - mutations_rate
        e[:, REF_HOM_OBS_HET] = 2 * mutations_rate * (1 - mutations_rate)
        e[:, REF_HET_OBS_HOM] = mutations_rate * (1 - mutations_rate)

        V, P, log_likelihood = forwards_viterbi_dip_low_mem(
            n, m, reference_panel, query, e, recombination_rate
        )
        unphased_path = backwards_viterbi_dip(m, V, P)
        most_likely_path = get_phased_path(n, unphased_path)

    return most_likely_path, log_likelihood
