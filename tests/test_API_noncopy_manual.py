import numpy as np
import pytest

import lshmm.vit_haploid as vh

MISSING = -1
NONCOPY = -2


# Helper functions
# TODO: Use the functions in the API instead.
def _get_emission_probabilities(m, p_mutation, n_alleles):
    # Note that this is different than `set_emission_probabilities` in `api.py`.
    # No scaling.
    e = np.zeros((m, 2))
    for j in range(m):
        if n_alleles[j] == 1:
            e[j, 0] = 0
            e[j, 1] = 1
        else:
            e[j, 0] = p_mutation[j] / (n_alleles[j] - 1)
            e[j, 1] = 1 - p_mutation[j]
    return e


def _get_num_alleles_per_site(H):
    # Used to rescale mutation and recombination probabilities.
    m = H.shape[0]  # Number of sites
    n_alleles = np.zeros(m, dtype=np.int64) - 1
    for i in range(m):
        uniq_a = np.unique(H[i, :])
        assert len(uniq_a) > 0
        assert MISSING not in uniq_a
        n_alleles[i] = np.sum(uniq_a != NONCOPY)
    return n_alleles


# Prepare test data for testing.
def get_example_data():
    """
    Assumptions:
    1. Non-NONCOPY states are contiguous.
    2. No MISSING states in ref. panel.
    """
    NC = NONCOPY    # Sugar
    # Trivial case 1
    H_trivial_1 = np.array([
        [NC, NC],
        [ 0,  1],
    ]).T
    query_trivial_1 = np.array([[0, 1]])
    path_trivial_1 = np.array([1, 1])
    # Trivial case 2
    H_trivial_2 = np.array([
        [NC,  1],
        [ 0,  0],
    ]).T
    query_trivial_2 = np.array([[0, 1]])
    path_trivial_2 = np.array([1, 0])
    # Only NONCOPY
    H_only_noncopy = np.array([
        [NC, NC, NC, NC, NC, NC, NC, NC, NC, NC],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    ]).T
    query_only_noncopy = np.array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
    path_only_noncopy = np.array([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1])
    # NONCOPY on right
    H_noncopy_on_right = np.array([
        [ 0,  0,  0,  0,  0, NC, NC, NC, NC, NC],
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    ]).T
    query_noncopy_on_right = np.array([[ 0,  0,  0,  0,  0,  1,  1,  1,  1,  1]])
    path_noncopy_on_right = np.array([ 0,  0,  0,  0,  0,  1,  1,  1,  1,  1])
    # NONCOPY on left
    H_noncopy_on_left = np.array([
        [NC, NC, NC, NC, NC,  0,  0,  0,  0,  0],
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    ]).T
    query_noncopy_on_left = np.array([[ 1,  1,  1,  1,  1,  0,  0,  0,  0,  0]])
    path_noncopy_on_left = np.array([ 1,  1,  1,  1,  1,  0,  0,  0,  0,  0])
    # NONCOPY in middle
    H_noncopy_middle = np.array([
        [NC, NC, NC,  0,  0,  0,  0, NC, NC, NC],
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    ]).T
    query_noncopy_middle = np.array([[ 1,  1,  1,  0,  0,  0,  0,  1,  1,  1]])
    path_noncopy_middle = np.array([ 1,  1,  1,  0,  0,  0,  0,  1,  1,  1])
    # Two switches
    H_two_switches = np.array([
        [ 0,  0,  0, NC, NC, NC, NC, NC, NC, NC],
        [NC, NC, NC,  0,  0,  0, NC, NC, NC, NC],
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    ]).T
    query_two_switches = np.array([[ 0,  0,  0,  0,  0,  0,  1,  1,  1,  1]])
    path_two_switches = np.array([ 0,  0,  0,  1,  1,  1,  2,  2,  2,  2])
    # MISSING at switch position
    # This causes more than one best paths
    H_miss_switch = np.array([
        [NC, NC, NC,  0,  0,  0,  0, NC, NC, NC],
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    ]).T
    query_miss_switch = np.array([[ 1,  1,  1, -1,  0,  0,  0,  1,  1,  1]])
    path_miss_switch = np.array([ 1,  1,  1,  1,  0,  0,  0,  1,  1,  1])
    # MISSING left of switch position
    H_miss_next_switch = np.array([
        [NC, NC, NC,  0,  0,  0,  0, NC, NC, NC],
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    ]).T
    query_next_switch = np.array([[ 1,  1, -1,  0,  0,  0,  0,  1,  1,  1]])
    path_next_switch = np.array([ 1,  1,  1,  0,  0,  0,  0,  1,  1,  1])

    return [
        (H_trivial_1, query_trivial_1, path_trivial_1),
        (H_trivial_2, query_trivial_2, path_trivial_2),
        (H_only_noncopy, query_only_noncopy, path_only_noncopy),
        (H_noncopy_on_right, query_noncopy_on_right, path_noncopy_on_right),
        (H_noncopy_on_left, query_noncopy_on_left, path_noncopy_on_left),
        (H_noncopy_middle, query_noncopy_middle, path_noncopy_middle),
        (H_two_switches, query_two_switches, path_two_switches),
        (H_miss_switch, query_miss_switch, path_miss_switch),
        (H_miss_next_switch, query_next_switch, path_next_switch),
    ]


# Tests for naive matrix-based implementation.
@pytest.mark.parametrize(
    "H, s, expected_path", get_example_data()
)
def test_forwards_viterbi_hap_naive(H, s, expected_path):
    m, n = H.shape
    assert m == s.shape[1] == len(expected_path)

    r = np.zeros(m, dtype=np.float64) + 0.20
    p_mutation = np.zeros(m, dtype=np.float64) + 0.10

    n_alleles = _get_num_alleles_per_site(H)
    e = _get_emission_probabilities(m, p_mutation, n_alleles)

    _, _, actual_ll = vh.forwards_viterbi_hap_naive(n, m, H, s, e, r)
    expected_ll = vh.path_ll_hap(n, m, H, expected_path, s, e, r)

    assert np.allclose(expected_ll, actual_ll)


# Tests for naive matrix-based implementation using numpy.
@pytest.mark.parametrize(
    "H, s, expected_path", get_example_data()
)
def test_forwards_viterbi_hap_naive_vec(H, s, expected_path):
    m, n = H.shape
    assert m == s.shape[1] == len(expected_path)

    r = np.zeros(m, dtype=np.float64) + 0.20
    p_mutation = np.zeros(m, dtype=np.float64) + 0.10

    n_alleles = _get_num_alleles_per_site(H)
    e = _get_emission_probabilities(m, p_mutation, n_alleles)

    _, _, actual_ll = vh.forwards_viterbi_hap_naive_vec(n, m, H, s, e, r)
    expected_ll = vh.path_ll_hap(n, m, H, expected_path, s, e, r)

    assert np.allclose(expected_ll, actual_ll)


# Tests for naive matrix-based implementation with reduced memory.
@pytest.mark.parametrize(
    "H, s, expected_path", get_example_data()
)
def test_forwards_viterbi_hap_naive_low_mem(H, s, expected_path):
    m, n = H.shape
    assert m == s.shape[1] == len(expected_path)

    r = np.zeros(m, dtype=np.float64) + 0.20
    p_mutation = np.zeros(m, dtype=np.float64) + 0.10

    n_alleles = _get_num_alleles_per_site(H)
    e = _get_emission_probabilities(m, p_mutation, n_alleles)

    _, _, actual_ll = vh.forwards_viterbi_hap_naive_low_mem(n, m, H, s, e, r)
    expected_ll = vh.path_ll_hap(n, m, H, expected_path, s, e, r)

    assert np.allclose(expected_ll, actual_ll), f"{expected_ll} {actual_ll}"


# Tests for naive matrix-based implementation with reduced memory and rescaling.
@pytest.mark.parametrize(
    "H, s, expected_path", get_example_data()
)
def test_forwards_viterbi_hap_naive_low_mem_rescaling(H, s, expected_path):
    m, n = H.shape
    assert m == s.shape[1] == len(expected_path)

    r = np.zeros(m, dtype=np.float64) + 0.20
    p_mutation = np.zeros(m, dtype=np.float64) + 0.10

    n_alleles = _get_num_alleles_per_site(H)
    e = _get_emission_probabilities(m, p_mutation, n_alleles)

    _, _, actual_ll = vh.forwards_viterbi_hap_naive_low_mem_rescaling(n, m, H, s, e, r)
    expected_ll = vh.path_ll_hap(n, m, H, expected_path, s, e, r)

    assert np.allclose(expected_ll, actual_ll)


# Tests for implementation with reduced memory and rescaling.
@pytest.mark.parametrize(
    "H, s, expected_path", get_example_data()
)
def test_forwards_viterbi_hap_low_mem_rescaling(H, s, expected_path):
    m, n = H.shape
    assert m == s.shape[1] == len(expected_path)

    r = np.zeros(m, dtype=np.float64) + 0.20
    p_mutation = np.zeros(m, dtype=np.float64) + 0.10

    n_alleles = _get_num_alleles_per_site(H)
    e = _get_emission_probabilities(m, p_mutation, n_alleles)

    _, _, actual_ll = vh.forwards_viterbi_hap_low_mem_rescaling(n, m, H, s, e, r)
    expected_ll = vh.path_ll_hap(n, m, H, expected_path, s, e, r)

    assert np.allclose(expected_ll, actual_ll)


# Tests for implementation with even more reduced memory and rescaling.
@pytest.mark.parametrize(
    "H, s, expected_path", get_example_data()
)
def test_forwards_viterbi_hap_lower_mem_rescaling(H, s, expected_path):
    m, n = H.shape
    assert m == s.shape[1] == len(expected_path)

    r = np.zeros(m, dtype=np.float64) + 0.20
    p_mutation = np.zeros(m, dtype=np.float64) + 0.10

    n_alleles = _get_num_alleles_per_site(H)
    e = _get_emission_probabilities(m, p_mutation, n_alleles)

    _, _, actual_ll = vh.forwards_viterbi_hap_lower_mem_rescaling(n, m, H, s, e, r)
    expected_ll = vh.path_ll_hap(n, m, H, expected_path, s, e, r)

    assert np.allclose(expected_ll, actual_ll)


# Tests for implementation with even more reduced memory and rescaling, without keeping pointers.
@pytest.mark.parametrize(
    "H, s, expected_path", get_example_data()
)
def test_forwards_viterbi_hap_lower_mem_rescaling_no_pointer(H, s, expected_path):
    m, n = H.shape
    assert m == s.shape[1] == len(expected_path)

    r = np.zeros(m, dtype=np.float64) + 0.20
    p_mutation = np.zeros(m, dtype=np.float64) + 0.10

    n_alleles = _get_num_alleles_per_site(H)
    e = _get_emission_probabilities(m, p_mutation, n_alleles)

    _, _, _, actual_ll = vh.forwards_viterbi_hap_lower_mem_rescaling_no_pointer(n, m, H, s, e, r)
    expected_ll = vh.path_ll_hap(n, m, H, expected_path, s, e, r)

    assert np.allclose(expected_ll, actual_ll)
