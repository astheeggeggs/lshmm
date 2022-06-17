import itertools

import msprime
import numpy as np
from IPython.display import SVG, display

EQUAL_BOTH_HOM = 4
UNEQUAL_BOTH_HOM = 0
BOTH_HET = 7
REF_HOM_OBS_HET = 1
REF_HET_OBS_HOM = 2

import lshmm.forward_backward.fb_diploid_variants_samples as fbd_vs
import lshmm.forward_backward.fb_haploid_variants_samples as fbh_vs

# Create a set of haplotypes (tree with 10 leaves say, just use msprime without recombination).
# Pull out a pair of haplotypes from this.


def simple_n_10_no_recombination():

    ts = msprime.simulate(12, recombination_rate=0, mutation_rate=0.5, random_seed=40)
    assert ts.num_sites > 3

    H = ts.genotype_matrix()
    s = H[:, 0].reshape(1, H.shape[0]) + H[:, 1].reshape(1, H.shape[0])
    H = H[:, 2:]

    m = ts.get_num_sites()
    n = H.shape[1]

    G = np.zeros((m, 10, 10))
    for i in range(m):
        G[i, :, :] = np.add.outer(H[i, :], H[i, :])

    ts = ts.simplify(range(2, 10 + 2), filter_sites=False)

    return ts, G, H, s, m, n


def genotype_emission(mu, m):
    # Define the emission probability matrix
    e = np.zeros((m, 8))
    e[:, EQUAL_BOTH_HOM] = (1 - mu) ** 2
    e[:, UNEQUAL_BOTH_HOM] = mu ** 2
    e[:, BOTH_HET] = 1 - mu
    e[:, REF_HOM_OBS_HET] = 2 * mu * (1 - mu)
    e[:, REF_HET_OBS_HOM] = mu * (1 - mu)

    return e


r = np.zeros(m) + 0.01
mu = np.zeros(m) + 0.01
r[0] = 0
e = genotype_emission(mu, m)

# plot the tree
ts, G, H, s, m, n = simple_n_10_no_recombination()

# label the tree with the mutations and the likelihoods
F, c, ll = fbd_vs.forward_ls_dip_loop(n, m, G, s, e, r, norm=True)
# Move to the next site and do the same.

# Look for patterns
display(SVG(ts.draw_svg(size=(980, 600))))

plt.figure(figsize=(20, 20))

for i in range(m):
    plt.subplot(1, m, i + 1)
    F_sub = F[i, :, :]
    F_sub = F_sub[:, [0, 1, 5, 6, 2, 4, 3, 9, 7, 8]]
    F_sub = F_sub[[0, 1, 5, 6, 2, 4, 3, 9, 7, 8], :]
    plt.imshow(np.log10(F_sub), cmap="hot", interpolation="nearest")

plt.show()

F, c, ll = fbh_vs.forwards_ls_hap(n, m, H, s, e, r, norm=True)

for i in range(m):
    plt.subplot(1, m, i + 1)
    F_sub = F[i, [0, 1, 5, 6, 2, 4, 3, 9, 7, 8]].reshape(n, 1)
    plt.imshow(np.log10(F_sub), cmap="hot", interpolation="nearest")

plt.show()
