# lshmm
**lshmm** is a Python library for prototyping, experimenting, and testing implementations of algorithms using the Li & Stephens (2003) Hidden Markov Model.

## Usage

### Inputs
#### Data
* Sample and/or ancestral haplotypes comprising a reference panel.
* Query haplotypes.

In the haploid mode, the alleles in haplotypes can be represented by any integer value (besides `-1` and `-2`, which are special values). In the diploid mode, the genotypes (encoded as allele dosages) can be `0` (homozygous for the reference allele), `1` (heterozygous for the alternative allele), or `2` (homozygous for the alternative allele). Currently, multiallelic sites are supported in the haploid mode, but not the diploid mode.

Note that there are two special values `NONCOPY` and `MISSING`. `NONCOPY` (or `-2`) represent non-copiable states, and can only be found in partial ancestral haplotypes in the reference panel. `MISSING` (or `-1`) representing missing data, and can be found only in query haplotypes.

#### Parameters
* Recombination probabilities.
* Mutation probabilities.

### Models and algorithms
* Haploid LS HMM
  * Forward-backward algorithm
  * Viterbi algorithm
* Diploid LS HMM
  * Forward-backward algorithm
  * Viterbi algorithm

### Features
* Scaling of mutation rate by the number of distinct alleles per site.
* Non-copiable state in the reference panel (`NONCOPY`).
* Missing state in the query (`MISSING`).
* Multiallelic sites (haploid only).
