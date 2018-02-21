import unittest
import numpy as np
from scipy import stats

from src.simulation import *

# do not change, resources were generated using this values
POPULATION_SIZE = 100000
FREQ_A1 = 0.8
FREQ_B1 = 0.3
D = 0.01
BETA_A = 0.3
BETA_B = 0.2


class TestSimulations(unittest.TestCase):

    def test_get_haplotypes_probabilities(self):

        haplotypes_prob = get_haplotypes_probabilities(D, FREQ_A1, FREQ_B1)

        self.assertAlmostEqual(sum(haplotypes_prob.values()), 1.0, delta=np.finfo(float).eps)

        self.assertGreaterEqual(haplotypes_prob["a1_b1"], 0.0)
        self.assertGreaterEqual(haplotypes_prob["a1_b2"], 0.0)
        self.assertGreaterEqual(haplotypes_prob["a2_b1"], 0.0)
        self.assertGreaterEqual(haplotypes_prob["a2_b2"], 0.0)

    def test_haplotypes_probabilities_vs_frequencies(self):

        haplotypes_prob = {"a1_b1": 0.25,
                           "a1_b2": 0.55,
                           "a2_b1": 0.05,
                           "a2_b2": 0.15}

        haplotypes = get_haplotypes(haplotypes_prob, POPULATION_SIZE)

        # TODO: set delta as function of POPULATION_SIZE
        self.assertAlmostEqual(np.count_nonzero(haplotypes == 11) / (2 * POPULATION_SIZE),
                               haplotypes_prob["a1_b1"],
                               delta=0.03)
        self.assertAlmostEqual(np.count_nonzero(haplotypes == 12) / (2 * POPULATION_SIZE),
                               haplotypes_prob["a1_b2"],
                               delta=0.03)
        self.assertAlmostEqual(np.count_nonzero(haplotypes == 21) / (2 * POPULATION_SIZE),
                               haplotypes_prob["a2_b1"],
                               delta=0.03)
        self.assertAlmostEqual(np.count_nonzero(haplotypes == 22) / (2 * POPULATION_SIZE),
                               haplotypes_prob["a2_b2"],
                               delta=0.03)

    # TODO: test get_haplotypes with zero probability (D == 1)

    def test_allele_freq_genotypes(self):
        haplotypes = np.genfromtxt("./resources/haplotypes.txt", delimiter=',')
        genotypes = get_genotypes(haplotypes, POPULATION_SIZE)

        # TODO: set delta as function of POPULATION_SIZE
        self.assertAlmostEqual(1 - sum(genotypes[:, 0]) / (2 * POPULATION_SIZE), FREQ_A1, delta=0.03)
        self.assertAlmostEqual(1 - sum(genotypes[:, 1]) / (2 * POPULATION_SIZE), FREQ_B1, delta=0.03)

    def test_hardy_weinberg(self):
        haplotypes = np.genfromtxt("./resources/haplotypes.txt", delimiter=',')
        genotypes = get_genotypes(haplotypes, POPULATION_SIZE)

        mse_genotypes_a = np.mean((genotypes[:, 0] - np.mean(genotypes[:, 0])) ** 2)
        mse_genotypes_b = np.mean((genotypes[:, 1] - np.mean(genotypes[:, 1])) ** 2)

        # TODO: set delta as 5% of mse_genotypes_a(b)
        # TODO: try to use np.isclose() with absolute and relative tolerances

        self.assertAlmostEqual(mse_genotypes_a, 2 * FREQ_A1 * (1 - FREQ_A1), delta=0.02)
        self.assertAlmostEqual(mse_genotypes_b, 2 * FREQ_B1 * (1 - FREQ_B1), delta=0.02)

    def test_genotypes_mean_std(self):

        genotypes = np.loadtxt("./resources/genotypes.txt")

        genotypes_std = standardise_genotypes(genotypes, FREQ_A1, FREQ_B1)

        self.assertAlmostEqual(np.mean(genotypes_std[:, 0]), 0.0, delta=0.02)
        self.assertAlmostEqual(np.mean((genotypes_std[:, 0] - np.mean(genotypes_std[:, 0])) ** 2), 1.0, delta=0.02)

    def test_phenotypes_distribution(self):

        genotypes_std = np.loadtxt("./resources/genotypes_std.txt")

        phenotypes = get_phenotypes(genotypes_std, BETA_A, BETA_B, POPULATION_SIZE)

        self.assertGreaterEqual(stats.kstest(phenotypes, 'norm').pvalue, 0.05)
