import unittest

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
        self.assertAlmostEqual(sum(get_haplotypes_probabilities(D, FREQ_A1, FREQ_B1).values()), 1.0, delta=0.0001)

    def test_haplotypes_probabilities_vs_frequencies(self):

        haplotypes_prob = {"a1_b1": 0.25,
                           "a1_b2": 0.55,
                           "a2_b1": 0.05,
                           "a2_b2": 0.15}

        haplotypes = get_haplotypes(haplotypes_prob, POPULATION_SIZE)

        self.assertAlmostEqual(haplotypes.count(11) / (2 * POPULATION_SIZE), haplotypes_prob["a1_b1"], delta=0.03)
        self.assertAlmostEqual(haplotypes.count(12) / (2 * POPULATION_SIZE), haplotypes_prob["a1_b2"], delta=0.03)
        self.assertAlmostEqual(haplotypes.count(21) / (2 * POPULATION_SIZE), haplotypes_prob["a2_b1"], delta=0.03)
        self.assertAlmostEqual(haplotypes.count(22) / (2 * POPULATION_SIZE), haplotypes_prob["a2_b2"], delta=0.03)

    def test_allele_freq_genotypes(self):

        file_haplotypes = open("./resources/haplotypes.txt", 'r')
        haplotypes = list(map(int, file_haplotypes.readline().split(',')))
        file_haplotypes.close()

        genotypes = get_genotypes(haplotypes, POPULATION_SIZE)

        self.assertAlmostEqual(1 - sum(genotypes[:, 0]) / (2 * POPULATION_SIZE), FREQ_A1, delta=0.03)
        self.assertAlmostEqual(1 - sum(genotypes[:, 1]) / (2 * POPULATION_SIZE), FREQ_B1, delta=0.03)

    def test_hardy_weinberg(self):

        file_haplotypes = open("./resources/haplotypes.txt", 'r')
        haplotypes = list(map(int, file_haplotypes.readline().split(',')))
        file_haplotypes.close()

        genotypes = get_genotypes(haplotypes, POPULATION_SIZE)

        mse_genotypes_a = np.mean((genotypes[:, 0] - np.mean(genotypes[:, 0])) ** 2)
        mse_genotypes_b = np.mean((genotypes[:, 1] - np.mean(genotypes[:, 1])) ** 2)

        self.assertAlmostEqual(mse_genotypes_a, 2 * FREQ_A1 * (1 - FREQ_A1), delta=0.03)
        self.assertAlmostEqual(mse_genotypes_b, 2 * FREQ_B1 * (1 - FREQ_B1), delta=0.03)

    def test_phenotypes_mean_sigma(self):

        file_genotypes = open("./resources/genotypes.txt", 'r')
        genotypes = np.loadtxt("./resources/genotypes.txt")
        file_genotypes.close()

        phenotypes = get_phenotypes(genotypes, BETA_A, BETA_B, POPULATION_SIZE)

        self.assertAlmostEqual(np.mean((phenotypes - np.mean(phenotypes)) ** 2), 1.0, delta=0.05)
