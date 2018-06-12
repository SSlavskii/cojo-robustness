import pandas as pd
from src.simulation import *

NUMBER_OF_SNPS = 2
POPULATION_SIZE = 100000
REF_POPULATION_SIZE = 100000

FREQ_A1 = 0.7
FREQ_B1 = 0.6

REF_FREQ_A1 = 0.7
REF_FREQ_B1 = 0.6

R = 0.7
REF_R = 0.7

D = R * np.sqrt(FREQ_A1 * (1 - FREQ_A1) * FREQ_B1 * (1 - FREQ_B1))
REF_D = REF_R * np.sqrt(REF_FREQ_A1 * (1 - REF_FREQ_A1) * REF_FREQ_B1 * (1 - REF_FREQ_B1))
# Restrictions on D: -min(P_A * P_B, P_a * P_b) <= D <= min(P_A * P_b, P_a * P_B)

BETA_A = 0.2
BETA_B = 0.2


def generate_ped(genotypes, phenotypes):
    ped_data = []
    return ped_data


def main():
    haplotypes_prob = get_haplotypes_probabilities(D, FREQ_A1, FREQ_B1)
    haplotypes = get_haplotypes(haplotypes_prob, POPULATION_SIZE)
    genotypes = get_genotypes(haplotypes, POPULATION_SIZE)
    genotypes_std = standardise_genotypes(genotypes, FREQ_A1, FREQ_B1)
    phenotypes = get_phenotypes(genotypes, BETA_A, BETA_B, POPULATION_SIZE)

    print(type(genotypes_std))
    print(type(phenotypes))


if __name__ == "__main__":
    main()