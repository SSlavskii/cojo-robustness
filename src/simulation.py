import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import statsmodels.api as sm
import statsmodels.formula.api as smf


def get_haplotypes_probabilities(d, freq_a1, freq_b1):

    freq_a2, freq_b2 = 1 - freq_a1, 1 - freq_b1

    haplotypes_prob = {"a1_b1": d + freq_a1 * freq_b1,
                       "a1_b2": -d + freq_a1 * freq_b2,
                       "a2_b1": -d + freq_a2 * freq_b1,
                       "a2_b2": d + freq_a2 * freq_b2}

    return haplotypes_prob


def get_haplotypes(haplotypes_prob, population_size):

    haplotypes = []

    for counter in range(2 * population_size):

        x = float(np.random.uniform(0, 1, 1))
        if x < haplotypes_prob["a1_b1"]:
            haplotypes.append(11)
        elif x < haplotypes_prob["a1_b1"] + haplotypes_prob["a1_b2"]:
            haplotypes.append(12)
        elif x < haplotypes_prob["a1_b1"] + haplotypes_prob["a1_b2"] + haplotypes_prob["a2_b1"]:
            haplotypes.append(21)
        else:
            haplotypes.append(22)

    return haplotypes


def get_genotypes(haplotypes, population_size):

    genotypes = []

    for i in range(population_size):
        genotype = str(haplotypes[i] // 10 + haplotypes[i + population_size] // 10 - 2)
        genotype += str(haplotypes[i] % 10 + haplotypes[i + population_size] % 10 - 2)
        genotypes.append(genotype)

    return np.array([list(i) for i in genotypes], dtype='int')


def get_phenotypes(genotypes, beta_a, beta_b, population_size):

    mse_genotypes_a = np.mean((genotypes[:, 0] - np.mean(genotypes[:, 0])) ** 2)
    mse_genotypes_b = np.mean((genotypes[:, 1] - np.mean(genotypes[:, 1])) ** 2)

    sigma_err = 1.0 - mse_genotypes_a * beta_a ** 2 - mse_genotypes_b * beta_b ** 2

    phenotypes = genotypes[:, 0] * beta_a + genotypes[:, 1] * beta_b + np.random.normal(0,
                                                                                        np.sqrt(sigma_err),
                                                                                        population_size)
    return phenotypes


def run(population_size, freq_a1, freq_b1, d, beta_a, beta_b):

    haplotypes_prob = get_haplotypes_probabilities(d, freq_a1, freq_b1)
    haplotypes = get_haplotypes(haplotypes_prob, population_size)
    genotypes = get_genotypes(haplotypes, population_size)
    phenotypes = get_phenotypes(genotypes, beta_a, beta_b, population_size)

    simulated_data = pd.DataFrame({"phenotype": phenotypes,
                                   "snp_a_gen": genotypes[:, 0],
                                   "snp_b_gen": genotypes[:, 1]})

    model = smf.ols('phenotype ~ snp_a_gen + snp_b_gen', data=simulated_data).fit()

    return model.params.snp_a_gen, model.params.snp_b_gen


def main():

    population_size = 10000

    freq_a1 = 0.7
    freq_b1 = 0.6

    r = 0.7
    d = r * np.sqrt(freq_a1 * (1 - freq_a1) * freq_b1 * (1 - freq_b1))

    # Restrictions on D: -min(P_A * P_B, P_a * P_b) <= D <= min(P_A * P_b, P_a * P_B)

    beta_a = 0.15
    beta_b = 0.13

    file_out = open("../out/1000_iter_beta1_beta2.csv", 'w')

    for i in range(1000):
        print(i)
        results = run(population_size, freq_a1, freq_b1, d, beta_a, beta_b)
        file_out.write(str(results[0]) + ',' + str(results[1]) + '\n')

    file_out.close()
    return 0


if __name__ == "__main__":
    main()
