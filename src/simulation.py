from src.plotting import *


POPULATION_SIZE = 100000

FREQ_A1 = 0.7
FREQ_B1 = 0.6

R = 0.7
D = R * np.sqrt(FREQ_A1 * (1 - FREQ_A1) * FREQ_B1 * (1 - FREQ_B1))

# Restrictions on D: -min(P_A * P_B, P_a * P_b) <= D <= min(P_A * P_b, P_a * P_B)

BETA_A = 0.15
BETA_B = 0.13

NUMBER_OF_ITERATIONS = 1000


def get_haplotypes_probabilities(d, freq_a1, freq_b1):

    freq_a2, freq_b2 = 1 - freq_a1, 1 - freq_b1

    haplotypes_prob = {"a1_b1": d + freq_a1 * freq_b1,
                       "a1_b2": -d + freq_a1 * freq_b2,
                       "a2_b1": -d + freq_a2 * freq_b1,
                       "a2_b2": d + freq_a2 * freq_b2}

    return haplotypes_prob


def get_haplotypes(haplotypes_prob, population_size):

    possible_haplotypes = [11, 12, 21, 22]
    haplotypes = np.random.choice(possible_haplotypes,
                                  size=2 * population_size,
                                  p=list(haplotypes_prob.values()))

    return haplotypes


def get_genotypes(haplotypes, population_size):

    genotypes_a = haplotypes[:population_size] // 10 + haplotypes[population_size:] // 10 - 2
    genotypes_b = haplotypes[:population_size] % 10 + haplotypes[population_size:] % 10 - 2

    return np.column_stack((genotypes_a, genotypes_b))


def standardise_genotypes(genotypes, freq_a1, freq_b1):

    genotypes_a = (genotypes[:, 0] - 2 * (1 - freq_a1)) / np.sqrt(2 * freq_a1 * (1 - freq_a1))
    genotypes_b = (genotypes[:, 1] - 2 * (1 - freq_b1)) / np.sqrt(2 * freq_b1 * (1 - freq_b1))

    return np.column_stack((genotypes_a, genotypes_b))


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

    # plot_simulated_data_with_regressions(simulated_data)

    model = smf.ols('phenotype ~ snp_a_gen + snp_b_gen', data=simulated_data).fit()

    return model.params.snp_a_gen / model.bse.snp_a_gen, model.params.snp_b_gen / model.bse.snp_b_gen


def main():

    joint_z1_z2 = {"z1": [], "z2": []}

    for i in range(NUMBER_OF_ITERATIONS):
        print(i)
        results = run(POPULATION_SIZE, FREQ_A1, FREQ_B1, D, BETA_A, BETA_B)
        joint_z1_z2["z1"].append(results[0])
        joint_z1_z2["z2"].append(results[1])

    joint_z1_z2 = pd.DataFrame.from_dict(joint_z1_z2)

    plot_joint_z1_z2(joint_z1_z2)

    return 0


if __name__ == "__main__":
    main()
