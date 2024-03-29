import logging
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2


LOG_FILE = "../logs/cojo_simulations.log"

logging.basicConfig(format=u'%(filename)s[LINE:%(lineno)d]# %(levelname)-8s %(asctime)s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%H:%M:%S',
                    filename=LOG_FILE)


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
                                  p=[haplotypes_prob.get(haplotype)
                                     for haplotype in ["a1_b1", "a1_b2", "a2_b1", "a2_b2"]])

    return haplotypes


def get_genotypes(haplotypes, population_size):
    # -2 because AA - 0, AB - 1, BB=2
    genotypes_a = haplotypes[:population_size] // 10 + haplotypes[population_size:] // 10 - 2
    genotypes_b = haplotypes[:population_size] % 10 + haplotypes[population_size:] % 10 - 2

    return np.column_stack((genotypes_a, genotypes_b))


def standardise_genotypes(genotypes, freq_a1, freq_b1):

    genotypes_a = (genotypes[:, 0] - 2 * (1 - freq_a1)) / np.sqrt(2 * freq_a1 * (1 - freq_a1))
    genotypes_b = (genotypes[:, 1] - 2 * (1 - freq_b1)) / np.sqrt(2 * freq_b1 * (1 - freq_b1))
    return np.column_stack((genotypes_a, genotypes_b))


def standardise_genotypes_paper(genotypes, freq_a1, freq_b1):

    genotypes_a = genotypes[:, 0] - 2 * (1 - freq_a1)
    genotypes_b = genotypes[:, 1] - 2 * (1 - freq_b1)

    return np.column_stack((genotypes_a, genotypes_b))


def get_phenotypes(genotypes, beta_a, beta_b, population_size):
    sigma_err = 1.0 - beta_a ** 2 - beta_b ** 2
    phenotypes = genotypes[:, 0] * beta_a + genotypes[:, 1] * beta_b + np.sqrt(sigma_err) * np.random.normal(0, 1, population_size)
    return phenotypes


def check_correct_input(freq_a1, freq_b1, r, d):

    freq_a2 = 1 - freq_a1
    freq_b2 = 1 - freq_b1

    check_freq_a1 = 0.0 <= freq_a1 <= 1.0
    check_freq_b1 = 0.0 <= freq_b1 <= 1.0
    check_r = -1.0 <= r <= 1.0
    check_d = -min(freq_a1 * freq_b1, freq_a2 * freq_b2) <= d <= min(freq_a1 * freq_b2, freq_a2 * freq_b1)

    return check_freq_a1 and check_freq_b1 and check_r and check_d


def get_simulated_data(population_size, freq_a1, freq_b1, d, beta_a, beta_b):

    haplotypes_prob = get_haplotypes_probabilities(d, freq_a1, freq_b1)
    haplotypes = get_haplotypes(haplotypes_prob, population_size)
    genotypes = get_genotypes(haplotypes, population_size)
    genotypes_std = standardise_genotypes(genotypes, freq_a1, freq_b1)
    phenotypes = get_phenotypes(genotypes_std, beta_a, beta_b, population_size)

    # print("G_std^T G_std = \n", np.asmatrix(genotypes_std).transpose() * np.asmatrix(genotypes_std))
    # print("G^T G = \n", np.asmatrix(genotypes).transpose() * np.asmatrix(genotypes))

    simulated_data = pd.DataFrame({"phenotype": phenotypes,
                                   "snp_a_gen": genotypes_std[:, 0],
                                   "snp_b_gen": genotypes_std[:, 1]})

    return simulated_data


def get_gwas(simulated_data, freq_a1, freq_b1):

    model_a = smf.ols("phenotype ~ snp_a_gen", data=simulated_data).fit()
    model_b = smf.ols("phenotype ~ snp_b_gen", data=simulated_data).fit()

    model = smf.ols('phenotype ~ snp_a_gen + snp_b_gen', data=simulated_data).fit()
    # print(model.summary())

    gwas_dict = {"snp_num": [1, 2],
                 "freq1": [freq_a1, freq_b1],
                 "freq2": [1 - freq_a1, 1 - freq_b1],
                 "beta": [model_a.params.snp_a_gen, model_b.params.snp_b_gen],
                 "se": [model_a.bse.snp_a_gen, model_b.bse.snp_b_gen],
                 "p": [chi2.sf((model_a.params.snp_a_gen / model_a.bse.snp_a_gen) ** 2, 1),
                       chi2.sf((model_b.params.snp_b_gen / model_b.bse.snp_b_gen) ** 2, 1)]}

    gwas = pd.DataFrame.from_dict(gwas_dict)
    gwas = gwas[["snp_num", "freq1", "freq2", "beta", "se", "p"]]
    gwas["z_u"] = gwas["beta"] / gwas["se"]
    return gwas


def simulate_gwas(population_size, freq_a1, freq_b1, d, beta_a, beta_b):
    return get_gwas(get_simulated_data(population_size, freq_a1, freq_b1, d, beta_a, beta_b), freq_a1, freq_b1)


def run(population_size, freq_a1, freq_b1, r, d, beta_a, beta_b):

    if not check_correct_input(freq_a1, freq_b1, r, d):
        print("Input error!")
        exit()

    simulated_data = get_simulated_data(population_size, freq_a1, freq_b1, d, beta_a, beta_b)

    # plot_simulated_data_with_regressions(simulated_data)
    # model = smf.ols('phenotype ~ snp_a_gen + snp_b_gen', data=simulated_data).fit()

    model_a = smf.ols('phenotype ~ snp_a_gen', data=simulated_data).fit()
    model_b = smf.ols('phenotype ~ snp_b_gen', data=simulated_data).fit()

    # X = simulated_data[["snp_a_gen", "snp_b_gen"]]
    # y = simulated_data["phenotype"]

    # model = sm.OLS(y, X).fit()

    # return model.params.snp_a_gen / model.bse.snp_a_gen, model.params.snp_b_gen / model.bse.snp_b_gen
    return model_a.params.snp_a_gen / model_a.bse.snp_a_gen, model_b.params.snp_b_gen / model_b.bse.snp_b_gen


"""
def main():
    
    joint_z1_z2 = {"z1": [], "z2": []}

    for i in range(NUMBER_OF_ITERATIONS):
        print(i + 1)
        results = run(POPULATION_SIZE, FREQ_A1, FREQ_B1, R, D, BETA_A, BETA_B)
        joint_z1_z2["z1"].append(results[0])
        joint_z1_z2["z2"].append(results[1])

    joint_z1_z2 = pd.DataFrame.from_dict(joint_z1_z2)

    plot_joint_z1_z2(joint_z1_z2)

    simulated_data = get_simulated_data(POPULATION_SIZE, FREQ_A1, FREQ_B1, R, D, BETA_A, BETA_B)
    gwas = get_gwas(simulated_data)

    return 0


if __name__ == "__main__":
    main()
"""
