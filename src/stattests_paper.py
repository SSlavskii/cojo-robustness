from numpy.linalg import inv
from scipy.linalg import sqrtm

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

NUMBER_OF_ITERATIONS = 1000


def joint_test(gwas, population_size, ref_population_size, ref_r):

    # D_x_std = np.diag(np.diag(np.asmatrix(genotypes_std).transpose() * np.asmatrix(genotypes_std)))

    D_x_hw = np.matrix([[population_size, 0.0],
                        [0.0, population_size]])

    # D_w_std = D_x_std.copy()
    D_w_hw = np.asarray(np.matrix([[ref_population_size, 0.0],
                                   [0.0, ref_population_size]]))

    # print("D_w_hw = \n", D_w_hw)

    W_t_W = ref_population_size * np.matrix([[1.0, ref_r],
                                             [ref_r, 1.0]])
    # print("W_t_W = \n", W_t_W)

    # B_std = sqrtm(D_x_std) * inv(sqrtm(D_w_std)) * W_t_W * inv(sqrtm(D_w_std)) * sqrtm(D_x_std)
    # print("B_std = \n", B_std)
    B_hw = sqrtm(D_x_hw) * inv(sqrtm(D_w_hw)) * W_t_W * inv(sqrtm(D_w_hw)) * sqrtm(D_x_hw)
    # print("B_hw = \n", B_hw)

    beta_gwas = np.array(gwas["beta"])
    beta_gwas.shape = (2, 1)

    # joint_beta_std = inv(B_std) * D_x_std * beta_gwas
    # print("beta_joint_std = \n", joint_beta_std)
    joint_beta_hw = inv(B_hw) * D_x_hw * beta_gwas
    joint_beta_hw.shape = (1, 2)
    # print("beta_joint_hw =", joint_beta_hw)

    beta_gwas.shape = (1, 2)

    y_t_y = np.mean(np.multiply(D_w_hw.diagonal(), np.square(gwas["se"])) * (population_size - 1) +
                    np.multiply(D_w_hw.diagonal(), np.square(gwas["beta"])))

    beta_gwas.shape = (2, 1)
    sigma_sqr_joint = float((y_t_y - np.dot(np.dot(joint_beta_hw, D_w_hw), beta_gwas)) /
                            (population_size - NUMBER_OF_SNPS))

    # sigma_sqr_joint_ph = float((np.dot(phenotypes, phenotypes) - np.dot(np.dot(joint_beta_hw, D_w_hw), beta_gwas)) /
    #                            (POPULATION_SIZE - NUMBER_OF_SNPS))

    joint_se = np.sqrt(np.diag(sigma_sqr_joint * inv(B_hw)))
    # print("joint_se =", np.diag(joint_se))
    joint_beta_hw.shape = (1, 2)
    # print("z_scores =", np.divide(joint_beta_hw, np.diag(joint_se)))
    # print(float(np.dot(np.dot(joint_beta_hw, inv(np.diag(joint_se))), joint_beta_hw.T)))
    p_value_joint = chi2.sf(float(np.dot(np.dot(joint_beta_hw, inv(sigma_sqr_joint * inv(B_hw))), joint_beta_hw.T)), 2)
    return joint_beta_hw, joint_se, p_value_joint


def conditional_test(gwas, y, r_ref, ref_freq_a1, population_size, ref_population_size):
    B1 = 2 * ref_freq_a1 * (1 - ref_freq_a1) * ref_population_size * ref_population_size ** 2
    B2 = 2 * ref_freq_a1 * (1 - ref_freq_a1) * ref_population_size * ref_population_size ** 2
    D1_x = population_size
    D2_x = population_size
    c11 = 2 * FREQ_A1 * (1 - FREQ_A1) * population_size
    c22 = 2 * FREQ_B1 * (1 - FREQ_B1) * population_size
    c21 = 2 * population_size / ref_population_size * np.sqrt(FREQ_A1)
    C = np.matrix([[c11, c21],
                   [c21, c22]])

    con_beta21 = (1 / B2) * D2_x * gwas["beta"][2] - (1 / B2) * C * (1 / B1) * D1_x * gwas["beta"][1]
    sigma_sqr_cond = (np.dot(y, y) - 1.0 - con_beta21 * 1.0) / (population_size - 2)
    con_se = 0.0 * sigma_sqr_cond
    return con_beta21, con_se


"""
def main():
    logging.info("Simulating GWAS using following parameters: \n"
                 "\t POPULATION_SIZE = {population_size} \n"
                 "\t FREQ_A1 = {freq_a1} \n"
                 "\t FREQ_B1 = {freq_b1} \n"
                 "\t R = {r} \n"
                 "\t D = {d} \n"
                 "\t BETA_A = {beta_a} \n"
                 "\t BETA_B = {beta_b} \n"
                 "\t NUMBER_OF_ITERATIONS={number_of_iterations}".format(population_size=POPULATION_SIZE,
                                                                         freq_a1=FREQ_A1,
                                                                         freq_b1=FREQ_B1,
                                                                         r=R,
                                                                         d=D,
                                                                         beta_a=BETA_A,
                                                                         beta_b=BETA_B,
                                                                         number_of_iterations=NUMBER_OF_ITERATIONS))

    haplotypes_prob = get_haplotypes_probabilities(D, FREQ_A1, FREQ_B1)
    haplotypes = get_haplotypes(haplotypes_prob, POPULATION_SIZE)
    genotypes = get_genotypes(haplotypes, POPULATION_SIZE)
    genotypes_std = standardise_genotypes(genotypes, FREQ_A1, FREQ_B1)
    phenotypes = get_phenotypes(genotypes, BETA_A, BETA_B, POPULATION_SIZE)

    # print("G_std^T G_std = \n", np.asmatrix(genotypes_std).transpose() * np.asmatrix(genotypes_std))
    # print("G^T G = \n", np.asmatrix(genotypes).transpose() * np.asmatrix(genotypes))

    simulated_data = pd.DataFrame({"phenotype": phenotypes,
                                   "snp_a_gen": genotypes_std[:, 0],
                                   "snp_b_gen": genotypes_std[:, 1]})

    gwas = get_gwas(simulated_data)

    # gwas = simulate_gwas(POPULATION_SIZE, FREQ_A1, FREQ_B1, R, D, BETA_A, BETA_B)

    logging.info('Simulated GWAS: \n' + gwas.to_string())
    joint_test(gwas)
    return None


if __name__ == "__main__":
    main()
"""