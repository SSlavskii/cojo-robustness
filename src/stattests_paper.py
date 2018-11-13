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

    d_x_hw = np.matrix([[population_size, 0.0],
                        [0.0, population_size]])

    d_w_hw = np.asarray(np.matrix([[ref_population_size, 0.0],
                                   [0.0, ref_population_size]]))

    w_t_w = ref_population_size * np.matrix([[1.0, ref_r],
                                             [ref_r, 1.0]])

    b_hw = sqrtm(d_x_hw) * inv(sqrtm(d_w_hw)) * w_t_w * inv(sqrtm(d_w_hw)) * sqrtm(d_x_hw)

    beta_gwas = np.array(gwas["beta"])
    beta_gwas.shape = (NUMBER_OF_SNPS, 1)

    joint_beta_hw = inv(b_hw) * d_x_hw * beta_gwas
    joint_beta_hw.shape = (1, NUMBER_OF_SNPS)

    beta_gwas.shape = (1, NUMBER_OF_SNPS)

    y_t_y = np.mean(np.multiply(d_w_hw.diagonal(), np.square(gwas["se"])) * (population_size - 1) +
                    np.multiply(d_w_hw.diagonal(), np.square(gwas["beta"])))

    beta_gwas.shape = (NUMBER_OF_SNPS, 1)
    sigma_sqr_joint = float((y_t_y - np.dot(np.dot(joint_beta_hw, d_w_hw), beta_gwas)) /
                            (population_size - NUMBER_OF_SNPS))

    joint_se = np.sqrt(np.diag(sigma_sqr_joint * inv(b_hw)))
    joint_beta_hw.shape = (1, 2)

    p_value_joint = chi2.sf(float(np.dot(np.dot(joint_beta_hw, inv(sigma_sqr_joint * inv(b_hw))), joint_beta_hw.T)), 2)

    return joint_beta_hw, joint_se, p_value_joint


def conditional_test(gwas, b_prev_est_1, ref_r, ref_freq_a1, population_size, ref_population_size):
    b1 = 2 * ref_freq_a1 * (1 - ref_freq_a1) * ref_population_size * ref_population_size ** 2
    b2 = 2 * ref_freq_a1 * (1 - ref_freq_a1) * ref_population_size * ref_population_size ** 2
    d1_x = population_size
    d2_x = population_size
    d_w_hw = np.asarray(np.matrix([[ref_population_size, 0.0],
                                   [0.0, ref_population_size]]))
    w_t_w = ref_population_size * np.matrix([[1.0, ref_r],
                                             [ref_r, 1.0]])
    c11 = 2 * FREQ_A1 * (1 - FREQ_A1) * population_size
    c22 = 2 * FREQ_B1 * (1 - FREQ_B1) * population_size
    c21 = 2 * population_size / ref_population_size * np.sqrt(FREQ_A1)
    c = np.matrix([[c11, c21],
                   [c21, c22]])

    cond_beta21 = (1 / b2) * d2_x * gwas["beta"][2] - (1 / b2) * c * (1 / b1) * d1_x * gwas["beta"][1]

    y_t_y = np.mean(np.multiply(d_w_hw.diagonal(), np.square(gwas["se"])) * (population_size - 1) +
                    np.multiply(d_w_hw.diagonal(), np.square(gwas["beta"])))

    sigma_sqr_cond = (y_t_y - b_prev_est_1 * d1_x * gwas["beta"][1] - cond_beta21 * d2_x * gwas["beta"][2]) / \
                     (population_size - 2)

    con_se = np.sqrt(sigma_sqr_cond / b2 - sigma_sqr_cond / b2 * c / b1 * c.T / b2)

    # p_value_cond = chi2.sf(float(np.dot(np.dot(cond_beta21, inv(sigma_sqr_cond * inv(b_hw))), cond_beta21.T)), 2)
    return cond_beta21, con_se


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
