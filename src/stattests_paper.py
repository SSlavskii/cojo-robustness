from src.simulation import *
from numpy.linalg import inv
from scipy.linalg import sqrtm

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


def joint_test(gwas, genotypes_std, phenotypes):

    D_x_std = np.diag(np.diag(np.asmatrix(genotypes_std).transpose() * np.asmatrix(genotypes_std)))
    # print("D_x_std = \n", D_x_std)
    D_x_hw = np.matrix([[POPULATION_SIZE, 0.0],
                        [0.0, POPULATION_SIZE]])
    # print("D_x_hw = \n", D_x_hw)

    D_w_std = D_x_std.copy()
    D_w_hw = np.matrix([[POPULATION_SIZE, 0.0],
                        [0.0, POPULATION_SIZE]])
    # print("D_w_hw = \n", D_w_hw)

    W_t_W = REF_POPULATION_SIZE * np.matrix([[1.0, REF_R],
                                             [REF_R, 1.0]])
    # print("W_t_W = \n", W_t_W)

    B_std = sqrtm(D_x_std) * inv(sqrtm(D_w_std)) * W_t_W * inv(sqrtm(D_w_std)) * sqrtm(D_x_std)
    # print("B_std = \n", B_std)
    B_hw = sqrtm(D_x_hw) * inv(sqrtm(D_w_hw)) * W_t_W * inv(sqrtm(D_w_hw)) * sqrtm(D_x_hw)
    # print("B_hw = \n", B_hw)

    beta_gwas = np.array(gwas["beta"])
    beta_gwas.shape = (2, 1)

    joint_beta_std = inv(B_std) * D_x_std * beta_gwas
    print("beta_joint_std = \n", joint_beta_std)
    joint_beta_hw = inv(B_hw) * D_x_hw * beta_gwas
    print("beta_joint_hw = \n", joint_beta_hw)

    beta_gwas.shape = (1, 2)
    sigma_joint_sqr = (np.dot(phenotypes, phenotypes) - np.dot(joint_beta_hw, beta_gwas)) / \
                      (POPULATION_SIZE - NUMBER_OF_SNPS)
    joint_se = sigma_joint_sqr * inv(B_hw)
    print(np.diag(joint_se))
    joint_beta_hw.shape = (1, 2)
    print(np.divide(joint_beta_hw, np.diag(joint_se)))
    return joint_beta_std, joint_se


def conditional_test(gwas, y, r_ref):
    B1 = 0.0
    B2 = 0.0
    D1 = 0.0
    D2 = 0.0
    c11 = 0.0
    c22 = 0.0
    c21 = 0.0
    C = np.matrix([[c11, c21],
                   [c21, c22]])
    sigma_sqr_cond = (np.dot(y, y) - 1.0) / (POPULATION_SIZE - 2)
    con_beta21 = inv(B2) * D2 * gwas["beta"][2] - inv(B2) * C * inv(B1) * D1 * gwas["beta"][1]
    con_se = 0.0 * sigma_sqr_cond
    return con_beta21, con_se


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
    joint_test(gwas, genotypes_std, phenotypes)
    return None


if __name__ == "__main__":
    main()
