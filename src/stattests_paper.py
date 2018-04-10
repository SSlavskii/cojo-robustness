from src.simulation import *
from numpy.linalg import inv
from scipy.linalg import sqrtm


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


def joint_test(gwas):

    D_x = np.matrix([[2 * FREQ_A1 * (1 - FREQ_A1) * POPULATION_SIZE, 0.0],
                    [0.0, 2 * FREQ_B1 * (1 - FREQ_B1) * POPULATION_SIZE]])
    print("D_x = \n", D_x)

    D_w = np.matrix([[2 * REF_FREQ_A1 * (1 - REF_FREQ_A1) * REF_POPULATION_SIZE, 0.0],
                    [0.0, 2 * REF_FREQ_B1 * (1 - REF_FREQ_B1) * REF_POPULATION_SIZE]])
    print("D_w = \n", D_w)

    W_t_W = REF_POPULATION_SIZE * np.matrix([[1.0, REF_R],
                                             [REF_R, 1.0]])
    print("W_t_W = \n", W_t_W)

    B = sqrtm(D_x) * inv(sqrtm(D_w)) * W_t_W * inv(sqrtm(D_w)) * sqrtm(D_x)
    print("B = \n", B)

    sigma_joint_sqr = 0.0

    beta = np.array(gwas["beta"])
    beta.shape = (2, 1)

    joint_beta = inv(B) * D_x * beta

    print(joint_beta)

    joint_se = sigma_joint_sqr * inv(B)
    return joint_beta, joint_se


"""
def conditional_test(gwas, y, r_ref,):
    B1 =
    B2 =
    D1 =
    D2 =
    c11 =
    c22 =
    c21 =
    C = np.matrix([[c11, c21],
                   [c21, c22]])
    sigma_sqr_cond = (np.dot(y, y) - ) / (POPULATION_SIZE - 2)
    con_beta21 = inv(B2) D2 gwas["beta"][2] - inv(B2) C inv(B1) D1 gwas["beta"][1]
    con_se = 0.0
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

    gwas = simulate_gwas(POPULATION_SIZE, FREQ_A1, FREQ_B1, R, D, BETA_A, BETA_B)

    logging.info('Simulated GWAS: \n' + gwas.to_string())
    joint_test(gwas)
    return None


if __name__ == "__main__":
    main()
