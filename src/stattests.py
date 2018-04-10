from src.simulation import *

POPULATION_SIZE = 100000

FREQ_A1 = 0.7
FREQ_B1 = 0.6

R = 0.7
D = R * np.sqrt(FREQ_A1 * (1 - FREQ_A1) * FREQ_B1 * (1 - FREQ_B1))
# Restrictions on D: -min(P_A * P_B, P_a * P_b) <= D <= min(P_A * P_b, P_a * P_B)

BETA_A = 0.15
BETA_B = 0.13

NUMBER_OF_ITERATIONS = 1000

def joint_test()

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
    gwas["z_m"] = gwas["z_u"] * R
    logging.info('Simulated GWAS: \n' + gwas.to_string())

    return None


if __name__ == "__main__":
    main()
