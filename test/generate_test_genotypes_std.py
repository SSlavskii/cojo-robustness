from src.simulation import *

POPULATION_SIZE = 100000
FREQ_A1 = 0.8
FREQ_B1 = 0.3

genotypes = np.loadtxt("./resources/genotypes.txt")

genotypes_std = standardise_genotypes(genotypes, FREQ_A1, FREQ_B1)

np.savetxt("./resources/genotypes_std.txt", genotypes_std)
