from src.simulation import *

POPULATION_SIZE = 100000

file_haplotypes = open("./resources/haplotypes.txt", 'r')
haplotypes = list(map(int, file_haplotypes.readline().split(',')))
file_haplotypes.close()

genotypes = get_genotypes(haplotypes, POPULATION_SIZE)

np.savetxt("./resources/genotypes.txt", genotypes)
