from src.simulation import *

POPULATION_SIZE = 100000

file_haplotypes = open("./resources/haplotypes.txt", 'r')
haplotypes = list(map(int, file_haplotypes.readline().split(',')))
file_haplotypes.close()

file_out = open("./resources/genotypes.txt", 'w')
genotypes = get_genotypes(haplotypes, POPULATION_SIZE)

for i in genotypes:
    file_out.write(str(i[0]) + ',' + str(i[1]) + '\n')

file_out.close()
