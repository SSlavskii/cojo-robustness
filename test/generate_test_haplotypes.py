from src.simulation import *

POPULATION_SIZE = 100000

haplotypes_prob = {"a1_b1": 0.25,
                   "a1_b2": 0.55,
                   "a2_b1": 0.05,
                   "a2_b2": 0.15}

file_out = open("./resources/haplotypes.txt", 'w')

haplotypes = get_haplotypes(haplotypes_prob, POPULATION_SIZE)

print(haplotypes.count(11) / 200000)
print(haplotypes.count(12) / 200000)
print(haplotypes.count(21) / 200000)
print(haplotypes.count(22) / 200000)

for i in haplotypes:
    file_out.write(str(i) + ",")

file_out.close()
