from src.simulation import *

population_size = 100000

haplotypes_prob = {"a1_b1": 0.2,
                   "a1_b2": 0.3,
                   "a2_b1": 0.4,
                   "a2_b2": 0.1}

file_out = open("./resources/haplotypes.txt", 'w')

haplotypes = get_haplotypes(haplotypes_prob, population_size)

print(haplotypes.count(11) / 200000)
print(haplotypes.count(12) / 200000)
print(haplotypes.count(21) / 200000)
print(haplotypes.count(22) / 200000)

for i in haplotypes:
    file_out.write(str(i) + ",")

file_out.close()
