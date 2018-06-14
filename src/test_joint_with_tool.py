import os
from src.stattests_paper import *


LOG_FILE = "../logs/cojo_simulations.log"
logging.basicConfig(format=u'%(filename)s[LINE:%(lineno)d]# %(levelname)-8s %(asctime)s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%H:%M:%S',
                    filename=LOG_FILE)

NUMBER_OF_SNPS = 2
POPULATION_SIZE = 100000
REF_POPULATION_SIZE = 100000

FREQ_A1 = 0.7
FREQ_B1 = 0.6

REF_FREQ_A1 = 0.7
REF_FREQ_B1 = 0.6

R = 0.6
D = R * np.sqrt(FREQ_A1 * (1 - FREQ_A1) * FREQ_B1 * (1 - FREQ_B1))

# REF_R = 0.5
COEFF_R_D = np.sqrt(REF_FREQ_A1 * (1 - REF_FREQ_A1) * REF_FREQ_B1 * (1 - REF_FREQ_B1))
# REF_D = REF_R * COEFF_R_D
# Restrictions on D: -min(P_A * P_B, P_a * P_b) <= D <= min(P_A * P_b, P_a * P_B)

BETA_A = 0.04
BETA_B = -0.02


def generate_ped(genotypes, phenotypes):
    ped_data_dict = {"FID": [], "IID": [], "FatherID": [], "MotherID": [],
                     "Sex": [], "Phenotype": [],
                     "SNP1_1": [], "SNP1_2": [], "SNP2_1": [], "SNP2_2": []}

    for i in range(len(phenotypes)):
        ped_data_dict["FID"].append(0)
        ped_data_dict["IID"].append(i + 1)
        ped_data_dict["FatherID"].append(0)
        ped_data_dict["MotherID"].append(0)
        ped_data_dict["Sex"].append(0)
        ped_data_dict["Phenotype"].append(phenotypes[i])

        if genotypes[i][0] == 0:
            letter1 = "AA"
        elif genotypes[i][0] == 1:
            letter1 = "AT"
        else:
            letter1 = "TT"

        if genotypes[i][1] == 0:
            letter2 = "AA"
        elif genotypes[i][1] == 1:
            letter2 = "AT"
        else:
            letter2 = "TT"

        ped_data_dict["SNP1_1"].append(letter1[0])
        ped_data_dict["SNP1_2"].append(letter1[1])
        ped_data_dict["SNP2_1"].append(letter2[0])
        ped_data_dict["SNP2_2"].append(letter2[1])

    ped_data = pd.DataFrame.from_dict(ped_data_dict)
    ped_data = ped_data[['FID', 'IID', 'Phenotype', 'SNP1_1', 'SNP1_2', 'SNP2_1', 'SNP2_2']]

    return ped_data


def main():

    plotting_data = {"beta1": [], "se1": [], "p1": [],
                     "beta2": [], "se2": [], "p2": [],
                     "beta1_tool": [], "se1_tool": [], "p1_tool": [],
                     "beta2_tool": [], "se2_tool": [], "p2_tool": [],
                     "r": []}
    r_iter = np.linspace(0.2, 0.85, 30)
    for REF_R in r_iter:
        REF_D = REF_R * COEFF_R_D
        print("REF_R =", REF_R)
        for i in range(5):
            print("\ti =", i + 1)
            gwas = simulate_gwas(POPULATION_SIZE, FREQ_A1, FREQ_B1, D, BETA_A, BETA_B)
            joint_beta, joint_se = joint_test(gwas=gwas,
                                              population_size=POPULATION_SIZE,
                                              ref_population_size=REF_POPULATION_SIZE,
                                              ref_r=REF_R)
            joint_beta = list(joint_beta.flat)

            gwas['A1'] = ['A', 'A']
            gwas['A2'] = ['T', 'T']
            gwas['N'] = [POPULATION_SIZE, POPULATION_SIZE]

            ma_data = gwas[["snp_num", "A1", "A2", "freq1", "beta", "se", "p", "N"]]
            ma_data.to_csv("../data/cojo_tool_test_run.ma", sep='\t', header=True, index=False)

            haplotypes_prob_ref = get_haplotypes_probabilities(REF_D, REF_FREQ_A1, REF_FREQ_B1)
            haplotypes_ref = get_haplotypes(haplotypes_prob_ref, REF_POPULATION_SIZE)
            genotypes_ref = get_genotypes(haplotypes_ref, REF_POPULATION_SIZE)
            # genotypes_std_ref = standardise_genotypes(genotypes_ref, REF_FREQ_A1, REF_FREQ_B1)
            phenotypes_ref = get_phenotypes(genotypes_ref, BETA_A, BETA_B, REF_POPULATION_SIZE)

            ped_data = generate_ped(genotypes_ref, phenotypes_ref)
            ped_data.to_csv("../data/cojo_tool_test_run.ped", sep='\t', header=False, index=False)

            os.system("../tools/plink-1.07-x86_64/plink --file ../data/cojo_tool_test_run "
                      "--make-bed --out ../data/cojo_tool_test_run --noweb --no-parents --no-sex > /dev/null")
            os.system("../tools/gcta/gcta64 --bfile ../data/cojo_tool_test_run "
                      "--cojo-file ../data/cojo_tool_test_run.ma --cojo-joint --out ../data/test_out > /dev/null")

            file_gcta_out = open("../data/test_out.jma.cojo")
            file_gcta_out.readline()
            snp1 = file_gcta_out.readline()
            snp2 = file_gcta_out.readline()

            plotting_data["r"].append(REF_R)
            plotting_data["beta1"].append(joint_beta[0])
            plotting_data["beta2"].append(joint_beta[1])
            plotting_data["se1"].append(joint_se[0])
            plotting_data["se2"].append(joint_se[1])
            plotting_data["p1"].append(chi2.sf((joint_beta[0] / joint_se[0]) ** 2, 1))
            plotting_data["p2"].append(chi2.sf((joint_beta[1] / joint_se[1]) ** 2, 1))
            plotting_data["beta1_tool"].append(float(snp1.split('\t')[10]))
            plotting_data["beta2_tool"].append(float(snp2.split('\t')[10]))
            plotting_data["se1_tool"].append(float(snp1.split('\t')[11]))
            plotting_data["se2_tool"].append(float(snp2.split('\t')[11]))
            plotting_data["p1_tool"].append(float(snp1.split('\t')[12]))
            plotting_data["p2_tool"].append(float(snp2.split('\t')[12]))

    print(gwas.head())
    plotting_data = pd.DataFrame.from_dict(plotting_data)
    plotting_data = plotting_data[['r',
                                   'beta1', 'se1', 'p1',
                                   'beta2', 'se2', 'p2',
                                   'beta1_tool', 'se1_tool', 'p1_tool',
                                   'beta2_tool', 'se2_tool', 'p2_tool']]
    print(plotting_data.head())
    plotting_data.to_csv("../data/test_diff_r.tsv", sep='\t', header=True, index=False)

    """     
    logging.info("\n********************************************** \n"
                 "Simulating GWAS using following parameters: \n"
                 "\t POPULATION_SIZE = {population_size} \n"
                 "\t FREQ_A1 = {freq_a1} \n"
                 "\t FREQ_B1 = {freq_b1} \n"
                 "\t R = {r} \n"
                 "\t D = {d} \n"
                 "\t BETA_A = {beta_a} \n"
                 "\t BETA_B = {beta_b} \n".format(population_size=POPULATION_SIZE,
                                                  freq_a1=FREQ_A1,
                                                  freq_b1=FREQ_B1,
                                                  r=R,
                                                  d=D,
                                                  beta_a=BETA_A,
                                                  beta_b=BETA_B))

    gwas = simulate_gwas(POPULATION_SIZE, FREQ_A1, FREQ_B1, D, BETA_A, BETA_B)
    logging.info("\nSimulated summary statistics: \n\t {gwas}\n".format(gwas=gwas))
    joint_beta, joint_se = joint_test(gwas=gwas,
                                      population_size=POPULATION_SIZE,
                                      ref_population_size=REF_POPULATION_SIZE,
                                      ref_r=REF_R)
    joint_beta = list(joint_beta.flat)
    logging.info("\nJoint test results: "
                 "\n\tjoint_beta={joint_beta} \n\tjoint_se={joint_se}\n".format(joint_beta=joint_beta,
                                                                                joint_se=np.diag(joint_se)))

    gwas['A1'] = ['A', 'A']
    gwas['A2'] = ['T', 'T']
    gwas['N'] = [POPULATION_SIZE, POPULATION_SIZE]

    ma_data = gwas[["snp_num", "A1", "A2", "freq1", "beta", "se", "p", "N"]]
    ma_data.to_csv("../data/cojo_tool_test_run.ma", sep='\t', header=True, index=False)

    logging.info("\nSimulating reference using following parameters: \n"
                 "\t REF_POPULATION_SIZE = {ref_population_size} \n"
                 "\t REF_FREQ_A1 = {ref_freq_a1} \n"
                 "\t REF_FREQ_B1 = {ref_freq_b1} \n"
                 "\t REF_R = {ref_r} \n"
                 "\t REF_D = {ref_d} \n".format(ref_population_size=REF_POPULATION_SIZE,
                                                ref_freq_a1=REF_FREQ_A1,
                                                ref_freq_b1=REF_FREQ_B1,
                                                ref_r=REF_R,
                                                ref_d=REF_D))
    
    haplotypes_prob_ref = get_haplotypes_probabilities(REF_D, REF_FREQ_A1, REF_FREQ_B1)
    haplotypes_ref = get_haplotypes(haplotypes_prob_ref, REF_POPULATION_SIZE)
    genotypes_ref = get_genotypes(haplotypes_ref, REF_POPULATION_SIZE)
    # genotypes_std_ref = standardise_genotypes(genotypes_ref, REF_FREQ_A1, REF_FREQ_B1)
    phenotypes_ref = get_phenotypes(genotypes_ref, BETA_A, BETA_B, REF_POPULATION_SIZE)

    ped_data = generate_ped(genotypes_ref, phenotypes_ref)
    ped_data.to_csv("../data/cojo_tool_test_run.ped", sep='\t', header=False, index=False)

    os.system("../tools/plink-1.07-x86_64/plink --file ../data/cojo_tool_test_run "
              "--make-bed --out ../data/cojo_tool_test_run --noweb --no-parents --no-sex")
    os.system("../tools/gcta/gcta64 --bfile ../data/cojo_tool_test_run "
              "--cojo-file ../data/cojo_tool_test_run.ma --cojo-joint --out ../data/test_out")

    file_gcta_out = open("../data/test_out.jma.cojo")
    file_gcta_out.readline()
    snp1 = file_gcta_out.readline()
    snp2 = file_gcta_out.readline()
    joint_beta_tool = [float(snp1.split('\t')[10]), float(snp2.split('\t')[10])]
    joint_se_tool = [float(snp1.split('\t')[11]), float(snp2.split('\t')[11])]
    logging.info("\nJoint test results (tool): "
                 "\n\tjoint_beta_tool={joint_beta_tool} "
                 "\n\tjoint_se_tool={joint_se_tool}\n".format(joint_beta_tool=joint_beta_tool,
                                                              joint_se_tool=joint_se_tool))
    """


if __name__ == "__main__":
    main()