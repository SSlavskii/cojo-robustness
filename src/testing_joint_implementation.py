import os
from src.stattests_paper import *


LOG_FILE = "../logs/cojo_simulations.log"
logging.basicConfig(format=u'%(filename)s[LINE:%(lineno)d]# %(levelname)-8s %(asctime)s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%H:%M:%S',
                    filename=LOG_FILE)

NUMBER_OF_SNPS = 2
POPULATION_SIZE = 10000  # 100000
REF_POPULATION_SIZE = 10000  # 100000

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

BETA_A = 0.4
BETA_B = -0.2

P_VALUE_THRESHOLD = 0.1


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

    data = {"beta1_single": [], "se1_single": [], "beta2_single": [], "se2_single": [],
            "beta1_multiple": [], "se1_multiple": [], "beta2_multiple": [], "se2_multiple": [],
            "beta1_tool": [], "se1_tool": [], "beta2_tool": [], "se2_tool": [],
            "beta1_sim": [], "se1_sim": [], "beta2_sim": [], "se2_sim": []}

    for i in range(100):

        haplotypes_prob = get_haplotypes_probabilities(D, FREQ_A1, FREQ_B1)
        haplotypes = get_haplotypes(haplotypes_prob, POPULATION_SIZE)
        genotypes = get_genotypes(haplotypes, POPULATION_SIZE)
        genotypes_std = standardise_genotypes(genotypes, FREQ_A1, FREQ_B1)
        phenotypes = get_phenotypes(genotypes, BETA_A, BETA_B, POPULATION_SIZE)

        simulated_data = pd.DataFrame({"phenotype": phenotypes,
                                       "snp_a_gen": genotypes_std[:, 0],
                                       "snp_b_gen": genotypes_std[:, 1]})

        model = smf.ols('phenotype ~ snp_a_gen + snp_b_gen', data=simulated_data).fit()

        gwas = get_gwas(simulated_data,
                        freq_a1=1 - sum(genotypes[:, 0]) / (2 * POPULATION_SIZE),
                        freq_b1=1 - sum(genotypes[:, 1]) / (2 * POPULATION_SIZE))

        logging.info("\nSimulated summary statistics: \n\t {gwas}\n".format(gwas=gwas))

        gwas['A1'] = ['A', 'A']
        gwas['A2'] = ['T', 'T']
        gwas['N'] = [POPULATION_SIZE, POPULATION_SIZE]

        ma_data = gwas[["snp_num", "A1", "A2", "freq1", "beta", "se", "p", "N"]]
        ma_data.to_csv("../data/cojo_tool_test_run.ma", sep='\t', header=True, index=False)

        ped_data = generate_ped(genotypes, phenotypes)
        ped_data.to_csv("../data/cojo_tool_test_run.ped", sep='\t', header=False, index=False)

        os.system("../tools/plink-1.07-x86_64/plink --file ../data/cojo_tool_test_run "
                  "--make-bed --out ../data/cojo_tool_test_run --noweb --no-parents --no-sex > /dev/null")
        os.system("../tools/gcta/gcta64 --bfile ../data/cojo_tool_test_run "
                  "--cojo-file ../data/cojo_tool_test_run.ma --cojo-joint --out ../data/test_out > /dev/null")

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

        ref_r = np.corrcoef(genotypes[:, 0], genotypes[:, 1])
        # print(ref_r[0, 1])

        joint_beta, joint_se, joint_p = joint_test(gwas=gwas,
                                                   population_size=POPULATION_SIZE,
                                                   ref_population_size=REF_POPULATION_SIZE,
                                                   ref_r=ref_r[0, 1])

        joint_beta = list(joint_beta.flat)
        logging.info("\nJoint test results:"
                     "\n\tjoint_beta={joint_beta}"
                     "\n\tjoint_se={joint_se}"
                     "\n\tjoint_p={joint_p}\n".format(joint_beta=joint_beta,
                                                      joint_se=joint_se,
                                                      joint_p=joint_p))

        data["beta1_single"].append(gwas.iloc[0]["beta"])
        data["beta2_single"].append(gwas.iloc[1]["beta"])
        data["se1_single"].append(gwas.iloc[0]["se"])
        data["se2_single"].append(gwas.iloc[1]["se"])

        data["beta1_multiple"].append(model.params.snp_a_gen)
        data["beta2_multiple"].append(model.params.snp_b_gen)
        data["se1_multiple"].append(model.bse.snp_a_gen)
        data["se2_multiple"].append(model.bse.snp_b_gen)

        data["beta1_tool"].append(joint_beta_tool[0])
        data["beta2_tool"].append(joint_beta_tool[1])
        data["se1_tool"].append(joint_se_tool[0])
        data["se2_tool"].append(joint_se_tool[1])

        data["beta1_sim"].append(joint_beta[0])
        data["beta2_sim"].append(joint_beta[1])
        data["se1_sim"].append(joint_se[0])
        data["se2_sim"].append(joint_se[0])

    data = pd.DataFrame.from_dict(data)
    data.to_csv("../data/testing_joint_implementation.tsv", sep='\t', header=True, index=False)


if __name__ == "__main__":
    main()
