import os

from src.stattests_paper import *

# change to absolute paths
PLINK_PATH = "../tools/plink-1.07-x86_64/plink"
GCTA_PATH = "../tools/gcta/gcta64"
REAL_DATA_PATH = "../../hacking_cojo_april/data"


def generate_ped(genotypes, phenotypes, ea, ra):
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
            letter1 = ra[0] + ra[0]
        elif genotypes[i][0] == 1:
            letter1 = ea[0] + ra[0]
        else:
            letter1 = ea[0] + ea[0]

        if genotypes[i][1] == 0:
            letter2 = ra[1] + ra[1]
        elif genotypes[i][1] == 1:
            letter2 = ea[1] + ra[1]
        else:
            letter2 = ea[1] + ea[1]

        ped_data_dict["SNP1_1"].append(letter1[0])
        ped_data_dict["SNP1_2"].append(letter1[1])
        ped_data_dict["SNP2_1"].append(letter2[0])
        ped_data_dict["SNP2_2"].append(letter2[1])

    ped_data = pd.DataFrame.from_dict(ped_data_dict)
    ped_data = ped_data[['FID', 'IID', 'Phenotype', 'SNP1_1', 'SNP1_2', 'SNP2_1', 'SNP2_2']]

    return ped_data


def main():

    # creating data for run

    gwas_height_data = pd.read_csv("../../hacking_cojo_april/data/gwas_height_with_clickhouse_r.csv")

    for locus in gwas_height_data.locus.unique()[:1]:
        locus_data = gwas_height_data[gwas_height_data.locus == locus]

        dict_of_r = locus_data[["ld", "SomaLogic_old", "UKBioBank", "metabolomics", "olink"]].to_dict('series')

        for group_of_gwases, r in dict_of_r.items():
            r = r[0]

            print("creating bfile")
            population_size = (locus_data.iloc[0]['n'] + locus_data.iloc[1]['n']) // 2

            freq_a1 = locus_data.iloc[0]["freq_ea"]
            freq_b1 = locus_data.iloc[1]["freq_ea"]

            d = r * np.sqrt(freq_a1 * (1 - freq_a1) * freq_b1 * (1 - freq_b1))

            beta_a = locus_data.iloc[0]["beta_gwas"]
            beta_b = locus_data.iloc[1]["beta_gwas"]

            haplotypes_prob = get_haplotypes_probabilities(d, freq_a1, freq_b1)
            haplotypes = get_haplotypes(haplotypes_prob, population_size)
            genotypes = get_genotypes(haplotypes, population_size)
            phenotypes = get_phenotypes(genotypes, beta_a, beta_b, population_size)

            ped_data = generate_ped(genotypes, phenotypes, locus_data.ea, locus_data.ra)
            ped_data.to_csv(f"{REAL_DATA_PATH}/cojo_tool_{locus}_run.ped", sep='\t', header=False, index=False)

            os.system(f'{PLINK_PATH} '
                      f'--file {REAL_DATA_PATH}/cojo_tool_{locus}_run '
                      f'--make-bed '
                      f'--out {REAL_DATA_PATH}/cojo_tool_{locus}_run '
                      f'--noweb '
                      f'--no-parents '
                      f'--no-sex > /dev/null')

            print("creating map file")
            file_map = open(f"{REAL_DATA_PATH}/cojo_tool_{locus}_run.map", 'w')
            file_map.write(str(locus_data.iloc[0]["chr"]) + " " +
                           str(locus_data.iloc[0]["rsid"] + " 0 " +
                           str(locus_data.iloc[0]["bp"]) + '\n'))
            file_map.write(str(locus_data.iloc[1]["chr"]) + " " +
                           str(locus_data.iloc[1]["rsid"] + " 0 " +
                               str(locus_data.iloc[1]["bp"])))
            file_map.close()

            print("creating ma file")
            ma_locus_data = locus_data[["rsid", "ea", "ra", "freq_ea", "beta_gwas", "se_gwas", "p_gwas", "n"]]
            ma_locus_data.to_csv(f"{REAL_DATA_PATH}/cojo_tool_{locus}_run.ma", sep='\t', header=True, index=False)

            print("creating snplist")
            file_snplist = open(f"{REAL_DATA_PATH}/{locus}.snplist", 'w')
            file_snplist.write(locus_data.iloc[1]['rsid'])
            file_snplist.close()

            print("running conditional")
            os.system(f'{GCTA_PATH} --bfile {REAL_DATA_PATH}/cojo_tool_{locus}_run '
                      f'--cojo-file {REAL_DATA_PATH}/cojo_tool_{locus}_run.ma '
                      f'--cojo-cond {REAL_DATA_PATH}/{locus}.snplist '
                      f'--out {REAL_DATA_PATH}/{locus}_{group_of_gwases}_out > /dev/null')

    return 0


if __name__ == "__main__":
    main()
