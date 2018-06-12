import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns

sns.set(style="white", color_codes=True)


def plot_simulated_data_with_regressions(simulated_data):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.gca(projection="3d")

    ax.plot(simulated_data.snp_a_gen,
            simulated_data.snp_b_gen,
            simulated_data.phenotype, 'co', zorder=0)

    ax.plot(simulated_data.snp_a_gen,
            simulated_data.phenotype,
            'ko', zdir='y', alpha=0.25, zs=3.0, mec=None, zorder=0)

    ax.plot(simulated_data.snp_b_gen,
            simulated_data.phenotype,
            'ko', zdir='x', alpha=0.25, zs=3.06, mec=None, zorder=0)

    ax.plot(simulated_data.snp_a_gen,
            simulated_data.snp_b_gen,
            'ko', zdir='z', alpha=0.25, zs=-5.1, mec=None, zorder=0)

    # adding plane
    model = smf.ols('phenotype ~ snp_a_gen + snp_b_gen', data=simulated_data).fit()
    # print(model.summary())
    xx, yy = np.meshgrid(np.linspace(0, 3.0, 20), np.linspace(0, 3.0, 20))
    zz = model.params[1] * xx + model.params[2] * yy + model.params[0]
    plane = ax.plot_surface(xx, yy, zz, color='blue', alpha=0.5, cmap=cm.coolwarm, zorder=1)

    fit_a = np.polyfit(simulated_data.snp_a_gen,
                       simulated_data.phenotype,
                       deg=1)
    ax.plot(np.array(simulated_data.snp_a_gen),
            fit_a[0] * np.array(simulated_data.snp_a_gen) + fit_a[1],
            color='k', zdir='y', zs=3.0, zorder=1)

    fit_b = np.polyfit(simulated_data.snp_b_gen,
                       simulated_data.phenotype,
                       deg=1)
    ax.plot(np.array(simulated_data.snp_b_gen),
            fit_b[0] * np.array(simulated_data.snp_b_gen) + fit_b[1],
            color='k', zdir='x', zs=3.06, zorder=1)

    for gen_a in range(3):
        for gen_b in range(3):
            """
            print("gen_a =", gen_a,
                  ", gen_b =", gen_b,
                  ", mean =", simulated_data[(simulated_data.snp_a_gen == gen_a) &
                                             (simulated_data.snp_b_gen == gen_b)].phenotype.mean())
            """
            ax.scatter(xs=[gen_a],
                       ys=[gen_b],
                       zs=[simulated_data[(simulated_data.snp_a_gen == gen_a) &
                                          (simulated_data.snp_b_gen == gen_b)].phenotype.mean()],
                       color='r',
                       zorder=10)

    # print(simulated_data[(simulated_data.snp_a_gen == 2) &
    #                      (simulated_data.snp_b_gen == 2)].phenotype.mean())

    for gen in range(3):
        ax.plot([gen],
                [simulated_data[simulated_data.snp_a_gen == gen].phenotype.mean()],
                color='k',
                zdir='y',
                zs=3.0,
                zorder=10)

        ax.plot([gen],
                [simulated_data[simulated_data.snp_b_gen == gen].phenotype.mean()],
                color='k',
                zdir='x',
                zs=3.0,
                zorder=10)

    ax.set_xlabel("A SNP Genotypes")
    ax.set_xlim(3.0, 0.0)

    ax.set_ylabel('B SNP Genotypes')
    ax.set_ylim(0.0, 3.0)

    ax.set_zlabel("Phenotype")
    ax.set_zlim(-5.0, 5.0)

    ax.set_yticks(range(3))
    ax.set_xticks(range(3))

    fig.colorbar(plane, shrink=0.5, aspect=20, ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    fig.savefig("../out/multivariate_regression_on_simulated_data.pdf", dpi=300)
    fig.savefig("../out/multivariate_regression_on_simulated_data.png", dpi=300)

    plt.show()
    plt.close(fig)


def plot_joint_z1_z2(joint_z1_z2):

    g = sns.jointplot("z1", "z2", data=joint_z1_z2, kind="kde", space=0)
    # g.savefig("../out/joint_dist_z1_z2_1000_iter.png", dpi=300)
    plt.show()
