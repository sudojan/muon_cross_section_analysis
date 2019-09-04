
# import matplotlib as mpl
# mpl.use('Agg')
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import argparse
import os
import json

def plot_dNdx(settings_dict, brems_multiplier, style='buildup'):
    bin_mids = np.array(settings_dict["energy_loss_bin_mids"])
    bin_edges = np.array(settings_dict["energy_loss_bin_edges"])

    sec_bins = np.genfromtxt(settings_dict["step01_file_{}_data".format(style)].format(brems_multiplier))
    sec_errs = np.genfromtxt(settings_dict["step01_file_{}_err_data".format(style)].format(brems_multiplier))
    sum_bins = np.sum(sec_bins[:-1], axis=0)
    max_bin_height = max(sum_bins)
    min_bin_height = 1. / settings_dict["n_muons"] * \
                    (settings_dict["propagation_length_min"] / settings_dict["propagation_length_max"])

    xerr_min = bin_mids - bin_edges[:-1]
    xerr_max = bin_edges[1:] - bin_mids

    fig = plt.figure(figsize=(8,5))
    # gs = gridspec.GridSpec(1, 5)
    # ax = fig.add_subplot(gs[:-1])
    ax = fig.add_subplot(111)

    for idx in range(len(sec_bins)-1):
        if np.count_nonzero(sec_bins[idx]) == 0:
            continue
        ax.plot(bin_edges,
                np.r_[sec_bins[idx,0], sec_bins[idx]],
                drawstyle='steps',
                label=settings_dict["secondary_types"][idx],
                )
    ax.errorbar(bin_mids,
                sec_bins[-1],
                yerr=sec_errs[-1],
                xerr=(xerr_min, xerr_max),
                linestyle='',
                label="Binned Track",
                )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim([0.5*min_bin_height, max_bin_height*2])
    ax.set_xlabel(r'secondary energy / MeV')
    ax.set_ylabel(r'd$N$ / 100 m / muon')
    ax.legend()#bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    ax.grid()
    fig.savefig(settings_dict["step01_file_{}_plots".format(style)].format(brems_multiplier))
    plt.close()

def loop_brems_multiplier_plot(settings_dict, style='buildup'):
    if not os.path.isdir(settings_dict["step01_path_{}_plots".format(style)]):
        os.mkdir(settings_dict["step01_path_{}_plots".format(style)])

    for brems_multiplier in settings_dict["brems_multiplier_{}_arr".format(style)]:
        plot_dNdx(settings_dict, brems_multiplier, style)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file', type=str,
                        dest='settings_file', default="build/settings.json",
                        help='json file containing the settings')
    args = parser.parse_args()

    with open(args.settings_file) as file:
        settings_dict = json.load(file)

        loop_brems_multiplier_plot(settings_dict, "buildup")
        # loop_brems_multiplier_plot(settings_dict, "testing")

        # for testing
        # multiplier = 0.6
        # plot_dNdx(settings_dict, brems_multiplier=multiplier)
