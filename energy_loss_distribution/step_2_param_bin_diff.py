
# import matplotlib as mpl
# mpl.use('Agg')
# from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import argparse
import os
import json


def all_bins_errs(settings_dict):
    sec_bins = np.empty((settings_dict["n_buildup_multiplier"],
                        settings_dict["n_energy_loss_bins"]))
    sec_errs = np.empty((settings_dict["n_buildup_multiplier"],
                        settings_dict["n_energy_loss_bins"]))
    for idx, brems_multiplier in enumerate(settings_dict["brems_multiplier_buildup_arr"]):
        tmp = np.genfromtxt(settings_dict["step01_file_buildup_data"].format(brems_multiplier))
        sec_bins[idx] = tmp[-1]
        tmp = np.genfromtxt(settings_dict["step01_file_buildup_err_data"].format(brems_multiplier))
        sec_errs[idx] = tmp[-1]
    return sec_bins, sec_errs

def param_bin_diff(settings_dict):
    bins_all, _ = all_bins_errs(settings_dict)

    param_arr = np.empty((settings_dict["n_energy_loss_bins"], 2))
    for idx in range(settings_dict["n_energy_loss_bins"]):
        params, covariance_matrix = np.polyfit(settings_dict["brems_multiplier_buildup_arr"],
                                               bins_all[:,idx],#/bins_all[5,idx],
                                               deg=1,
                                               cov=True)
        param_arr[idx] = params

    np.savetxt(settings_dict["step02_file_param_bin_diff"], param_arr)

def plot_param_bin_diff(settings_dict, bin_num):
    if not os.path.isfile(settings_dict["step02_file_param_bin_diff"]):
        param_bin_diff(settings_dict)

    bins_all, errs_all = all_bins_errs(settings_dict)
    param_arr = np.genfromtxt(settings_dict["step02_file_param_bin_diff"])

    x_plot = np.linspace(min(settings_dict["brems_multiplier_buildup_arr"]),
                         max(settings_dict["brems_multiplier_buildup_arr"]),
                         100)

    bin_edges = np.array(settings_dict["energy_loss_bin_edges"])

    sformat = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    fformat = mticker.FuncFormatter(lambda x,pos : "${}$".format(sformat._formatSciNotation('%1.2e' % x)))


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(settings_dict["brems_multiplier_buildup_arr"],
                bins_all[:,bin_num],
                yerr=errs_all[:,bin_num],
                xerr=None,
                linestyle='',
                marker='o',
                markersize=2,
                markerfacecolor='k',
                markeredgecolor='k',
                label="Data",
                )

    ax.plot(x_plot, param_arr[bin_num, 0] * x_plot + param_arr[bin_num, 1], label='Lin. Reg.')
    ax.set_xlabel(r'Bremsstrahlung Multiplier')
    ax.set_ylabel(r'd$N$ / 100 m / muon')
    ax.set_title('Energy bin {} - {} MeV'.format(fformat(bin_edges[bin_num]), fformat(bin_edges[bin_num+1])))
    ax.legend(loc="best")
    fig.tight_layout(pad=0, h_pad=1.02, w_pad=1.02)
    fig.savefig(settings_dict["step02_file_plots"].format(bin_num))

def plot_for_each_bin(settings_dict):
    if not os.path.isdir(settings_dict["step02_path_plots"]):
        os.mkdir(settings_dict["step02_path_plots"])

    for idx in range(settings_dict["n_energy_loss_bins"]):
        plot_param_bin_diff(settings_dict, idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file', type=str,
                        dest='settings_file', default="build/settings.json",
                        help='json file containing the settings')
    args = parser.parse_args()
    np.random.seed(123)

    with open(args.settings_file) as file:
        settings_dict = json.load(file)

        # init file names and directories
        if not os.path.isdir(settings_dict["step02_path"]):
            os.mkdir(settings_dict["step02_path"])

        # param_bin_diff(settings_dict)
        plot_for_each_bin(settings_dict)


