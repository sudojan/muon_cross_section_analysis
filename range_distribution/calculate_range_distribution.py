import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from argparse import ArgumentParser

def create_bin_edges_and_mids(low, high, nbins, logscale):
    if logscale:
        bins_tmp = np.logspace(np.log10(low), np.log10(high), num = 2*nbins + 1)
    else:
        bins_tmp = np.linspace(low, high, num = 2*nbins + 1)
    bin_edges = bins_tmp[::2]
    bin_mids = bins_tmp[1::2]
    return bin_edges, bin_mids

def plot_range_distribution(energies_prop_bin_edges,
                            energies_prop_bin_mids,
                            nrange_bins,
                            fit_params,
                            ranges,
                            output_file):

    # small_val = 0.1
    # min_range = np.min(ranges)-small_val
    # max_range = np.max(ranges)-small_val
    min_range = 1e3
    max_range = 1e7
    rbin_edges, rbin_mids = create_bin_edges_and_mids(min_range,
                                                      max_range,
                                                      nbins=nrange_bins,
                                                      logscale=True)

    fig = plt.figure()
    gs = gridspec.GridSpec(3, 1)
    ax1 = fig.add_subplot(gs[:-1])
    ax2 = fig.add_subplot(gs[-1], sharex=ax1)
    ranges_hist2d = np.empty((len(energies_prop_bin_mids), nrange_bins))
    for idx in range(len(energies_prop_bin_mids)):
        ranges_hist2d[idx], _ = np.histogram(ranges[idx], bins=rbin_edges)

    Xe, Ye = np.meshgrid(energies_prop_bin_edges, rbin_mids)
    im = ax1.pcolormesh(Xe, Ye, np.atleast_2d(ranges_hist2d.T), norm=LogNorm())

    def average_range_calc(energies, a, b):
        return np.log(1 + b*energies/a) / b
    dedx_ranges = average_range_calc(energies_prop_bin_mids,
                                     fit_params[0],
                                     fit_params[1])

    average_ranges = np.average(ranges, axis=1)
    median_ranges = np.median(ranges, axis=1)
    ax1.plot(energies_prop_bin_mids, dedx_ranges, label='dEdx Fit')
    ax1.plot(energies_prop_bin_mids, average_ranges, label='average Simulation')
    ax1.plot(energies_prop_bin_mids, median_ranges, label='median Simulation')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel('Range / cm')
    ax1.legend()

    ax2.plot(energies_prop_bin_mids, dedx_ranges/average_ranges, label='fit/average')
    ax2.plot(energies_prop_bin_mids, median_ranges/average_ranges, label='median/average')
    ax2.set_ylabel('ratio')
    ax2.set_xlabel('Energy / MeV')
    ax2.set_xscale('log')
    ax2.legend()
    plt.subplots_adjust(hspace=.0)
    plt.setp(ax1.get_xticklabels(), visible=False)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # fig.tight_layout()
    fig.savefig(output_file)
    plt.show()


def main():
    parser = ArgumentParser()
    parser.add_argument('-f','--file',
                        type=str,
                        dest='settings_file',
                        default="build/settings.json",
                        help='json file containing the settings')
    args = parser.parse_args()

    with open(args.settings_file) as file:
        settings_dict = json.load(file)

    fit_params = np.genfromtxt(settings_dict['dedx_data_fitparams'])
    ranges = np.genfromtxt(settings_dict['prop_data_ranges'])

    plot_range_distribution(np.array(settings_dict['prop_energy_bin_edges']),
                            np.array(settings_dict['prop_energy_bin_mids']),
                            settings_dict['prop_n_range_bins'],
                            fit_params,
                            ranges,
                            settings_dict['prop_plot_ranges'])

if __name__ == '__main__':
    main()
