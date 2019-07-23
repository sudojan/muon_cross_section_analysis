import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from argparse import ArgumentParser

def average_range_calc(energies, a, b):
    return np.log(1 + b*energies/a) / b

def plot_range_distribution(energies_prop_bin_edges,
                            energies_prop_bin_mids,
                            range_bin_edges,
                            range_bin_mids,
                            fit_params,
                            ranges,
                            output_file):

    fig = plt.figure()
    gs = gridspec.GridSpec(3, 1)
    ax1 = fig.add_subplot(gs[:-1])
    ax2 = fig.add_subplot(gs[-1], sharex=ax1)
    ax1.set_title('statistic per energy bin: {}'.format(len(ranges[0])))

    ranges_hist2d = np.empty((len(energies_prop_bin_mids), len(range_bin_mids)))
    for idx in range(len(energies_prop_bin_mids)):
        ranges_hist2d[idx], _ = np.histogram(ranges[idx], bins=range_bin_edges)

    Xe, Ye = np.meshgrid(energies_prop_bin_edges, range_bin_mids)
    im = ax1.pcolormesh(Xe, Ye, np.atleast_2d(ranges_hist2d.T), norm=LogNorm())

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
    ax2.set_xlabel('Muon Energy / MeV')
    ax2.set_xscale('log')
    ax2.legend()
    plt.subplots_adjust(hspace=.0)
    plt.setp(ax1.get_xticklabels(), visible=False)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # fig.tight_layout()
    fig.savefig(output_file, bbox_inches='tight', pad_inches=0.02, dpi=300)


def main():
    parser = ArgumentParser()
    parser.add_argument('-f','--file',
                        type=str,
                        dest='settings_file',
                        default="build/new_settings.json",
                        help='json file containing the settings')
    args = parser.parse_args()

    with open(args.settings_file) as file:
        settings_dict = json.load(file)

    fit_params = np.genfromtxt(settings_dict['dedx_data_fitparams'])
    ranges = np.genfromtxt(settings_dict['prop_data_ranges'])

    plot_range_distribution(np.array(settings_dict['prop_energy_bin_edges']),
                            np.array(settings_dict['prop_energy_bin_mids']),
                            settings_dict['range_bin_edges'],
                            settings_dict['range_bin_mids'],
                            fit_params,
                            ranges,
                            settings_dict['prop_plot_ranges'])

if __name__ == '__main__':
    main()
