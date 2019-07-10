import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from argparse import ArgumentParser

def average_range_calc(energies, a, b):
    return np.log(1 + b*energies/a) / b

def plot_range_distribution_comparison(energies_prop_bin_edges,
                                        energies_prop_bin_mids,
                                        range_bin_edges,
                                        range_bin_mids,
                                        fit_params_base,
                                        fit_params_new,
                                        ranges_base,
                                        ranges_new,
                                        output_file):

    fig = plt.figure()
    gs = gridspec.GridSpec(3, 1)
    ax1 = fig.add_subplot(gs[:-1])
    ax2 = fig.add_subplot(gs[-1], sharex=ax1)
    ax1.set_title('statistic per energy bin: {}'.format(len(ranges_base[0])))

    ranges_hist2d_base = np.empty((len(energies_prop_bin_mids), len(range_bin_mids)))
    for idx in range(len(energies_prop_bin_mids)):
        ranges_hist2d_base[idx] = np.histogram(ranges_base[idx], bins=range_bin_edges)[0]

    ranges_hist2d_new = np.empty((len(energies_prop_bin_mids), len(range_bin_mids)))
    for idx in range(len(energies_prop_bin_mids)):
        ranges_hist2d_new[idx] = np.histogram(ranges_new[idx], bins=range_bin_edges)[0]

    tmp_arr = ranges_hist2d_base - ranges_hist2d_new
    ranges_hist2d = np.ma.masked_where(np.abs(tmp_arr) < 0.1, tmp_arr)
    max_val = np.log10(np.max(np.abs(ranges_hist2d)))

    Xe, Ye = np.meshgrid(energies_prop_bin_edges, range_bin_mids)
    im = ax1.pcolormesh(Xe, Ye,
                        np.atleast_2d(ranges_hist2d.T),
                        cmap=plt.get_cmap('coolwarm'),
                        vmin=-max_val, vmax=max_val,
                        )

    dedx_ranges_base = average_range_calc(energies_prop_bin_mids,
                                         fit_params_base[0],
                                         fit_params_base[1])

    dedx_ranges_new = average_range_calc(energies_prop_bin_mids,
                                         fit_params_new[0],
                                         fit_params_new[1])

    average_ranges_base = np.average(ranges_base, axis=1)
    median_ranges_base = np.median(ranges_base, axis=1)
    average_ranges_new = np.average(ranges_new, axis=1)
    median_ranges_new = np.median(ranges_new, axis=1)

    ax1.plot(energies_prop_bin_mids, average_ranges_base, label='average base')
    ax1.plot(energies_prop_bin_mids, average_ranges_new, label='average new')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel('Range / cm')
    ax1.legend()

    ax2.plot(energies_prop_bin_mids, dedx_ranges_new/dedx_ranges_base, label='dedx fit')
    ax2.plot(energies_prop_bin_mids, median_ranges_new/median_ranges_base, label='median')
    ax2.plot(energies_prop_bin_mids, average_ranges_new/average_ranges_base, label='average')
    ax2.set_ylabel('new/base')
    ax2.set_xlabel('Muon Energy / MeV')
    ax2.set_xscale('log')
    ax2.grid()
    ax2.legend()
    plt.subplots_adjust(hspace=.0)
    plt.setp(ax1.get_xticklabels(), visible=False)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.86, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel('sgn(base - new) log10(|base - new|)')

    # fig.tight_layout()
    fig.savefig(output_file)
    plt.show()


def main():

    with open('build/baseline_settings.json') as file:
        settings_base = json.load(file)

    with open('build/new_settings.json') as file:
        settings_new = json.load(file)

    fit_params_base = np.genfromtxt(settings_base['dedx_data_fitparams'])
    ranges_base = np.genfromtxt(settings_base['prop_data_ranges'])

    fit_params_new = np.genfromtxt(settings_new['dedx_data_fitparams'])
    ranges_new = np.genfromtxt(settings_new['prop_data_ranges'])

    plot_range_distribution_comparison(np.array(settings_base['prop_energy_bin_edges']),
                                        np.array(settings_base['prop_energy_bin_mids']),
                                        settings_base['range_bin_edges'],
                                        settings_base['range_bin_mids'],
                                        fit_params_base, fit_params_new,
                                        ranges_base, ranges_new,
                                        'build/plot_range_distribution_compare.png')

if __name__ == '__main__':
    main()
