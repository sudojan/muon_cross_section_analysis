import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec

def create_bin_edges_and_mids(low, high, nbins, logscale):
    if logscale:
        bins_tmp = np.logspace(np.log10(low), np.log10(high), num = 2*nbins + 1)
    else:
        bins_tmp = np.linspace(low, high, num = 2*nbins + 1)
    bin_edges = bins_tmp[::2]
    bin_mids = bins_tmp[1::2]
    return bin_edges, bin_mids

def plot_range_distribution(energies, dedx_ranges, ranges, output_file):
    nrange_bins = 21
    small_val = 0.1
    ebin_edges, ebin_mids = create_bin_edges_and_mids(min(energies), max(energies), len(energies), True)
    rbin_edges, rbin_mids = create_bin_edges_and_mids(np.min(ranges)-small_val,
                                                      np.max(ranges)+small_val,
                                                      nbins=nrange_bins,
                                                      logscale=True)

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1)
    ax1 = fig.add_subplot(gs[:-1])
    ax2 = fig.add_subplot(gs[-1], sharex=ax1)
    ranges_hist2d = np.empty((len(energies), nrange_bins))
    for idx in range(len(energies)):
        ranges_hist2d[idx], _ = np.histogram(ranges[idx], bins=rbin_edges)

    Xe, Ye = np.meshgrid(ebin_edges, rbin_mids)
    im = ax1.pcolormesh(Xe, Ye, np.atleast_2d(ranges_hist2d.T), norm=LogNorm())

    average_ranges = np.average(ranges, axis=1)
    ax1.plot(energies, dedx_ranges, label='fit')
    ax1.plot(energies, average_ranges, label='simulation')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel('Range / cm')
    ax1.legend()

    ax2.plot(energies, dedx_ranges/average_ranges)
    ax2.set_ylabel('dEdx / Simulation')
    ax2.set_xlabel('Energy / MeV')
    ax2.set_xscale('log')
    plt.subplots_adjust(hspace=.0)
    plt.setp(ax1.get_xticklabels(), visible=False)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # fig.tight_layout()
    fig.savefig(output_file)
    plt.show()


def main():
    energies = np.logspace(4, 11, 10)
    fit_params = np.genfromtxt('build/data_dedx_fitparams.txt')
    ranges = np.genfromtxt('build/dedx_data_range.txt')

    def average_range_calc(energies, a, b):
        return np.log(1 + b*energies/a) / b
    dedx_ranges = average_range_calc(energies, fit_params[0], fit_params[1])
    output_file = 'build/plot_range_distribution'
    plot_range_distribution(energies, dedx_ranges, ranges, output_file)

if __name__ == '__main__':
    main()
