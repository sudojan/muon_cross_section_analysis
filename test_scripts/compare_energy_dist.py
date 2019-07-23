
import os
import gzip
import matplotlib.pyplot as plt
import numpy as np

def main():
    files = ['build/1904_mc_muon_energies.txt.gz',
            'build/1904_mc_muon_energies_L1.txt.gz',
            'build/1904_mc_muon_energies_l2.txt.gz',
            'build/1904_mc_muon_energies_l3.txt.gz']

    min_energy = 1e30
    max_energy = -1
    data_list = []
    for idx in range(len(files)):
        with gzip.open(files[idx], 'r') as file:
            data_list.append(np.genfromtxt(file))
        min_energy =  min(min(data_list[idx]), min_energy)
        max_energy =  max(max(data_list[idx]), max_energy)

    min_energy = min_energy - 1e-3
    max_energy = max_energy + 1e-3

    nbins = 30
    bin_edges = np.logspace(np.log10(min_energy), np.log10(max_energy), nbins+1)
    hist_data = np.empty((len(files), nbins))
    num_events = []
    for idx in range(len(files)):
        hist_data[idx] = np.histogram(data_list[idx], bins=bin_edges)[0]
        num_events.append(len(data_list[idx]))

    labels = ['L0', 'L1', 'L2', 'L3']
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for idx in range(len(files)):
        ax.plot(bin_edges,
            np.r_[hist_data[idx][0], hist_data[idx]],
            drawstyle='steps',
            label='{}: {}'.format(labels[idx], num_events[idx]),)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Energy / GeV')
    ax.legend()
    # fig.savefig(output)
    plt.show()

if __name__ == '__main__':
    main()