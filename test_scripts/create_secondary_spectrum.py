
import os
import gzip
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))

def get_secondary_hist(file_energy, file_loss, loss_bin_edges):
    if file_energy.endswith(".gz"):
        with gzip.open(file_energy, 'r') as file:
            energy_arr = np.genfromtxt(file)
    else:
        with open(file_energy) as file:
            energy_arr = np.genfromtxt(file)

    if file_loss.endswith(".gz"):
        with gzip.open(file_loss, 'r') as file:
            loss_arr = np.genfromtxt(file)
    else:
        with open(file_loss) as file:
            loss_arr = np.genfromtxt(file)

    tmp_energy = energy_arr[0]
    tmp_length = energy_arr[3]
    print(len(tmp_energy))

    pre_mask = np.ones(len(tmp_energy), dtype=bool)
    pre_mask = pre_mask & (tmp_energy < 3e4) & (tmp_energy > 1e4) & (tmp_length > 100)
    energies = tmp_energy[pre_mask]
    lengths = tmp_length[pre_mask]
    loss_arr = loss_arr[pre_mask]
    print(len(energies))

    max_loss = np.max(loss_arr)
    if max_loss > loss_bin_edges[-1]:
        print('max', max_loss)

    secondary_heights = np.zeros(len(loss_bin_edges)-1)
    nevents = 0

    for idx in tqdm(range(len(energies))):
        if np.count_nonzero(loss_arr[idx]) == 0:
            continue
        if np.sum(loss_arr[idx]) < 1:
            # print('here')
            continue
        nevents += 1
        secondary_bins_tmp = np.histogram(loss_arr[idx],
                                        bins=loss_bin_edges,
                                        density=False)[0]

        # norm to 100 m propagated distance
        secondary_heights += secondary_bins_tmp * 100 / lengths[idx]
    print(nevents)

    secondary_heights = secondary_heights / nevents
    return secondary_heights

def plot_secondary_hist(loss_bins, secondary_heights, output):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(loss_bins, np.r_[secondary_heights[0], secondary_heights], drawstyle='steps-pre',)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('loss energy / GeV')
    fig.savefig(output)
    plt.show()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        type=int,
        dest='dataset',
        default=1904,
        help='name of dataset')
    args = parser.parse_args()

    build_path = os.path.join(SCRIPT_FOLDER, 'build')
    if not os.path.isdir(build_path):
        os.makedirs(build_path)

    energy_len_file = os.path.join(build_path, '{}_energies_lens.txt.gz'.format(args.dataset))
    loss_file = os.path.join(build_path, '{}_loss.txt.gz'.format(args.dataset))
    loss_hist_file = os.path.join(build_path, '{}_loss_hist.txt'.format(args.dataset))
    plot_hist_file = os.path.join(build_path, '{}_loss_hist_plot.png'.format(args.dataset))

    nbins = 20
    loss_bin_edges = np.logspace(1, 5.1, nbins+1)

    if not os.path.isfile(loss_hist_file):
        secondary_heights = get_secondary_hist(energy_len_file, loss_file, loss_bin_edges)
        np.savetxt(loss_hist_file, secondary_heights)
    else:
        secondary_heights = np.genfromtxt(loss_hist_file)

    plot_secondary_hist(loss_bin_edges, secondary_heights, plot_hist_file)

if __name__ == '__main__':
    main()
