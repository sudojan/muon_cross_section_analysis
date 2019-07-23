
import os
import gzip
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))

def plot_bin_diff(brems_multiplier, hist_arr, output):
    for idx in range(len(hist_arr[0])):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(brems_multiplier, hist_arr[:,idx])
        ax.set_xlabel('bremsstrahlung multiplier')
        ax.set_ylabel('bin height')
        fig.savefig(output.format(idx))
        ax.cla()
        plt.close()


def main():
    # parser = ArgumentParser()
    # parser.add_argument(
    #     '-d', '--dataset',
    #     type=int,
    #     dest='dataset',
    #     default=1904,
    #     help='name of dataset')
    # args = parser.parse_args()

    build_path = os.path.join(SCRIPT_FOLDER, 'build')
    if not os.path.isdir(build_path):
        os.makedirs(build_path)

    brems_multiplier = [0.9, 1.0, 1.1]
    datasets = [1902, 1904, 1906]
    nbins = 20
    hist_arr = np.empty((3, 20))
    for idx in range(len(brems_multiplier)):
        loss_hist_file = os.path.join(build_path, '{}_loss_hist.txt'.format(datasets[idx]))
        hist_arr[idx] = np.genfromtxt(loss_hist_file)

    bin_diff_file = os.path.join(build_path, 'plot_bin_diff_{}.png')
    plot_bin_diff(brems_multiplier, hist_arr, bin_diff_file)

    

if __name__ == '__main__':
    main()
