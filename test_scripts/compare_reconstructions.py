
import matplotlib.pyplot as plt
from matplotlib import lines
import numpy as np

def mask_data_arr(arr_list):
    # mask every zero in data
    mask_arr = np.ones(len(arr_list[0]), dtype=bool)
    # mask_arr = np.full(len(arr_list[0]), True) # since numpy 1.12
    for idx in range(len(arr_list)):
        mask_arr = mask_arr & (arr_list[idx] != 0)

    new_arr = np.empty((len(arr_list), np.count_nonzero(mask_arr)))
    for idx in range(len(arr_list)):
        new_arr[idx] = arr_list[idx][mask_arr]

    return new_arr

def create_pull_arr(arr_list):
    pull_arr = np.empty((len(arr_list)-1, len(arr_list[0])))
    for idx in range(len(arr_list)-1):
        pull_arr[idx] = arr_list[idx+1] / arr_list[0] - 1

    return pull_arr

def create_bin_mid_and_edges(arr_min, arr_max, nbins, dolog=False):
    # set loss bins
    if dolog:
        bin_arr = np.logspace(np.log10(arr_min),
                              np.log10(arr_max),
                              num = 2 * nbins + 1)
    else:
        bin_arr = np.linspace(arr_min, arr_max, 2 * nbins + 1)
    bin_edges = bin_arr[::2]
    bin_mids = bin_arr[1::2]
    return bin_edges, bin_mids

def plot_resolution(arr_list, label_list, output, xlabel=None, xlog=False, ylog=False):

    new_arr = mask_data_arr(arr_list)
    pull_arr = np.abs(create_pull_arr(new_arr))

    nbins = 30
    bins_edges, bin_mids = create_bin_mid_and_edges(np.min(new_arr[0]),
                                                    np.max(new_arr[0]),
                                                    nbins,
                                                    dolog=xlog)

    quantiles = [0.25, 0.5, 0.9]
    bins_indices = np.digitize(new_arr[0], bins_edges) - 1 # -1 because digitize starts at one
    dev_arr = np.zeros((len(pull_arr), len(quantiles), nbins-1))
    for idx in range(nbins - 1):
        # get a mask of all events in this label bin
        mask_events_in_bin = bins_indices == idx
        # print(np.count_nonzero(mask_events_in_bin))
        if np.count_nonzero(mask_events_in_bin) > 10:            
            for jdx in range(len(pull_arr)):
                for kdx in range(len(quantiles)):
                    dev_arr[jdx, kdx, idx] = np.quantile(pull_arr[jdx][mask_events_in_bin], quantiles[kdx])


    the_linestyles = list(lines.lineStyles)
    # the_linestyles = [':', '-', '-.']
    ls_counter = 0
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('statistic: {}, masked: {}'.format(len(pull_arr[0]), len(arr_list[0])-len(new_arr[0])))
    for idx in range(len(pull_arr)):
        tmp=ax.plot(bin_mids,
                    np.r_[dev_arr[idx, 0, 0], dev_arr[idx, 0]],
                    drawstyle='steps-pre',
                    label=label_list[idx],
                    linestyle=the_linestyles[ls_counter])
        ls_counter += 1
        for jdx in range(len(quantiles)-1):
            ax.plot(bin_mids,
                    np.r_[dev_arr[idx, jdx+1,0], dev_arr[idx, jdx+1]],
                    drawstyle='steps-pre',
                    color=tmp[0].get_color(),
                    linestyle=the_linestyles[ls_counter])
            ls_counter += 1
        ls_counter = 0

    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_ylabel('|Reco/Truth - 1|')
    ax.legend()
    fig.savefig(output)
    plt.show()


def plot_pull(arr_list, label_list, output, ylog=False):
    new_arr = mask_data_arr(arr_list)
    pull_arr = create_pull_arr(new_arr)

    # bins = np.linspace(np.min(pull_arr), np.max(pull_arr), 100)
    bins = np.linspace(-5, 5, 100)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('statistic: {}'.format(len(pull_arr[0])))
    for idx in range(len(pull_arr)):
        ax.hist(pull_arr[idx],
                histtype='step',
                # alpha=0.6,
                label=label_list[idx],
                bins=bins,)
    ax.set_xlabel('Reco/Truth - 1')
    if ylog:
        ax.set_yscale('log')
    ax.legend()
    # ax.set_ylim([5e-1, 1e2])
    fig.savefig(output)
    plt.show()
    # plt.close()

def plot_data_distr(arr_list, label_list, output, xlabel=None, xlog=False):
    new_arr = mask_data_arr(arr_list)
    if xlog:
        bins = np.logspace(np.log10(np.min(new_arr)), np.log10(np.max(new_arr)), 20)
    else:
        bins = np.linspace(np.min(new_arr), np.max(new_arr), 20)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('statistic: {}'.format(len(new_arr[0])))
    for idx in range(len(new_arr)):
        ax.hist(new_arr[idx],
                histtype='step',
                label=label_list[idx],
                bins=bins,)
    ax.set_yscale('log')
    if xlog:
        ax.set_xscale('log')
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.legend(loc='lower right')
    fig.savefig(output)
    plt.show()

def plot_correlation(arr_list, label_list, output):
    new_arr = np.log10(mask_data_arr(arr_list))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('statistics: {}'.format(len(new_arr[0])))
    hb = ax.hexbin(new_arr[0], new_arr[1], gridsize=100, bins='log', mincnt=1)
    ax.plot([np.min(new_arr), np.max(new_arr)], [np.min(new_arr), np.max(new_arr)])
    ax.set_xlabel('log10({})'.format(label_list[0]))
    ax.set_ylabel('log10({})'.format(label_list[1]))
    ax.set_xlim(np.min(new_arr[0]), np.max(new_arr[0]))
    ax.set_ylim(np.min(new_arr[0]), np.max(new_arr[0]))
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    cb = fig.colorbar(hb)
    # cb.set_label('log10(N)')
    fig.savefig(output)
    plt.show()

def do_energy_plots(filename):
    with open(filename) as file:
        first_line = file.readline()
        all_labels = first_line[1:].split()

    all_arr = np.genfromtxt(filename)
    # dnn_uncert = all_arr[:, all_labels.index('DeepLearning_PrimaryMuonEnergyEntry_log_uncertainty')]
    # dnn_entry = all_arr[:, all_labels.index('DeepLearning_PrimaryMuonEnergyEntry')]
    pre_mask = np.ones(len(all_arr[:,0]), dtype=bool)
    # pre_mask = (dnn_uncert < 0.2) & (dnn_entry > 3e3) & (dnn_entry < 1e6)
    labels_to_extract = [
        'MMCTrackList_CenterEnergy',
        # 'MostVisibleMuonEnergyEntry',
        'DeepLearning_PrimaryMuonEnergyEntry',
        'SplineMPETruncatedEnergy_SPICEMie_ORIG_Muon',
        'SplineMPEMuEXDifferential',
    ]
    combined_arr = np.array([all_arr[:, all_labels.index(idx)][pre_mask] for idx in labels_to_extract])
    combined_arr[1] = combined_arr[1]
    plot_pull(combined_arr,
              labels_to_extract[1:],
              output='plot_pull_energy.png',
              ylog=True,
              )
    plot_data_distr(combined_arr,
                    labels_to_extract,
                    output='plot_distribution_energy.png',
                    xlabel='True Energy / GeV',
                    xlog=True,
                    )
    plot_resolution(combined_arr,
                    labels_to_extract[1:],
                    output='plot_resolution_energy.png',
                    xlabel='True Energy / GeV',
                    xlog=True,
                    )
    for idx in range(1, len(labels_to_extract)):
        plot_correlation([combined_arr[0],combined_arr[idx]],
                        [labels_to_extract[0],labels_to_extract[idx]],
                        output='plot_correlation_energy_{}.png'.format(labels_to_extract[idx]),)

def do_length_plots(filename):
    with open(filename) as file:
        first_line = file.readline()
        all_labels = first_line[1:].split()

    all_arr = np.genfromtxt(filename)
    mask_hitdetector = all_arr[:, all_labels.index('MostVisibleMuonEnergyEntry')] != 0
    true_entry_labels = [
        'MostVisibleMuonEntryx',
        'MostVisibleMuonEntryy',
        'MostVisibleMuonEntryz',
    ]
    true_entries = np.array([all_arr[:, all_labels.index(idx)] for idx in true_entry_labels])
    true_exit_labels = [
        'MostVisibleMuonExitx',
        'MostVisibleMuonExity',
        'MostVisibleMuonExitz',
    ]
    true_exits = np.array([all_arr[:, all_labels.index(idx)] for idx in true_exit_labels])
    dnn_entry_labels = [
        'DeepLearning_MuonEntryPoint_x',
        'DeepLearning_MuonEntryPoint_y',
        'DeepLearning_MuonEntryPoint_z',
    ]
    dnn_entries = np.array([all_arr[:, all_labels.index(idx)] for idx in dnn_entry_labels])
    dnn_exit_labels = [
        'DeepLearning_MuonExitPoint_x',
        'DeepLearning_MuonExitPoint_y',
        'DeepLearning_MuonExitPoint_z',
    ]
    dnn_exits = np.array([all_arr[:, all_labels.index(idx)] for idx in dnn_exit_labels])
    true_length = all_arr[:, all_labels.index('MostVisibleMuonInDetectorTrackLength')]
    # true_length = np.linalg.norm(true_entries - true_exits, axis=0)
    dnn_length = np.linalg.norm(dnn_entries - dnn_exits, axis=0)
    pre_mask = dnn_length > 200
    combined_arr = np.array([true_length[pre_mask], dnn_length[pre_mask]])
    plot_data_distr(combined_arr,
                    ['True', 'DNN'],
                    output='plot_distribution_length.png',
                    xlabel='True Length / m',
                    )
    plot_pull(combined_arr,
              ['DNN'],
              output='plot_pull_length.png',
              ylog=True)
    plot_resolution(combined_arr,
                    ['DNN'],
                    output='plot_resolution_length.png',
                    xlabel='True Length / m',
                    ylog=True,
                    )
    plot_correlation([combined_arr[0],combined_arr[1]],
                    ['MostVisibleMuonInDetectorTrackLength', 'DeepLearning_MuonEntryPoint - DeepLearning_MuonExitPoint'],
                    output='plot_correlation_length_{}.png'.format('DNN'),)


if __name__ == '__main__':
    # filename = 'new_list.txt'
    filename = 'energy_list.txt'
    do_energy_plots(filename)
    # do_length_plots(filename)

