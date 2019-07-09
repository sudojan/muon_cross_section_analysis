
import json
import numpy as np
import os


def create_settings_dict():
    set_dict = {}

    # create build directory in current path
    current_path = os.getcwd()
    # or use src path
    # current_path = os.path.dirname(os.path.abspath(__file__))
    build_folder = os.path.join(current_path, 'build')
    set_dict['build_path'] = build_folder
    if not os.path.isdir(set_dict['build_path']):
        os.mkdir(set_dict['build_path'])

    set_dict['path_to_inerpolation_tables'] = '~/.local/share/PROPOSAL/tables'

    # set muon energies for dEdx fit
    set_dict['dedx_n_muons'] = 100
    set_dict['dedx_muon_energy_min'] = 1e4 # MeV
    set_dict['dedx_muon_energy_max'] = 1e11 # MeV
    tmp_arr = np.logspace(np.log10(set_dict['dedx_muon_energy_min']),
                                   np.log10(set_dict['dedx_muon_energy_max']),
                                   set_dict['dedx_n_muons'])
    set_dict['dedx_energies'] = tmp_arr.tolist()
    set_dict['dedx_data_filename_all'] = os.path.join(build_folder, 'data_dedx_all_cross_sections.txt')
    set_dict['dedx_data_filename_sum'] = os.path.join(build_folder, 'data_dedx_sum_cross_sections.txt')

    # step2
    set_dict['dedx_data_fitparams'] = os.path.join(build_folder, 'data_dedx_fitparams.txt')
    set_dict['dedx_plot_test_fitter'] = os.path.join(build_folder, 'plot_dedx_test_fitter.png')
    set_dict['dedx_plot'] = os.path.join(build_folder, 'plot_dedx.png')

    # step3
    set_dict['prop_oversampling'] = 1000
    set_dict['prop_n_muon_energy_bins'] = 30
    set_dict['prop_muon_energy_min'] = 1e4 # MeV
    set_dict['prop_muon_energy_max'] = 1e11 # MeV
    tmp_arr = np.logspace(np.log10(set_dict['prop_muon_energy_min']),
                          np.log10(set_dict['prop_muon_energy_max']),
                          2 * set_dict['prop_n_muon_energy_bins'] + 1)
    set_dict['prop_energy_bin_edges'] = tmp_arr[::2].tolist()
    set_dict['prop_energy_bin_mids'] = tmp_arr[1::2].tolist()
    set_dict['prop_data_ranges'] = os.path.join(build_folder, 'data_prop_ranges.txt')

    # step 4
    set_dict['prop_plot_ranges'] = os.path.join(build_folder, 'plot_range_distribution.png')
    set_dict['prop_n_range_bins'] = 31

    with open(os.path.join(set_dict["build_path"], "settings.json"), "w") as file:
        json.dump(set_dict, fp=file, indent=2, separators=(",", ":"))


if __name__ == "__main__":
    np.random.seed(123)
    create_settings_dict()
