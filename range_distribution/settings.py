
import json
import numpy as np
import os
from argparse import ArgumentParser


def create_settings_dict(cross_section_type):
    set_dict = {}

    # create build directory in current path
    current_path = os.getcwd()
    # or use src path
    # current_path = os.path.dirname(os.path.abspath(__file__))
    set_dict['build_path'] = os.path.join(current_path, 'build')
    if not os.path.isdir(set_dict['build_path']):
        os.mkdir(set_dict['build_path'])

    set_dict['path_to_inerpolation_tables'] = '~/.local/share/PROPOSAL/tables'
    if cross_section_type == 'baseline':
        set_dict['brems_param_name'] = 'BremsKelnerKokoulinPetrukhin'
        set_dict['epair_param_name'] = 'EpairKelnerKokoulinPetrukhin'
        set_dict['photo_param_name'] = 'PhotoAbramowiczLevinLevyMaor97'
    elif cross_section_type == 'high':
        set_dict['brems_param_name'] = 'BremsSandrockSoedingreksoRhode'
        set_dict['epair_param_name'] = 'EpairKelnerKokoulinPetrukhin'
        set_dict['photo_param_name'] = 'PhotoBezrukovBugaev'
    elif cross_section_type == 'low':
        set_dict['brems_param_name'] = 'BremsPetrukhinShestakov'
        set_dict['epair_param_name'] = 'EpairSandrockSoedingreksoRhode'
        set_dict['photo_param_name'] = 'PhotoAbramowiczLevinLevyMaor91'
    elif cross_section_type == 'new':
        set_dict['brems_param_name'] = 'BremsSandrockSoedingreksoRhode'
        set_dict['epair_param_name'] = 'EpairSandrockSoedingreksoRhode'
        set_dict['photo_param_name'] = 'PhotoAbramowiczLevinLevyMaor97'
    else:
        raise KeyError('cross_section_type is not correct')

    build_folder = os.path.join(set_dict['build_path'], cross_section_type)
    if not os.path.isdir(build_folder):
        os.mkdir(build_folder)

    # set muon energies for dEdx calculation for fit
    set_dict['dedx_n_muons'] = 100
    set_dict['dedx_muon_energy_min'] = 1e4 # MeV
    set_dict['dedx_muon_energy_max'] = 1e11 # MeV
    tmp_arr = np.logspace(np.log10(set_dict['dedx_muon_energy_min']),
                                   np.log10(set_dict['dedx_muon_energy_max']),
                                   set_dict['dedx_n_muons'])
    set_dict['dedx_energies'] = tmp_arr.tolist()
    set_dict['dedx_data_filename_all'] = os.path.join(build_folder, 'data_dedx_all_cross_sections.txt')
    set_dict['dedx_data_filename_sum'] = os.path.join(build_folder, 'data_dedx_sum_cross_sections.txt')

    # step2 do fit
    set_dict['dedx_data_fitparams'] = os.path.join(build_folder, 'data_dedx_fitparams.txt')
    set_dict['dedx_plot_test_fitter'] = os.path.join(build_folder, 'plot_dedx_test_fitter.png')
    set_dict['dedx_plot'] = os.path.join(build_folder, 'plot_dedx.png')

    # step3 propagate muons for range distribution
    set_dict['prop_oversampling'] = 100
    set_dict['prop_n_muon_energy_bins'] = 10
    set_dict['prop_muon_energy_min'] = 1e4 # MeV
    set_dict['prop_muon_energy_max'] = 1e11 # MeV
    tmp_arr = np.logspace(np.log10(set_dict['prop_muon_energy_min']),
                          np.log10(set_dict['prop_muon_energy_max']),
                          2 * set_dict['prop_n_muon_energy_bins'] + 1)
    set_dict['prop_energy_bin_edges'] = tmp_arr[::2].tolist()
    set_dict['prop_energy_bin_mids'] = tmp_arr[1::2].tolist()
    set_dict['prop_data_ranges'] = os.path.join(build_folder, 'data_prop_ranges.txt')

    # step 4 plot range
    set_dict['prop_plot_ranges'] = os.path.join(build_folder, 'plot_range_distribution.png')
    set_dict['prop_n_range_bins'] = 11

    setting_file_name = '{}_settings.json'.format(cross_section_type)
    with open(os.path.join(set_dict['build_path'], setting_file_name), 'w') as file:
        json.dump(set_dict, fp=file, indent=2, separators=(',', ':'))

def main():
    parser = ArgumentParser()
    parser.add_argument('-t','--type',
                        type=str,
                        dest='cross_section_type',
                        default='baseline',
                        help='type of cross sections: baseline, high, low or new')
    args = parser.parse_args()

    np.random.seed(123)
    create_settings_dict(args.cross_section_type)

if __name__ == "__main__":
    main()
