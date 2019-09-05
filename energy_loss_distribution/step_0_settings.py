
import json
import numpy as np
import os


def create_settings_dict():
    settings_dict = {}

    # set multiplier for which the linear parametrization is build on
    multiplier_min = 0.5
    multiplier_max = 1.5
    multiplier_step = 0.1
    settings_dict['multiplier_build_step'] = 0.1
    brems_multiplier_build_arr = np.arange(multiplier_min, multiplier_max+1e-9, multiplier_step)
    settings_dict['n_buildup_multiplier'] = len(brems_multiplier_build_arr)
    settings_dict["brems_multiplier_buildup_arr"] = brems_multiplier_build_arr.tolist()


    settings_dict['brems_param_name'] = 'BremsKelnerKokoulinPetrukhin'
    settings_dict['epair_param_name'] = 'EpairKelnerKokoulinPetrukhin'
    settings_dict['photo_param_name'] = 'PhotoAbramowiczLevinLevyMaor97'

    settings_dict['brems_param_name_high'] = 'BremsSandrockSoedingreksoRhode'
    settings_dict['epair_param_name_high'] = 'EpairKelnerKokoulinPetrukhin'
    settings_dict['photo_param_name_high'] = 'PhotoBezrukovBugaev'

    settings_dict['brems_param_name_low'] = 'BremsPetrukhinShestakov'
    settings_dict['epair_param_name_low'] = 'EpairSandrockSoedingreksoRhode'
    settings_dict['photo_param_name_low'] = 'PhotoAbramowiczLevinLevyMaor91'

    settings_dict['brems_param_name_new'] = 'BremsSandrockSoedingreksoRhode'
    settings_dict['epair_param_name_new'] = 'EpairSandrockSoedingreksoRhode'
    settings_dict['photo_param_name_new'] = 'PhotoAbramowiczLevinLevyMaor97'


    # set the multiplier against the fitter ist tested
    n_tests_multiplier = 100
    brems_multiplier_test_arr = (multiplier_max - multiplier_min) * np.random.random(n_tests_multiplier) + multiplier_min
    settings_dict["brems_multiplier_testing_arr"] = brems_multiplier_test_arr.tolist()

    # set muon energies between its randomly sampled in log10 (power law)
    settings_dict["n_muons"] = int(1e3)
    settings_dict["n_energy_loss_bins"] = 20
    settings_dict["muon_energy_min"] = 1e7 # MeV
    settings_dict["muon_energy_max"] = 3e7 # MeV
    settings_dict["energy_loss_min"] = 500. # MeV - this is the ecut
    settings_dict["powerlaw_sampler"] = True # - choose between powerlaw or logspace sample
    settings_dict["spectral_index"] = 3.0 # - if powerlaw sampler is used

    # set loss bins
    bin_arr = np.logspace(np.log10(settings_dict["energy_loss_min"]),
                          np.log10(settings_dict["muon_energy_max"]),
                          num = 2 * settings_dict["n_energy_loss_bins"] + 1)
    energy_loss_bin_edges = bin_arr[::2]
    energy_loss_bin_mids = bin_arr[1::2]
    settings_dict["energy_loss_bin_edges"] = energy_loss_bin_edges.tolist()
    settings_dict["energy_loss_bin_mids"] = energy_loss_bin_mids.tolist()

    # set min and max propagation length between which its linear randomly sampled
    settings_dict["propagation_length_min"] = 1e4 # cm
    settings_dict["propagation_length_max"] = 1e5# cm
    settings_dict["bin_len_step"] = 1500. # cm

    # set secondary types
    settings_dict["secondary_types"] = [
        "Pair Production",
        "Bremsstrahlung",
        "Ionization",
        "Photonuclear",
        "Decay Electron",
        'MuPair',
        'MuPair_secondaries',
        'Weak'
        "Binned Track",
    ]

    # set path to interpolation tables
    settings_dict["path_interpolation_tables_buildup"] = "~/.local/share/PROPOSAL/tables"
    # the tables for the random test multiplier shouldn't be stored
    settings_dict["path_interpolation_tables_testing"] = ""

    # create build directory in current path
    current_path = os.getcwd()
    # or use src path
    # current_path = os.path.dirname(os.path.abspath(__file__))
    settings_dict["build_path"] = os.path.join(current_path, "build")
    if not os.path.isdir(settings_dict["build_path"]):
        os.mkdir(settings_dict["build_path"])

    # set directory for step 01
    settings_dict["step01_path"] = os.path.join(settings_dict["build_path"],
                                                "step01")
    for idx in ["buildup", "testing"]:
        settings_dict["step01_path_{}".format(idx)] = os.path.join(settings_dict["step01_path"],
                                                                   idx)
        settings_dict["step01_path_{}_data".format(idx)] = os.path.join(settings_dict["step01_path_{}".format(idx)],
                                                                        "data")
        settings_dict["step01_path_{}_plots".format(idx)] = os.path.join(settings_dict["step01_path_{}".format(idx)],
                                                                         "plots")

        settings_dict["step01_file_{}_data".format(idx)] = os.path.join(settings_dict["step01_path_{}_data".format(idx)],
                                                                        "losses_bins_{:.4}.txt")
        settings_dict["step01_file_{}_err_data".format(idx)] = os.path.join(settings_dict["step01_path_{}_data".format(idx)],
                                                                        "losses_err_{:.4}.txt")
        settings_dict["step01_file_{}_plots".format(idx)] = os.path.join(settings_dict["step01_path_{}_plots".format(idx)],
                                                                         "losses_spectrum_{:.4}.pdf")
        settings_dict["step01_file_multiplier_compare_{}_plot".format(idx)] = os.path.join(settings_dict["step01_path_{}_plots".format(idx)],
                                                                         "multiplier_compare.pdf")

    # set directory for step 02
    settings_dict["step02_path"] = os.path.join(settings_dict["build_path"],
                                                "step02")
    settings_dict["step02_path_plots"] = os.path.join(settings_dict["step02_path"],
                                                      "plots")
    settings_dict["step02_file_plots"] = os.path.join(settings_dict["step02_path_plots"],
                                                      "bin_diff_fit_{}.pdf")
    settings_dict["step02_file_param_bin_diff"] = os.path.join(settings_dict["step02_path"],
                                                               "param_bin_diff.txt")

    # set directory for step 03
    settings_dict["step03_path"] = os.path.join(settings_dict["build_path"],
                                                "step03")
    settings_dict["step03_file_fit_result"] = os.path.join(settings_dict["step03_path"],
                                                            "fit_results.txt")
    settings_dict["step03_file_fit_resolution"] = os.path.join(settings_dict["step03_path"],
                                                               "fit_resolution.pdf")
    settings_dict["step03_file_pullplot"] = os.path.join(settings_dict["step03_path"],
                                                         "pull_dist.pdf")

    with open(os.path.join(settings_dict["build_path"], "settings.json"), "w") as file:
        json.dump(settings_dict, fp=file, indent=2, separators=(",", ":"))


if __name__ == "__main__":
    np.random.seed(123)
    create_settings_dict()
