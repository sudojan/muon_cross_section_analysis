import os
import json
import numpy as np
import pyPROPOSAL as pp
from argparse import ArgumentParser

def create_cross_section_calculators(path_to_inerpolation_tables='~/.local/share/PROPOSAL/tables'):
    mu = pp.particle.MuMinusDef.get()
    medium = pp.medium.Ice(1.0)
    cuts = pp.EnergyCutSettings(-1, -1)
    multiplier_all = 1.0
    lpm = True
    interpolation_def = pp.InterpolationDef()
    interpolation_def.path_to_tables = path_to_inerpolation_tables
    interpolation_def.path_to_tables_readonly = path_to_inerpolation_tables

    brems = pp.crosssection.BremsInterpolant(
                pp.parametrization.bremsstrahlung.KelnerKokoulinPetrukhin(
                    mu,
                    medium,
                    cuts,
                    multiplier_all,
                    lpm),
                interpolation_def)

    photo = pp.crosssection.PhotoInterpolant(
                pp.parametrization.photonuclear.AbramowiczLevinLevyMaor97Interpolant(
                    mu,
                    medium,
                    cuts,
                    multiplier_all,
                    pp.parametrization.photonuclear.ShadowButkevichMikhailov(),
                    interpolation_def),
                interpolation_def)

    epair = pp.crosssection.EpairInterpolant(
                pp.parametrization.pairproduction.KelnerKokoulinPetrukhinInterpolant(
                    mu,
                    medium,
                    cuts,
                    multiplier_all,
                    lpm,
                    interpolation_def),
                interpolation_def)

    ioniz = pp.crosssection.IonizInterpolant(
                pp.parametrization.ionization.Ionization(
                    mu,
                    medium,
                    cuts,
                    lpm),
                interpolation_def)
    return [ioniz, brems, epair, photo]

def calculate_dedx(cs_calc_list, energies):
    dedx_arr = np.empty((len(energies), len(cs_calc_list)))
    for idx in range(len(energies)):
        for jdx in range(len(cs_calc_list)):
            dedx_arr[idx, jdx] = cs_calc_list[jdx].calculate_dEdx(energies[idx])
    return dedx_arr

def main():
    parser = ArgumentParser()
    parser.add_argument('-f','--file',
                        type=str,
                        dest='settings_file',
                        default="build/settings.json",
                        help='json file containing the settings')
    args = parser.parse_args()

    with open(args.settings_file) as file:
        settings_dict = json.load(file)

    cs_calc_list = create_cross_section_calculators(settings_dict['path_to_inerpolation_tables'])
    tmp_arr = np.array(settings_dict['dedx_energies'])
    dedx_arr = calculate_dedx(cs_calc_list, tmp_arr)
    dedx_sum = np.sum(dedx_arr, axis=1)

    np.savetxt(settings_dict['dedx_data_filename_all'], dedx_arr)
    np.savetxt(settings_dict['dedx_data_filename_sum'], dedx_sum)

if __name__ == '__main__':
    main()
