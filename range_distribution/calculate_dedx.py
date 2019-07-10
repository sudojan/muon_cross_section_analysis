import os
import json
import numpy as np
import pyPROPOSAL as pp
from argparse import ArgumentParser

def create_cross_section_calculators(path_to_inerpolation_tables='~/.local/share/PROPOSAL/tables',
                                    brems_param_name='BremsKelnerKokoulinPetrukhin',
                                    epair_param_name='EpairKelnerKokoulinPetrukhin',
                                    photo_param_name='PhotoAbramowiczLevinLevyMaor97'):
    mu = pp.particle.MuMinusDef.get()
    medium = pp.medium.Ice(1.0)
    cuts = pp.EnergyCutSettings(-1, -1)
    multiplier_all = 1.0
    lpm = True
    interpolation_def = pp.InterpolationDef()
    interpolation_def.path_to_tables = path_to_inerpolation_tables
    interpolation_def.path_to_tables_readonly = path_to_inerpolation_tables

    brems_param = pp.parametrization.bremsstrahlung.BremsFactory.get().get_enum_from_str(brems_param_name)
    brems_def = pp.parametrization.bremsstrahlung.BremsDefinition()
    brems_def.parametrization = brems_param
    brems_def.lpm_effect = lpm
    brems_def.multiplier = multiplier_all
    brems = pp.parametrization.bremsstrahlung.BremsFactory.get().create_bremsstrahlung_interpol(
        mu, medium, cuts, brems_def, interpolation_def)
    # brems = pp.crosssection.BremsInterpolant(
    #             pp.parametrization.bremsstrahlung.KelnerKokoulinPetrukhin(
    #                 mu,
    #                 medium,
    #                 cuts,
    #                 multiplier_all,
    #                 lpm),
    #             interpolation_def)

    epair_param = pp.parametrization.pairproduction.EpairFactory.get().get_enum_from_str(epair_param_name)
    epair_def = pp.parametrization.pairproduction.EpairDefinition()
    epair_def.parametrization = epair_param
    epair_def.lpm_effect = lpm
    epair_def.multiplier = multiplier_all
    epair = pp.parametrization.pairproduction.EpairFactory.get().create_pairproduction_interpol(
        mu, medium, cuts, epair_def, interpolation_def)
    # epair = pp.crosssection.EpairInterpolant(
    #             pp.parametrization.pairproduction.KelnerKokoulinPetrukhinInterpolant(
    #                 mu,
    #                 medium,
    #                 cuts,
    #                 multiplier_all,
    #                 lpm,
    #                 interpolation_def),
    #             interpolation_def)

    photo_param = pp.parametrization.photonuclear.PhotoFactory.get().get_enum_from_str(photo_param_name)
    photo_def = pp.parametrization.photonuclear.PhotoDefinition()
    photo_def.parametrization = photo_param
    photo_def.multiplier = multiplier_all
    photo = pp.parametrization.photonuclear.PhotoFactory.get().create_photonuclear_interpol(
        mu, medium, cuts, photo_def, interpolation_def)
    # photo = pp.crosssection.PhotoInterpolant(
    #             pp.parametrization.photonuclear.AbramowiczLevinLevyMaor97Interpolant(
    #                 mu,
    #                 medium,
    #                 cuts,
    #                 multiplier_all,
    #                 pp.parametrization.photonuclear.ShadowButkevichMikhailov(),
    #                 interpolation_def),
    #             interpolation_def)

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

    cs_calc_list = create_cross_section_calculators(settings_dict['path_to_inerpolation_tables'],
                                                    settings_dict['brems_param_name'],
                                                    settings_dict['epair_param_name'],
                                                    settings_dict['photo_param_name'],)
    tmp_arr = np.array(settings_dict['dedx_energies'])
    dedx_arr = calculate_dedx(cs_calc_list, tmp_arr)
    dedx_sum = np.sum(dedx_arr, axis=1)

    np.savetxt(settings_dict['dedx_data_filename_all'], dedx_arr)
    np.savetxt(settings_dict['dedx_data_filename_sum'], dedx_sum)

if __name__ == '__main__':
    main()
