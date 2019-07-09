import os
import numpy as np
import pyPROPOSAL as pp
from argparse import ArgumentParser

def create_cross_section_calculators():
    mu = pp.particle.MuMinusDef.get()
    medium = pp.medium.Ice(1.0)
    cuts = pp.EnergyCutSettings(-1, -1)
    multiplier_all = 1.0
    lpm = True
    interpolation_def = pp.InterpolationDef()
    interpolation_def.path_to_tables = "~/.local/share/PROPOSAL/tables"
    interpolation_def.path_to_tables_readonly = "~/.local/share/PROPOSAL/tables"

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
    parser.add_argument(
        '-i', '--emin',
        type=float,
        dest='energy_min_log',
        default=4,
        help='log10 of minimum energy in MeV for dEdx')
    parser.add_argument(
        '-f', '--emax',
        type=float,
        dest='energy_max_log',
        default=11,
        help='log10 of maximum energy in MeV for dEdx')
    parser.add_argument(
        '-n', '--nenergies',
        type=int,
        dest='num_energies',
        default=100,
        help='number of energies to evaluate for dEdx')
    args = parser.parse_args()

    energies = np.logspace(args.energy_min_log, args.energy_max_log, args.num_energies)

    script_folder = os.path.dirname(os.path.abspath(__file__))
    build_folder = os.path.join(script_folder, 'build')
    if not os.path.isdir(build_folder):
        os.makedirs(build_folder)

    data_filename_all = os.path.join(build_folder, 'data_dedx_all_cross_sections.txt')
    data_filename_sum = os.path.join(build_folder, 'data_dedx_sum.txt')

    cs_calc_list = create_cross_section_calculators()
    dedx_arr = calculate_dedx(cs_calc_list, energies)
    dedx_sum = np.sum(dedx_arr, axis=1)
    np.savetxt(data_filename_all, dedx_arr)
    np.savetxt(data_filename_sum, dedx_sum)

if __name__ == '__main__':
    main()
