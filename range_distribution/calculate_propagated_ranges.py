import os
import json
import numpy as np
from tqdm import tqdm
import pyPROPOSAL as pp
from argparse import ArgumentParser

def create_propagator(path_to_inerpolation_tables='~/.local/share/PROPOSAL/tables',
                      brems_param_name='BremsKelnerKokoulinPetrukhin',
                      epair_param_name='EpairKelnerKokoulinPetrukhin',
                      photo_param_name='PhotoAbramowiczLevinLevyMaor97'):
    mu_def = pp.particle.MuMinusDef.get()
    geometry = pp.geometry.Sphere(pp.Vector3D(), 1.e20, 0.0)
    ecut = 500
    vcut = -1

    sector_def = pp.SectorDefinition()
    sector_def.cut_settings = pp.EnergyCutSettings(ecut, vcut)
    sector_def.medium = pp.medium.Ice(1.0)
    sector_def.geometry = geometry
    sector_def.scattering_model = pp.scattering.ScatteringModel.NoScattering
    sector_def.crosssection_defs.brems_def.lpm_effect = True
    sector_def.crosssection_defs.epair_def.lpm_effect = True
    sector_def.do_continuous_randomization = False
    sector_def.do_exact_time_calculation = False
    sector_def.crosssection_defs.brems_def.parametrization = pp.parametrization.bremsstrahlung.BremsFactory.get().get_enum_from_str(brems_param_name)
    sector_def.crosssection_defs.epair_def.parametrization = pp.parametrization.pairproduction.EpairFactory.get().get_enum_from_str(epair_param_name)
    sector_def.crosssection_defs.photo_def.parametrization = pp.parametrization.photonuclear.PhotoFactory.get().get_enum_from_str(photo_param_name)

    detector = geometry

    interpolation_def = pp.InterpolationDef()
    interpolation_def.path_to_tables = path_to_inerpolation_tables
    interpolation_def.path_to_tables_readonly = path_to_inerpolation_tables

    return pp.Propagator(mu_def, [sector_def], detector, interpolation_def)

def propagate(prop, energy, oversampling):
    propagation_length = 1e9 # cm
    muon_ranges = np.empty(oversampling)

    for idx in tqdm(range(oversampling)):
        prop.particle.position = pp.Vector3D(0, 0, 0)
        prop.particle.direction = pp.Vector3D(1, 0, 0)
        prop.particle.propagated_distance = 0
        prop.particle.energy = energy
        prop.particle.time = 0

        secondarys = prop.propagate(propagation_length)

        muon_ranges[idx] = prop.particle.propagated_distance

    return muon_ranges


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

    pp.RandomGenerator.get().set_seed(1234)

    ranges = np.empty((settings_dict['prop_n_muon_energy_bins'], settings_dict['prop_oversampling']))
    prop = create_propagator(settings_dict['path_to_inerpolation_tables'],
                            settings_dict['brems_param_name'],
                            settings_dict['epair_param_name'],
                            settings_dict['photo_param_name'],)
    for jdx in tqdm(range(settings_dict['prop_n_muon_energy_bins'])):
        ranges[jdx] = propagate(prop,
                                settings_dict['prop_energy_bin_mids'][jdx],
                                settings_dict['prop_oversampling'])
    np.savetxt(settings_dict['prop_data_ranges'], ranges)

if __name__ == '__main__':
    main()
