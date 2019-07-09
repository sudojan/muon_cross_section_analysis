import os
import numpy as np
from tqdm import tqdm
import pyPROPOSAL as pp

def create_propagator():
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

    detector = geometry

    interpolation_def = pp.InterpolationDef()
    interpolation_def.path_to_tables = "~/.local/share/PROPOSAL/tables"
    interpolation_def.path_to_tables_readonly = "~/.local/share/PROPOSAL/tables"

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

def calc_ranges(energies, oversampling):
    ranges = np.empty((len(energies), oversampling))
    prop = create_propagator()
    for jdx in tqdm(range(len(energies))):
        ranges[jdx] = propagate(prop, energies[jdx], oversampling)
    return ranges

def main():
    energies = np.logspace(4, 11, 10)
    pp.RandomGenerator.get().set_seed(1234)
    data_filename = 'build/dedx_data_range.txt'
    ranges = calc_ranges(energies, oversampling=100)
    np.savetxt(data_filename, ranges)

if __name__ == '__main__':
    main()
