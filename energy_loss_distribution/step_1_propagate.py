
import pyPROPOSAL as pp
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import os
import json


def distance_on_line(start, direction, point):
    return - np.dot(start - point, direction) / np.linalg.norm(direction)

def random_logspace_sampler(start, end, num):
    return 10**(np.random.uniform(np.log10(start), np.log10(end), num))

def random_powerlaw_sampler(xlow, xhig, num, gamma=3.7):
    u = np.random.uniform(size=int(num))

    if gamma == 1:
        return np.exp(u * np.log(xhig / xlow)) * xlow
    else:
        radicant = (u * (xhig**(1. - gamma) - xlow**(1. - gamma)) +
                    xlow**(1. - gamma))
        return radicant**(1. / (1. - gamma))

def create_propagator(path_to_interpolation_tables="~/.local/share/PROPOSAL/tables",
                      brems_multiplier=1.0,
                      brems_param_name='BremsKelnerKokoulinPetrukhin',
                      epair_param_name='EpairKelnerKokoulinPetrukhin',
                      photo_param_name='PhotoAbramowiczLevinLevyMaor97'):
    particle_def=pp.particle.MuMinusDef.get()
    geometry = pp.geometry.Sphere(pp.Vector3D(), 1.e20, 0.0)

    sector_def = pp.SectorDefinition()
    sector_def.cut_settings = pp.EnergyCutSettings(500, -1)
    sector_def.medium = pp.medium.Ice(1.0)
    sector_def.geometry = geometry
    sector_def.scattering_model = pp.scattering.ScatteringModel.NoScattering
    sector_def.crosssection_defs.brems_def.lpm_effect = False
    sector_def.crosssection_defs.epair_def.lpm_effect = False
    sector_def.crosssection_defs.brems_def.multiplier = brems_multiplier
    sector_def.crosssection_defs.brems_def.parametrization = pp.parametrization.bremsstrahlung.BremsFactory.get().get_enum_from_str(brems_param_name)
    sector_def.crosssection_defs.epair_def.parametrization = pp.parametrization.pairproduction.EpairFactory.get().get_enum_from_str(epair_param_name)
    sector_def.crosssection_defs.photo_def.parametrization = pp.parametrization.photonuclear.PhotoFactory.get().get_enum_from_str(photo_param_name)

    interpolation_def = pp.InterpolationDef()
    interpolation_def.path_to_tables = path_to_interpolation_tables
    interpolation_def.path_to_tables_readonly = path_to_interpolation_tables

    prop = pp.Propagator(particle_def,
                        [sector_def],
                        geometry,
                        interpolation_def)
    return prop

def classify_secondaries(secondaries, len_bins):
    ioniz_energies = []
    epair_energies = []
    brems_energies = []
    photo_energies = []
    decay_energies = []
    binned_energies = []

    if len(secondaries) == 1:
        # only one interaction
        do_len_binning = False
    elif secondaries[0].id == pp.particle.Data.Particle:
        # directly a decay
        do_len_binning = False
    else:
        do_len_binning = True
        start_point = np.array([secondaries[0].position.x,
                                secondaries[0].position.y,
                                secondaries[0].position.z])
        end_point = np.array([secondaries[-1].position.x,
                              secondaries[-1].position.y,
                              secondaries[-1].position.z])
        direction = end_point - start_point
        start_len = np.linalg.norm(start_point)

    all_energies = []
    all_lengths = []
    for sec in secondaries:

        if sec.id == pp.particle.Data.DeltaE:
            ioniz_energies.append(sec.energy)
        elif sec.id == pp.particle.Data.Epair:
            epair_energies.append(sec.energy)
        elif sec.id == pp.particle.Data.Brems:
            brems_energies.append(sec.energy)
        elif sec.id == pp.particle.Data.NuclInt:
            photo_energies.append(sec.energy)
        elif sec.id == pp.particle.Data.Particle:
            # decay
            if sec.particle_def == pp.particle.EMinusDef.get():
                decay_energies.append(sec.energy)
            elif sec.particle_def == pp.particle.EPlusDef.get():
                decay_energies.append(sec.energy)
            elif sec.particle_def.name[:2] == 'Nu':
                # neutrino energies dont count
                continue
            else:
                print("unknown decay particle")
                print(sec.id)
        else:
            print("unknown secondary type")
            print(sec.id)

        if do_len_binning:
            posi = np.array([sec.position.x,
                             sec.position.y,
                             sec.position.z])
            porp_len = start_len + distance_on_line(start_point, direction, posi)
            all_lengths.append(porp_len)
            all_energies.append(sec.energy)

    if do_len_binning:
        len_indices = np.digitize(all_lengths, len_bins)
        bincount = np.bincount(len_indices, all_energies)
        binned_energies.extend(bincount[bincount>0].tolist())
    else:
        # add the single cascade
        binned_energies.append(secondaries[0].energy)

    return ioniz_energies, epair_energies, brems_energies, photo_energies, decay_energies, binned_energies


def propagate_and_return_secondary_hist(prop, muon_energies, propagation_lengths, loss_bin_edges, len_bins):
    n_sec_types = 6
    secondary_bins = np.zeros((n_sec_types, len(loss_bin_edges)-1))

    for idx in tqdm(range(len(muon_energies))):
        prop.particle.position = pp.Vector3D(0, 0, 0)
        prop.particle.direction = pp.Vector3D(0, 0, -1)
        prop.particle.propagated_distance = 0
        prop.particle.energy = muon_energies[idx]

        secondaries = prop.propagate(propagation_lengths[idx])

        if len(secondaries) < 1:
            print("no secondaries")
            continue

        secondaries_energies = classify_secondaries(secondaries, len_bins)

        for jdx in range(n_sec_types):
            secondary_bins_tmp, _ = np.histogram(secondaries_energies[jdx],
                                                bins=loss_bin_edges,
                                                density=False)
            # norm to 100 m propagated distance
            secondary_bins[jdx] += secondary_bins_tmp * 1e4 / propagation_lengths[idx]

    # returns histogramed secondaries per muon per 100 meter
    return secondary_bins / float(len(muon_energies))


def create_secondaries_hist(settings_dict, style):
    if not os.path.isdir(settings_dict["step01_path_{}".format(style)]):
        os.mkdir(settings_dict["step01_path_{}".format(style)])
    if not os.path.isdir(settings_dict["step01_path_{}_data".format(style)]):
        os.mkdir(settings_dict["step01_path_{}_data".format(style)])

    len_bins = np.arange(start=0,
                        stop=settings_dict["propagation_length_max"],
                        step=settings_dict["bin_len_step"])

    # init energy binning
    nbins = settings_dict["n_energy_loss_bins"]
    loss_bin_edges = np.array(settings_dict["energy_loss_bin_edges"])

    # interaction labels
    n_sec_types = len(settings_dict["secondary_types"])


    for brems_multiplier in settings_dict["brems_multiplier_{}_arr".format(style)]:
        print("brems_multiplier ", brems_multiplier)

        # init muon propagation properties
        if settings_dict["powerlaw_sampler"]:
            muon_energies = random_powerlaw_sampler(settings_dict["muon_energy_min"],
                                                    settings_dict["muon_energy_max"],
                                                    settings_dict["n_muons"],
                                                    settings_dict["spectral_index"])
        else:
            muon_energies = random_logspace_sampler(settings_dict["muon_energy_min"],
                                                    settings_dict["muon_energy_max"],
                                                    settings_dict["n_muons"])

        propagation_lengths = np.random.uniform(settings_dict["propagation_length_min"],
                                                settings_dict["propagation_length_max"],
                                                settings_dict["n_muons"])


        bin_file_name = settings_dict["step01_file_{}_data".format(style)].format(brems_multiplier)
        if os.path.isfile(bin_file_name):
            continue

        prop = create_propagator(settings_dict["path_interpolation_tables_{}".format(style)],
                                brems_multiplier)
        sec_bins = propagate_and_return_secondary_hist(prop,
                                                    muon_energies,
                                                    propagation_lengths,
                                                    loss_bin_edges,
                                                    len_bins)
        np.savetxt(bin_file_name, sec_bins)


def main():
    parser = ArgumentParser()
    parser.add_argument('-f','--file',
                        type=str,
                        dest='settings_file', default="build/settings.json",
                        help='json file containing the settings')
    args = parser.parse_args()

    pp.RandomGenerator.get().set_seed(1234)
    np.random.seed(123)

    with open(args.settings_file) as file:
        settings_dict = json.load(file)

    # init file names and directories
    if not os.path.isdir(settings_dict["step01_path"]):
        os.mkdir(settings_dict["step01_path"])

    # simulate with different multiplier to param bin diffs
    create_secondaries_hist(settings_dict=settings_dict, style="buildup")
    # create test multiplier datasets
    # create_secondaries_hist(settings_dict=settings_dict, style="testing")


if __name__ == "__main__":
    main()
