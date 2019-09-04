
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
        radicant = (u * (xhig**(1. - gamma) - xlow**(1. - gamma)) + xlow**(1. - gamma))
        return radicant**(1. / (1. - gamma))

def create_propagator(path_to_interpolation_tables="~/.local/share/PROPOSAL/tables",
                      brems_multiplier=1.0,
                      brems_param_name='BremsKelnerKokoulinPetrukhin',
                      epair_param_name='EpairKelnerKokoulinPetrukhin',
                      photo_param_name='PhotoAbramowiczLevinLevyMaor97',
                      ecut=500,
                      vcut=-1,
                      lpm=True,
                      mupair_interaction=False,
                      mupair_singlemuons=False,
                      weak_interaction=False):
    particle_def=pp.particle.MuMinusDef.get()
    geometry = pp.geometry.Sphere(pp.Vector3D(), 1.e20, 0.0)

    sector_def = pp.SectorDefinition()
    sector_def.cut_settings = pp.EnergyCutSettings(ecut, vcut)
    sector_def.medium = pp.medium.Ice(1.0)
    sector_def.geometry = geometry
    sector_def.scattering_model = pp.scattering.ScatteringModel.NoScattering
    sector_def.do_continuous_randomization = False
    sector_def.do_exact_time_calculation = False
    sector_def.crosssection_defs.mupair_def.mupair_enable = mupair_interaction
    sector_def.crosssection_defs.mupair_def.particle_output = mupair_singlemuons
    sector_def.crosssection_defs.weak_def.weak_enable = weak_interaction
    sector_def.crosssection_defs.brems_def.lpm_effect = lpm
    sector_def.crosssection_defs.epair_def.lpm_effect = lpm
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
    mupair_secondary_energies = []
    mupair_muon_energies = []
    weak_energies = []

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
    is_single_muon_from_mupair = False
    for sec in secondaries:
        if sec.id == pp.particle.Data.Epair:
            epair_energies.append(sec.energy)
        elif sec.id == pp.particle.Data.Brems:
            brems_energies.append(sec.energy)
        elif sec.id == pp.particle.Data.DeltaE:
            ioniz_energies.append(sec.energy)
        elif sec.id == pp.particle.Data.NuclInt:
            photo_energies.append(sec.energy)
        elif sec.id == pp.particle.Data.MuPair:
            mupair_secondary_energies.append(sec.energy)
        elif sec.id == pp.particle.Data.Particle:
            # mupair particle output
            if sec.particle_def == pp.particle.MuMinusDef.get():
                mupair_muon_energies.append(sec.energy)
                is_single_muon_from_mupair = True
            elif sec.particle_def == pp.particle.MuPlusDef.get():
                mupair_muon_energies.append(sec.energy)
                is_single_muon_from_mupair = True
            # decay
            elif sec.particle_def == pp.particle.EMinusDef.get():
                decay_energies.append(sec.energy)
            elif sec.particle_def == pp.particle.EPlusDef.get():
                decay_energies.append(sec.energy)
            elif sec.particle_def.name[:2] == 'Nu':
                # neutrino energies dont count
                continue
            else:
                print("unknown decay particle")
                print(sec.id, sec.particle_def)
        elif sec.id == pp.particle.Data.WeakInt:
            weak_energies.append(sec.energy)
        else:
            print("unknown secondary type")
            print(sec.id)

        if do_len_binning:
            if is_single_muon_from_mupair:
                # the single muons will further be propagated
                # therefore don't count them in the binned energy losses
                is_single_muon_from_mupair = False
                continue

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

    secondaries = [epair_energies,
                   brems_energies,
                   ioniz_energies,
                   photo_energies,
                   decay_energies,
                   mupair_secondary_energies,
                   mupair_muon_energies,
                   weak_energies,
                   binned_energies
    ]
    return secondaries

def prop_particle(prop, energy, max_propagation_len=1e20):
    prop.particle.position = pp.Vector3D(0, 0, 0)
    prop.particle.direction = pp.Vector3D(1, 0, 0)
    prop.particle.propagated_distance = 0
    prop.particle.energy = energy
    prop.particle.time = 0

    return prop.propagate(max_propagation_len)


def propagate_and_return_secondary_hist(prop, muon_energies, propagation_lengths, loss_bin_edges, len_bins):
    n_sec_types = 9
    nbins = len(loss_bin_edges)-1
    secondary_bins = np.zeros((n_sec_types, nbins))
    secondary_errs = np.zeros((n_sec_types, nbins))

    for idx in tqdm(range(len(muon_energies))):
        secondaries = prop_particle(prop, muon_energies[idx], propagation_lengths[idx])

        if len(secondaries) < 1:
            continue

        sec_energies = classify_secondaries(secondaries, len_bins)

        # midx = 0
        sum_2nd_mus = []
        while(len(sec_energies[5]) > 0):
            # midx += 1
            # print('muon iter: {}'.format(midx))
            for jdx in sec_energies[5]:
                secs2 = prop_particle(prop, jdx, propagation_lengths[idx])
                if len(secs2) < 1:
                    continue
                sec_energies2 = classify_secondaries(secs2, len_bins)
                for kdx in range(len(sec_energies2)):
                    if kdx != 5:
                        sec_energies[kdx].extend(sec_energies2[kdx])
                        sum_2nd_mus.extend(sec_energies2[kdx])
                    else:
                        sec_energies[kdx] = sec_energies2[kdx]
            if len(sec_energies[5]) > 0:
                print('nmuons left: {}'.format(sec_energies[5]))
        # del sec_energies[5]
        sec_energies[5] = sum_2nd_mus


        for jdx in range(n_sec_types):
            # norm to 100 m propagated distance
            weights = 1e4 / propagation_lengths[idx] * np.ones(len(sec_energies[jdx]))
            secondary_bins_tmp = np.histogram(sec_energies[jdx],
                                                bins=loss_bin_edges,
                                                weights=weights,
                                                density=False)[0]
            sec_err_tmp = np.histogram(sec_energies[jdx],
                                                bins=loss_bin_edges,
                                                weights=weights**2,
                                                density=False)[0]
            secondary_bins[jdx] += secondary_bins_tmp
            secondary_errs[jdx] += sec_err_tmp
    
    secondary_errs = np.sqrt(secondary_errs)

    # returns histogramed secondaries per muon per 100 meter
    return secondary_bins / float(len(muon_energies)), secondary_errs / float(len(muon_energies))


def create_secondaries_hist(settings_dict, overwrite, style):
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
        err_file_name = settings_dict["step01_file_{}_err_data".format(style)].format(brems_multiplier)
        if os.path.isfile(bin_file_name) and os.path.isfile(err_file_name) and not overwrite:
            continue

        prop = create_propagator(settings_dict["path_interpolation_tables_{}".format(style)],
                                brems_multiplier)
        sec_bins, sec_errs = propagate_and_return_secondary_hist(prop,
                                                                muon_energies,
                                                                propagation_lengths,
                                                                loss_bin_edges,
                                                                len_bins)
        np.savetxt(bin_file_name, sec_bins)
        np.savetxt(err_file_name, sec_errs)


def main():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config',
                        type=str,
                        dest='settings_file', default="build/settings.json",
                        help='json file containing the settings')
    parser.add_argument('-f', '--force',
                        type=bool,
                        dest='overwrite', default=False,
                        help='recalculate histogram and propagate')
    args = parser.parse_args()

    pp.RandomGenerator.get().set_seed(1234)
    np.random.seed(123)

    with open(args.settings_file) as file:
        settings_dict = json.load(file)

    # init file names and directories
    if not os.path.isdir(settings_dict["step01_path"]):
        os.mkdir(settings_dict["step01_path"])

    # simulate with different multiplier to param bin diffs
    create_secondaries_hist(settings_dict, args.overwrite, style="buildup")
    # create test multiplier datasets
    # create_secondaries_hist(settings_dict, overwrite, style="testing")


if __name__ == "__main__":
    main()
