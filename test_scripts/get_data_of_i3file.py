
import os
import gzip
from icecube import icetray, dataio, dataclasses, recclasses, simclasses
from I3Tray import I3Tray
import numpy as np
from tqdm import tqdm
from glob import glob
from argparse import ArgumentParser

def generate_file_list(input_file_or_dir):
    if os.path.isfile(input_file_or_dir):
        return [input_file_or_dir]
    elif os.path.isdir(input_file_or_dir):
        return [file for folder in os.walk(input_file_or_dir) for file in glob(os.path.join(folder[0], '*.i3.bz2'))]
    else:
        raise NameError ("no such input file or directory: {}".format(input_file_or_dir))

def gather_information_of_frame(frame,
                                mc_label_names,
                                dnn_double_names,
                                dnn_positions_names,
                                truncated_recos):
    def extract_data_of_label(feature, frame_key=None, particle_attribute='energy', num_output_values=1):
        if frame_key is None:
            if feature in frame:
                if type(frame[feature]) == dataclasses.I3Double:
                    if np.isfinite(frame[feature].value):
                        return frame[feature].value
                    print('{} is not finite: {}'.format(feature, frame[feature].value))
                    # return 0.
                elif type(frame[feature]) == dataclasses.I3Position:
                    pos = []
                    for idx in range(3):
                        if np.isfinite(frame[feature][idx]):
                            pos.append(frame[feature][idx])
                        else:
                            print('{} at {} is not finite: {}'.format(feature, idx, frame[feature][idx]))
                            pos.append(0.)
                    return pos
                elif type(frame[feature]) == dataclasses.I3Particle:
                    if particle_attribute == 'energy':
                        if np.isfinite(frame[feature].energy):
                            return frame[feature].energy
                        else:
                            print('{} is not finite: {}'.format(feature, frame[feature].energy))
                            # return 0.
                    else:
                        NameError('particle_attribute {} not implemented yet'.format(particle_attribute))
                else:
                    NameError('extraction for dataclasses type {} not implemented yet'.format(type(frame[feature])))
            # print('{} not in frame'.format(feature))

        else:
            if frame_key in frame and type(frame[frame_key]) == dataclasses.I3MapStringDouble:
                if feature in frame[frame_key]:
                    if np.isfinite(frame[frame_key][feature]):
                        return frame[frame_key][feature]
                    else:
                        print('{} is not finite: {}'.format(feature, frame[frame_key][feature]))
                        # return 0.0
                # print('{} not in MapString {}'.format(feature, frame_key))
            # print('{} not in frame'.format(feature))

        if num_output_values == 1:
            return 0.
        elif num_output_values > 1:
            return np.zeros(num_output_values).tolist()


    frame_labels = []

    center_energy = 0.
    if 'MMCTrackList' in frame:
        if len(frame['MMCTrackList']) > 0:
            center_energy = frame['MMCTrackList'][0].Ec
    frame_labels.append(center_energy)

    num_pulses = 0
    charge_total = 0
    if 'InIceDSTPulses' in frame:
        pulse_series = frame['InIceDSTPulses'].apply(frame)
        for omkey in pulse_series.keys():
            for pulse in pulse_series[omkey]:
                num_pulses += 1
                charge_total += pulse.charge
    frame_labels.append(num_pulses)
    frame_labels.append(charge_total)

    # get MC labels
    for feature in mc_label_names:
        frame_labels.append(extract_data_of_label(feature, 'LabelsDeepLearning'))

    # get DNN reco doubles
    for feature in dnn_double_names:
        frame_labels.append(extract_data_of_label(feature))

    # get DNN reco positions
    for feature in dnn_positions_names:
        frame_labels += extract_data_of_label(feature, num_output_values=3)

    # possible SplineMPE Truncated Energy recos
    for feature in truncated_recos:
        frame_labels.append(extract_data_of_label(feature))

    return frame_labels

def retrieve_data_out_of_i3_file(input_file_or_dir, output_file):

    labels_list = []
    file_list = generate_file_list(input_file_or_dir)

    mc_label_names = [
        'MostVisibleMuonEnergyEntry',
        'MostVisibleMuonInDetectorTrackLength',
        'MostVisibleMuonEntryx',
        'MostVisibleMuonEntryy',
        'MostVisibleMuonEntryz',
        'MostVisibleMuonExitx',
        'MostVisibleMuonExity',
        'MostVisibleMuonExitz',
    ]
    dnn_double_names = [
        'DeepLearning_PrimaryMuonEnergyEntry',
        'DeepLearning_PrimaryMuonEnergyEntry_log_uncertainty',
    ]
    dnn_positions_names = [
        'DeepLearning_MuonEntryPoint',
        'DeepLearning_MuonExitPoint',
    ]
    truncated_recos = [
        'SplineMPETruncatedEnergy_SPICEMie_AllBINS_Muon',
        'SplineMPETruncatedEnergy_SPICEMie_AllDOMS_Muon',
        'SplineMPETruncatedEnergy_SPICEMie_BINS_Muon',
        'SplineMPETruncatedEnergy_SPICEMie_DOMS_Muon',
        'SplineMPETruncatedEnergy_SPICEMie_ORIG_Muon',
        'SplineMPEMuEXDifferential',
    ]


    for input_file in tqdm(file_list):
        i3file = dataio.I3File(input_file)
        while(i3file.more()):
            frame = i3file.pop_physics()
            # frame = i3file.pop_frame()
            # check if end of file
            if(frame == None):
                break
            # check if its a gcd frame, daq frame or a physics frame
            if 'SplineMPE' not in frame:
                continue
            #if 'MCMuon' not in frame:
            #    continue

            labels_list.append(
                gather_information_of_frame(
                    frame,
                    mc_label_names,
                    dnn_double_names,
                    dnn_positions_names,
                    truncated_recos))

            #if len(labels_list) > 10:
            #    break

    print('num frames: {}'.format(len(labels_list)))

    frame_labels_names = ['MMCTrackList_CenterEnergy',]
    frame_labels_names += ['InIceDSTPulses_NumPulses', 'InIceDSTPulses_Qtot']
    frame_labels_names += mc_label_names
    frame_labels_names += dnn_double_names

    dnn_positions_names_enlarged = []
    for idx in range(len(dnn_positions_names)-1) + [-1]:
        dnn_positions_names_enlarged += [dnn_positions_names[idx] + xyz for xyz in ['_x', '_y', '_z']]
    frame_labels_names += dnn_positions_names_enlarged

    frame_labels_names += truncated_recos

    saving_arr = np.array(labels_list)

    # delete coulumns, that have no entry
    index_to_delete = []
    for idx in range(len(saving_arr[0])):
        if np.count_nonzero(saving_arr[:,idx]) == 0:
            index_to_delete.append(idx)
    saving_arr = np.delete(saving_arr, index_to_delete, axis=1)
    # delete also from fram labelnames
    for idx in index_to_delete[::-1]:
        del frame_labels_names[idx]

    if output_file.endswith(".gz"):
        with gzip.open(output_file, 'w') as file:
            np.savetxt(file, saving_arr, header=' '.join(frame_labels_names))
    else:
        with open(output_file, 'w') as file:
            np.savetxt(file, saving_arr, header=' '.join(frame_labels_names))


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-i', '--input',
        type=str,
        dest='input_file_dir',
        help='input i3file or directory of i3files')
    parser.add_argument(
        '-o', '--output',
        type=str,
        dest='output_file',
        help='output file with list of features')
    args = parser.parse_args()

    script_folder = os.path.dirname(os.path.abspath(__file__))
    build_folder = os.path.join(script_folder, 'build')
    if not os.path.isdir(build_folder):
        os.makedirs(build_folder)
    output_file = os.path.join(build_folder, args.output_file)

    retrieve_data_out_of_i3_file(args.input_file_dir, output_file)

if __name__ == '__main__':
    main()
