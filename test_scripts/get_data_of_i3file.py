
import os
from icecube import icetray, dataio, dataclasses, recclasses, simclasses
from I3Tray import I3Tray
import numpy as np
from tqdm import tqdm
from glob import glob

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
    def get_mc_ic3_label(feature):
        if np.isfinite(frame['LabelsDeepLearning'][feature]):
            return frame['LabelsDeepLearning'][feature]
        print('{} is not finite: {}'.format(feature, frame['LabelsDeepLearning'][feature]))
        return 0.0

    def get_value_of_i3double(feature):
        if np.isfinite(frame[feature].value):
            return frame[feature].value
        print('{} is not finite: {}'.format(feature, frame[feature].value))
        return 0.0

    def get_coordinates_of_i3position(feature):
        pos = []
        for idx in range(3):
            if np.isfinite(frame[feature][idx]):
                pos.append(frame[feature][idx])
            else:
                print('{} at {} is not finite: {}'.format(feature, idx, frame[feature][idx]))
                pos.append(0.)
        return pos

    def get_energy_of_i3particle(feature):
        if feature in frame:
            return frame[feature].energy
        else:
            return 0.0


    frame_labels = []

    frame_labels.append(frame['MMCTrackList'][0].Ec)

    # get MC labels
    for feature in mc_label_names:
        frame_labels.append(get_mc_ic3_label(feature))

    # get DNN reco doubles
    for feature in dnn_double_names:
        frame_labels.append(get_value_of_i3double(feature))

    # get DNN reco positions
    for feature in dnn_positions_names:
        frame_labels += get_coordinates_of_i3position(feature)

    # possible SplineMPE Truncated Energy recos
    for feature in truncated_recos:
        frame_labels.append(get_energy_of_i3particle(feature))

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
            frame = i3file.pop_frame()
            # check if end of file
            if(frame == None):
                break
            # check if its a gcd frame, daq frame or a physics frame
            if 'SplineMPE' not in frame:
                continue

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
    frame_labels_names += mc_label_names
    frame_labels_names += dnn_double_names

    dnn_positions_names_enlarged = []
    for idx in range(len(dnn_positions_names)-1) + [-1]:
        dnn_positions_names_enlarged += [dnn_positions_names[idx] + xyz for xyz in ['_x', '_y', '_z']]
    frame_labels_names += dnn_positions_names_enlarged

    frame_labels_names += truncated_recos

    saving_arr = np.array(labels_list)
    np.savetxt(output_file, saving_arr, header=' '.join(frame_labels_names))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input',
        type=str,
        dest='input_file_dir',
        default='/data/user/jsoedingrekso/muongun_crosssections/1904/step_7_dnn_reco/00000-00999/Level3.2_muongun_singlemuons_IC86.pass2.001904.000012.i3.bz2',
        help='input i3file or directory of i3files')
    parser.add_argument(
        '-o', '--output',
        type=str,
        dest='output_file',
        default='feature_list.txt',
        help='output file with list of features')
    args = parser.parse_args()

    retrieve_data_out_of_i3_file(args.input_file_dir, args.output_file)

if __name__ == '__main__':
    main()
