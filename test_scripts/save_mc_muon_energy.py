
import os
import gzip
from icecube import icetray, dataio, dataclasses, recclasses, simclasses
from I3Tray import I3Tray
import numpy as np
from tqdm import tqdm
from glob import glob
from argparse import ArgumentParser

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))

def generate_file_list(input_file_or_dir):
    if os.path.isfile(input_file_or_dir):
        return [input_file_or_dir]
    elif os.path.isdir(input_file_or_dir):
        return [file for folder in os.walk(input_file_or_dir) for file in glob(os.path.join(folder[0], '*.i3.bz2'))]
    else:
        raise NameError ("no such input file or directory: {}".format(input_file_or_dir))

def get_init_energy_of_i3files(input_file_list):
    energies = []
    tmp_minor_id = -1

    for input_file in tqdm(input_file_list):
        i3file = dataio.I3File(input_file)
        while(i3file.more()):
            frame = i3file.pop_frame()
            # check if end of file
            if(frame == None):
                break
            # check if its a gcd frame, daq frame
            if 'MCMuon' not in frame:
                continue

            if frame['MCMuon'].minor_id != tmp_minor_id:
                tmp_minor_id = frame['MCMuon'].minor_id
                energies.append(frame['MCMuon'].energy)

    return energies

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

    build_folder = os.path.join(SCRIPT_FOLDER, 'build')
    if not os.path.isdir(build_folder):
        os.makedirs(build_folder)
    output_file = os.path.join(build_folder, args.output_file)

    file_list = generate_file_list(args.input_file_dir)
    energies = get_init_energy_of_i3files(file_list)

    print('num frames: {}'.format(len(energies)))

    if output_file.endswith(".gz"):
        with gzip.open(output_file, 'w') as file:
            np.savetxt(file, energies)
    else:
        with open(output_file, 'w') as file:
            np.savetxt(file, energies)


if __name__ == '__main__':
    main()
