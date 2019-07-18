
import os
import gzip
from icecube import icetray, dataio, dataclasses, recclasses, simclasses
from I3Tray import I3Tray
import numpy as np
from tqdm import tqdm
from glob import glob
from scipy.spatial import Delaunay

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))

def generate_file_list(input_file_or_dir):
    if os.path.isfile(input_file_or_dir):
        return [input_file_or_dir]
    elif os.path.isdir(input_file_or_dir):
        return [file for folder in os.walk(input_file_or_dir) for file in glob(os.path.join(folder[0], '*.i3.bz2'))]
    else:
        raise NameError ("no such input file or directory: {}".format(input_file_or_dir))

def create_icecube_delaunay():
    z_min = -502.0
    z_max =  501.0
    edge_points = np.array([
        [-570.90002441, -125.13999939, z_max], # string 31
        [-256.14001465, -521.08001709, z_max], # string 1
        [ 361.        , -422.82998657, z_max], # string 6
        [ 576.36999512,  170.91999817, z_max], # string 50
        [ 338.44000244,  463.72000122, z_max], # string 74
        [  22.11000061,  509.5       , z_max], # string 78
        [-347.88000488,  451.51998901, z_max], # string 75
        [-570.90002441, -125.13999939, z_min], # string 31
        [-256.14001465, -521.08001709, z_min], # string 1
        [ 361.        , -422.82998657, z_min], # string 6
        [ 576.36999512,  170.91999817, z_min], # string 50
        [ 338.44000244,  463.72000122, z_min], # string 74
        [  22.11000061,  509.5       , z_min], # string 78
        [-347.88000488,  451.51998901, z_min], # string 75
    ])
    return Delaunay(edge_points)

def points_in_detector(delaunay, positions):
    r"""
    Calculate mask array deciding whether points are inside the detector

    Parameters
    ----------
    positions : array-like
        3d point or array of 3d points

    Returns
    -------
    is_inside_mask : array-like
        bool array which index is True if the point is inside the detector and False if not
    """
    return delaunay.find_simplex(positions) >= 0


def get_losses_of_i3files(file_list):
    losses_list = []
    longest_event = 0
    miliped_key = 'SplineMPE_MillipedeHighEnergyMIE'
    delaunay = create_icecube_delaunay()

    for input_file in tqdm(file_list):
        i3file = dataio.I3File(input_file)
        while(i3file.more()):
            frame = i3file.pop_frame()
            # check if end of file
            if(frame == None):
                break
            # check if its a gcd frame, daq frame or a physics frame
            if miliped_key not in frame:
                continue

            event_losses_inside = np.zeros(150)
            frame_idx = 0
            milipede_list = frame[miliped_key]
            for loss_bin in milipede_list:
                if loss_bin.energy > 0:
                    if points_in_detector(delaunay, [loss_bin.pos.x, loss_bin.pos.y, loss_bin.pos.z]):
                        event_losses_inside[frame_idx] = loss_bin.energy
                        frame_idx += 1

            losses_list.append(event_losses_inside)
            if np.count_nonzero(event_losses_inside) > longest_event:
                longest_event = np.count_nonzero(event_losses_inside)

    print('num events: ', len(losses_list))
    print('most nonzero bins: ', longest_event)
    return losses_list

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
        default='loss_list.txt.gz',
        help='output file with list of features')
    args = parser.parse_args()

    build_folder = os.path.join(script_folder, 'build')
    if not os.path.isdir(build_folder):
        os.makedirs(build_folder)
    output_file = os.path.join(build_folder, args.output_file)

    file_list = generate_file_list(args.input_file_dir)
    datas = get_losses_of_i3files(file_list)

    if output_file.endswith(".gz"):
        with gzip.open(output_file, 'w') as file:
            np.savetxt(file, datas)
    else:
        with open(output_file, 'w') as file:
            np.savetxt(file, datas)


if __name__ == '__main__':
    main()

