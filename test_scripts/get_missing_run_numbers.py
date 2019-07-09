import os
from glob import glob
from argparse import ArgumentParser

def generate_file_list(input_files):
    if os.path.isfile(input_files):
        return [input_files]
    elif os.path.isdir(input_files):
        return [file for folder in os.walk(input_files) for file in glob(os.path.join(folder[0], '*.i3.bz2'))]
    else:
        raise NameError ("no such input file or directory: {}".format(input_files))

def retrieve_run_numbers(file_list):
    base_file_name = os.path.basename(file_list[0])

    # split name for '.' to extract dataset number and replace run number
    split_base_name = base_file_name.split('.')
    numbers_list = [stmp for stmp in split_base_name if stmp.isdigit() and len(stmp) == 6]
    if len(numbers_list) != 2:
        raise NameError('problems splitting file name {}'.format(tmp))

    # the first one is the dataset number, the second one the run number
    idx_list = [split_base_name.index(tmp_num) for tmp_num in numbers_list]

    run_numbers = []
    # get run numbers
    for file_name in file_list:
        base_name = os.path.basename(file_name)
        split_base_name = base_name.split('.')
        run_numbers.append(int(split_base_name[idx_list[1]]))

    return run_numbers

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-i', '--input',
        type=str,
        dest='input_file_dir',
        default='/data/user/jsoedingrekso/muongun_crosssections/1904/step_5_pass2_muon_L3/00000-00999/',
        help='input i3file or directory of i3files')
    parser.add_argument(
        '-r', '--runnumber',
        type=int,
        dest='max_run_number',
        default=None,
        help='maximum number of the run numbers')
    args = parser.parse_args()

    file_list = generate_file_list(args.input_file_dir)
    run_numbers = retrieve_run_numbers(file_list)

    max_run_number = args.max_run_number
    if max_run_number is None:
        max_run_number = max(run_numbers)
    missing_run_numbers = [idx for idx in range(max_run_number) if idx not in run_numbers]
    print(missing_run_numbers)

if __name__ == '__main__':
    main()
