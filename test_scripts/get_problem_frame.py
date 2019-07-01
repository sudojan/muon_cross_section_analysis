
from icecube import dataio, icetray, simclasses, dataclasses, phys_services
from matplotlib import pyplot as plt
import numpy as np


def get_problem_frame(file_w_problem, file_wo_problem):
    problem_file = dataio.I3File(file_w_problem)
    comparison_file = dataio.I3File(file_wo_problem)

    minor_ids = []
    # find last frame in problem file
    while(problem_file.more()):
        frame = problem_file.pop_frame()
        # check if end of file
        if(frame == None):
            break
        if not 'MCMuon' in frame.keys():
            continue
        minor_ids.append(frame['MCMuon'].minor_id)

    last_working_event_id = minor_ids[-1]

    # find frame in comparison file
    while (comparison_file.more()):
        frame = comparison_file.pop_frame()
        # check if end of file
        if(frame == None):
            break
        if not 'MCMuon' in frame.keys():
            continue
        if frame['MCMuon'].minor_id == last_working_event_id:
            break

    # get next frame
    while(comparison_file.more()):
        frame = comparison_file.pop_frame()
        if not 'MCMuon' in frame.keys():
            continue
        if frame['MCMuon'].minor_id != last_working_event_id:
            break
    return frame


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-p', '--problemfile',
        type=str,
        dest='file_w_problem',
        help='input i3file that is processed until the error frame')
    parser.add_argument(
        '-c', '--comparisonfile',
        type=str,
        dest='file_wo_problem',
        help='input i3file that is completely processed for comparison')
    args = parser.parse_args()

    frame = get_problem_frame(args.file_w_problem, args.file_wo_problem)
    print(frame['MCMuon'])
    print(frame['MMCTrackList'])


if __name__ == '__main__':
    main()
