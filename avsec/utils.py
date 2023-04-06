
import numpy as np
from seca import analysis


def fp_str(overbaseline=True, style='new'):
    if style == 'old':
        outstr = 'nFP{}'.format('_over_baseline' if overbaseline else '')
    elif style == 'new':
        outstr = 'FP_near_inject'
    else:
        raise RuntimeError('Unknown style')
    return outstr

def fn_str(overbaseline=True, style='new'):
    if style == 'old':
        outstr = 'nFN{}'.format('_over_baseline' if overbaseline else '')
    elif style == 'new':
        outstr = 'FN_of_target'
    else:
        raise RuntimeError('Unknown style')
    return outstr


def find_common_combos(dcm_list):
    """
    Gets the common frames and objects tested by all data collects

    Since each algorithm only runs when the baseline succeeded, not all
    algorithms will test each object from each frame.
    """
    # Build dictionary of which algs did which frames
    frame_inter = set(dcm_list[0]['frame'])

    # Get frame intersection
    for dcm in dcm_list[1:]:
        frame_set = set(dcm['frame'])
        frame_inter = frame_inter.intersection(frame_set)

    # Get double dictionary
    frame_obj_double_dict = []
    for idcm, dcm in enumerate(dcm_list):
        frame_set = set(dcm['frame'])
        fr_arr = np.asarray(dcm['frame'])
        obj_arr = np.asarray(dcm['obj'])
        frame_obj_double_dict.append({frame:{obj:[] for obj in obj_arr[np.argwhere(fr_arr==frame)].flatten()} \
                                      for frame in frame_set})
        # Add indices
        for idx in range(len(dcm['frame'])):
            frame_obj_double_dict[idcm][dcm['frame'][idx]][dcm['obj'][idx]] = idx

    # Get common (frame, iobj) pairs
    common_combos = []
    running_frame_obj_set_dict = {}
    for idcm, dcm in enumerate(dcm_list):
        fr_arr = np.asarray(dcm['frame'])
        obj_arr = np.asarray(dcm['obj'])
        for frame in frame_inter:
            obj_for_this_frame = set(obj_arr[np.argwhere(fr_arr == frame)].flatten())
            if frame not in running_frame_obj_set_dict:
                running_frame_obj_set_dict[frame] = set(obj_for_this_frame)
            else:
                running_frame_obj_set_dict[frame] = running_frame_obj_set_dict[frame].intersection(set(obj_for_this_frame))

    return running_frame_obj_set_dict, frame_obj_double_dict


def get_number_of_succeed_each_case(dcm_list, iframe, iobj, running_frame_obj_set_dict=None, frame_obj_double_dict=None):
    """Get matrix of number of algortihms that succeed for each frame, object

    Matrix is defined over number of points and distance of injection
    """
    # Get the combinations
    if (running_frame_obj_set_dict is None) or (frame_obj_double_dict is None):
        running_frame_obj_set_dict, frame_obj_double_dict = find_common_combos(dcm_list)

    # Check if requested input is valid
    if iframe not in running_frame_obj_set_dict:
        print('Cannot find frame in dictionary...')
        return
    elif iobj not in running_frame_obj_set_dict[iframe]:
        print('Cannot find object for this frame in dictionary...')
        return

    # Populate matrix
    matrix_all = np.zeros(dcm_list[0]['FP_near_inject'][0].shape)
    for idcm, dcm in enumerate(dcm_list):
        idx_run = frame_obj_double_dict[idcm][iframe][iobj]
        matrix_this = dcm['FP_near_inject'][idx_run]

        # Update aggregate matrix -- binary, so we just add
        if matrix_this.shape[0] <= matrix_all.shape[0]:
            nless = matrix_all.shape[0] - matrix_this.shape[0]
            matrix_all[nless:,:] += matrix_this

    return matrix_all


def find_common_attack(dcm_list, npts, dist_behind, nalgs_needed, verbose=False):
    if verbose:
        print('Number of algorithms present: %i' % len(dcm_list))
        print('Number of algorithms requested: %i' % nalgs_needed)

    # Get mesh grids
    xm = dcm_list[0]['attack_pts'][0]
    ym = dcm_list[0]['attack_dist'][0] - dcm_list[0]['dist'][0]

    # Get the index into where this is
    idx_lin, idx_tup = analysis.get_closest_idx_mesh(xm, ym, npts, dist_behind)

    # Get frame-obj intersection
    fo_set_dict, frame_dict = find_common_combos(dcm_list)

    # Now loop over common frames
    found_f_o_list = []
    for frame in fo_set_dict:
        for obj in fo_set_dict[frame]:
            nfound = 0
            for idcm, dcm in enumerate(dcm_list):
                idx_into_dcm = frame_dict[idcm][frame][obj]
                if dcm[fp_str(True)][idx_into_dcm][idx_tup]:
                    nfound += 1
            if nfound >= nalgs_needed:
                dist_obj = dcm_list[idcm]['dist'][idx_into_dcm]
                found_f_o_list.append((frame, obj, dist_obj))

    return sorted(found_f_o_list, key=lambda x: x[0])
