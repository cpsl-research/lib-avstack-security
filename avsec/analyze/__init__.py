# @Author: Spencer Hallyburton <spencer>
# @Date:   2021-05-08
# @Filename: analysis.py
# @Last modified by:   spencer
# @Last modified time: 2021-09-09

import os
import sys
from copy import copy, deepcopy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import pickle

# Add paths
ROOT_DIR = os.path.dirname(os.getcwd())
BASE_DIR = os.path.dirname(ROOT_DIR)

# Import AV utils
from avapi import evaluation, visualize

# ========================================================
# ANALYSIS CRUNCHING
# ========================================================

def load_from_batch(path_folder, ibatch):
    batch = pickle.load(open(os.path.join(path_folder, 'batch_%d.p'%ibatch), 'rb'))
    return batch


def run_frustum_spoof_analysis(DM, result_path, data_dict_in,
                                  baseline_data=None):
    """
    Run the standard analysis as seen in our first paper.

    INPUTS:
    DM          --> data manager
    result_path --> path to the directory with results
    data_dict   --> dictionary of information on the run
    """
    data_dict = deepcopy(data_dict_in)

    # Analyze data
    results, conf_mat = evaluation.get_detection_results_from_folder(DM, result_path, metric='3D_IoU', whitelist_types=['Car'])
    idx_result_start = 0

    # Convert results to a list
    key_arr = np.asarray(list(results.keys()))
    result_arr = np.asarray(list(results.values()))

    # Loop over instances from the data
    for i_run in range(len(data_dict['frame'])):
        target_xyz = data_dict['target_xyz'][i_run]
        # Determine if it is frustum sample or frustum mesh
        try:
            iterator = iter(data_dict['attack_pts'][i_run])
        except TypeError:
            # -- frustum sample
            result = result_arr[i_run]
            attack_xyz = data_dict['attack_xyz'][i_run]
            nassign, nfp, nfn, fp_near_inject, fn_of_target = \
                _single_frustum_attack_analysis(result, attack_xyz, target_xyz)
        else:
            # -- frustum mesh
            mesh_shape = data_dict['attack_pts'][i_run].shape
            n_this_run = np.size(data_dict['attack_pts'][i_run])
            nassign, nfp, nfn, fp_near_inject, fn_of_target = ([] for i in range(5))
            for j_instance in range(n_this_run):
                attack_xyz = data_dict['attack_xyz'][i_run][np.unravel_index(j_instance, mesh_shape)]
                result_this = result_arr[idx_result_start + j_instance]
                nassign_v, nfp_v, nfn_v, fp_near_inject_v, fn_of_target_v = _single_frustum_attack_analysis(result_this, attack_xyz, target_xyz)
                nassign.append(nassign_v)
                nfp.append(nfp_v)
                nfn.append(nfn_v)
                fp_near_inject.append(fp_near_inject_v)
                fn_of_target.append(fn_of_target_v)
            # Update index for next run
            idx_result_start = idx_result_start + n_this_run
            # Reshape
            nassign, nfp, nfn, fp_near_inject, fn_of_target = \
                [np.reshape(l, mesh_shape) for l in [nassign, nfn, nfn, fp_near_inject, fn_of_target]]

        # Append data elements
        data_dict['nAssigned'].append(nassign)
        data_dict['nFP'].append(nfp)
        data_dict['nFN'].append(nfn)
        data_dict['FP_near_inject'].append(fp_near_inject)
        data_dict['FN_of_target'].append(fn_of_target)

        if baseline_data is not None:
            raise NotImplementedError
            # # Get values
            # nFP_baseline = baseline_data[data_dict['frame'][i_run]].get_number_of('false_positives')
            # nFN_baseline = baseline_data[data_dict['frame'][i_run]].get_number_of('false_negatives')
            # # Lowest is 0
            # FPs = np.maximum(0, np.reshape(fp_list, mesh_shape) - nFP_baseline)
            # FNs = np.maximum(0, np.reshape(fn_list, mesh_shape) - nFN_baseline)
            #
            # # Add
            # data_dict['nFP_over_baseline'].append(FPs)
            # data_dict['nFN_over_baseline'].append(FNs)
        else:
            data_dict['nFP_over_baseline'].append([])
            data_dict['nFN_over_baseline'].append([])

    return data_dict, results, conf_mat


def _single_frustum_attack_analysis(result, attack_xyz, target_xyz, thresh_for_fp=5, thresh_for_fn=0.1):
    nassign = result.get_number_of('assigned')
    fp_this = result.get_objects_of('false_positives')
    fn_this = result.get_objects_of('false_negatives')

    # Get boolean if FP exists where we inject points
    thresh_for_fp = 5
    fp_near_inject = np.any([np.linalg.norm(fp.box3d.t - attack_xyz) < thresh_for_fp for fp in fp_this])
    # Get boolean if FN exists on target vehicle
    thresh_for_fn = 0.1
    fn_of_target = np.any([np.linalg.norm(fn.box3d.t - target_xyz) < thresh_for_fn for fn in fn_this])
    return nassign, len(fp_this), len(fn_this), fp_near_inject, fn_of_target


def run_naive_spoof_analysis(DM, result_path, data_dict_in, baseline_data=None):
    """
    Runs analysis on naive spoofing
    """
    data_dict = deepcopy(data_dict_in)

    # Analyze data in usual way
    results, conf_mat = evaluation.get_detection_results_from_folder(DM, result_path, metric='3D_IoU', whitelist_types=['Car'])
    idx_result_start = 0

    # Convert results to list
    key_arr = np.asarray(list(results.keys()))
    result_arr = np.asarray(list(results.values()))

    # Loop over instances from data
    for i_run in range(len(data_dict['frame'])):
        assign_list = []
        fp_list = []
        fn_list = []
        fp_near_inject_list = []

        attack_xyz = data_dict['attack_xyz'][i_run]

        # Results
        result_this = result_arr[i_run]
        fp_this = result_this.get_objects_of('false_positives')

        # Get boolean if FP exists where we injected
        thresh_for_fp = 5
        fp_near_inject_this = np.any([np.linalg.norm(fp.box3d.t - attack_xyz) < thresh_for_fp for fp in fp_this])

        # Reshape grid
        data_dict['nAssigned'].append(result_this.get_number_of('assigned'))
        data_dict['nFP'].append(result_this.get_number_of('false_positives'))
        data_dict['nFN'].append(result_this.get_number_of('false_negatives'))
        data_dict['FP_near_inject'].append(fp_near_inject_this)

        # Compare to baseline data
        if baseline_data is not None:
            # Get values
            nFP_baseline = baseline_data[data_dict['frame'][i_run]].get_number_of('false_positives')
            nFN_baseline = baseline_data[data_dict['frame'][i_run]].get_number_of('false_negatives')
            # Lowest is 0
            FPs = np.maximum(0, data_dict['nFP'][i_run])
            FNs = np.maximum(0, data_dict['nFN'][i_run])
            # Add
            data_dict['nFP_over_baseline'].append(FPs)
            data_dict['nFN_over_baseline'].append(FNs)

    return data_dict, results, conf_mat


# ========================================================
# VISUALIZATIONS
# ========================================================

def heatmap_attack_presence(data_dict, idx):

    fig, axs = plt.subplots(1,2, sharey=True, figsize=(12,4))
    xm = data_dict['attack_pts'][idx]
    ym = data_dict['attack_dist'][idx] - data_dict['target_dist'][idx]

    # For false positive existence
    heat_wrapper(axs[0], xm, ym, data_dict, idx, 'FP_near_inject', vmin=0, vmax=1)

    # for false negative existence
    if np.all(data_dict['FN_of_target'][idx].flatten()):
        print('WARNING: BASE CASE LIKELY HAS FN')
    heat_wrapper(axs[1], xm, ym, data_dict, idx, 'FN_of_target', vmin=0, vmax=1)
    plt.show()


def heatmap_fp_fn_occurrences(data_dict, idx, over_baseline=False, fix_color_max=None):

    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(14,4))
    xm = data_dict['attack_pts'][idx]
    ym = data_dict['attack_dist'][idx] - data_dict['target_dist'][idx]

    if over_baseline:
        fp_str = 'nFP_over_baseline'
        fn_str = 'nFN_over_baseline'
    else:
        fp_str = 'nFP'
        fn_str = 'nFN'

    # Make max colorvalue the max possible
    vmin = 0
    false_max = np.maximum(data_dict[fn_str][idx], data_dict[fp_str][idx])
    if fix_color_max is None:
        vmax = np.max(data_dict['nAssigned'][idx] + false_max)
    else:
        vmax = fix_color_max

    # Heatmaps of the assigned
    heat_wrapper(axs[0], xm, ym, data_dict, idx, 'nAssigned', vmin, vmax)

    # Heatmap of the false positives
    heat_wrapper(axs[1], xm, ym, data_dict, idx, fp_str, vmin, vmax)

    # Heatmap of the false negatives
    heat_wrapper(axs[2], xm, ym, data_dict, idx, fn_str, vmin, vmax)
    # plt.tight_layout()
    plt.show()


def heat_wrapper(ax, xm, ym, data_dict, idx, index_string, vmin, vmax):
    if max(xm.flatten()) > 1:
        xstr = 'number of points'
    else:
        xstr = 'downsample_pct'
    ystr = 'range of injection (m)'
    d = {xstr:xm.flatten(), ystr:ym.flatten(), index_string:data_dict[index_string][idx].flatten()}
    df = pd.DataFrame(data=d)
    table = df.pivot(ystr, xstr, index_string)
    heatmap_data(xm, ym, table,
                 title=index_string,
                 vmin=vmin, vmax=vmax, ax=ax)


def heatmap_data(xdata, ydata, table, title,
                 vmin=0, vmax=1, nxlab=10, nylab=10, ax=None):
    if vmin < 0:
        cmap = 'seismic'
    else:
        # cmap = 'PuBuGn'
        cmap = 'inferno'

    xd = np.unique(xdata)
    yd = np.unique(ydata)
    xticklabels = np.round(xd, 2)
    yticklabels = np.round(yd)
    ax = sns.heatmap(table, xticklabels=xticklabels, yticklabels=yticklabels,\
                     cmap=cmap, vmin=vmin, vmax=vmax, ax=ax)

    # Set lower bounds on number of labels
    xfreq = int(np.ceil(len(xd) / nxlab))
    yfreq = int(np.ceil(len(yd) / nylab))

    for ind, label in enumerate(ax.get_yticklabels()):
        if ind % yfreq == 0:  # every 10th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    for ind, label in enumerate(ax.get_xticklabels()):
        if ind % xfreq == 0:  # every 2nd label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    ax.invert_yaxis()
    ax.set_title(title)
    # plt.show()


# ========================================================
# UTILITIES
# ========================================================

def get_closest_idx_mesh(xm, ym, xq, yq, verbose=False):
    x_flat = xm.flatten()
    y_flat = ym.flatten()
    x_min = x_flat[np.argmin(np.abs(x_flat - xq))]
    y_min = y_flat[np.argmin(np.abs(y_flat - yq))]
    idx_linear = np.argwhere((x_flat == x_min) & (y_flat == y_min))[0][0]

    if verbose:
        print('Closest x in mesh is: %.2f' % x_min)
        print('Closest y in mesh is: %.2f' % y_min)

    return idx_linear, np.unravel_index(idx_linear, xm.shape)
