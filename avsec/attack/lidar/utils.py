# @Author: Spencer Hallyburton <spencer>
# @Date:   2021-10-24
# @Filename: attack.py
# @Last modified by:   Spencer
# @Last modified time: 2021-12-09T22:17:31-05:00

import abc
import os, shutil, sys
import numpy as np
from tqdm import tqdm

import random

from multiprocessing import Pool, cpu_count
from copy import copy, deepcopy
from functools import partial

from copy import copy, deepcopy
from percep_attacks import analysis

import percep_attacks
from percep_attacks.lidar import defend
from avutils import algorithms
from avutils.perception import types
from avutils.perception import maskfilters

import pickle
import time

from avutils.perception import bbox

"""
TODO:
- Add numba to accelerate
- Make multi-process
"""


# ==============================================================================
# POINT CLOUD MANIPULATION
# ==============================================================================

def sample_pc(pc, keep, strategy='random', upsample=False):
    """
    Take points from the full point cloud

    INPUTS:
    pc       --> lidar point cloud
    keep     --> either a fraction on [0,1) or an integeter on [1,inf)
    strategy --> options are: random,
    upsample --> whether upsampling is allowed or if error thrown when too many
    """
    import warnings
    pc2 = np.copy(pc)

    # If there are no points, get out of here
    if pc2.shape[0] == 0:
        return pc2

    if keep > 1:
        # to allow to pass in integer for number of points to keep
        pts_choose = int(keep)
    else:
        # to allow to pass in a fraction
        pts_choose = int(np.round(pc2.shape[0]*keep))

    # Ensure at least 1 point (?)
    pts_choose = max(1, pts_choose)

    # check if too many points
    need_upsample = False
    if (pts_choose > pc2.shape[0]) and not upsample:
        warnings.warn('Number of desired points (%i) is more than the points that exist (%i)...going with max points' %
                      (pts_choose, pc2.shape[0]))
        pts_choose = pc2.shape[0]
    elif (pts_choose > pc2.shape[0]):
        need_upsample = True

    # perform downsampling
    if strategy=='random':
        # Upsample points if needed
        # randomly sample points from the entire space
        if need_upsample:
            # Create additional points with jitter
            nNew = pts_choose - pc2.shape[0]
            idx_new_pts = np.random.choice(pc2.shape[0], size=nNew)
            new_pts = pc2[idx_new_pts,:]
            jf = 0.1  # meters
            jitter = np.array([[jf, 0, 0], [0, jf, 0], [0, 0, jf]])
            new_pts[:,0:3] += np.random.randn(nNew, 3) @ jitter
            # Force bounds
            maxarr = np.max(pc2[:,0:3], axis=0)
            minarr = np.min(pc2[:,0:3], axis=0)
            new_pts[:,0:3] = np.minimum(new_pts[:,0:3], np.tile(maxarr, (nNew,1)))
            new_pts[:,0:3] = np.maximum(new_pts[:,0:3], np.tile(minarr, (nNew,1)))
            pc2 = np.vstack([pc2, new_pts])
        else:
            # Sample from there
            idx_pcs = np.random.choice(pc2.shape[0], size=pts_choose, replace=False)
            pc2 = pc2[idx_pcs, :]
    elif strategy=='exterior':
        # sample points mostly from the exterior of the object
        raise NotImplementedError

    return pc2


def range_out_points(lidar, label, calib, r_new,
                     gp_consistent=False, rescale=False, scale_range=False):
    """
    Slide points along the range dimension, optionally update label

    INPUTS:---------------------------------
    lidar         --> nx3 or nx4 point cloud
    label         --> original label, will output a new one at range
    calib         --> calibration class
    gp_consistent --> whether the output will be made consistent with ground
    rescale       --> whether to rescale the outputs to be the same spread as original
    """

    # Copy the data
    new_lidar = copy(lidar)
    new_label = deepcopy(label)

    # Get 3D box center
    box_center_rect = np.asarray(new_label.box3d.t)
    box_center_velo = calib.project_rect_to_velo(np.expand_dims(box_center_rect,0))[0,:]
    bcv = box_center_velo

    # Manipulate range
    r_0 = np.linalg.norm(box_center_velo)
    bcv2 = copy(box_center_velo)

    # Handle if needs to be ground-plane consistent
    if gp_consistent:
        # Apply shift in velo coordinates to maintain original Z (height) coordinate
        # solving this: rf^2 = z0^2 + (s*y0)^2 + (s*x0)^2
        s = np.sqrt( (r_new**2 - bcv[2]**2) / (bcv[0]**2 + bcv[1]**2) )
        bcv2[0:2] *= s
    else:
        # Apply shift along unit vector and allow to change Z coordinate
        s = r_new / r_0
        bcv2[0:3] *= s

    # Apply change in center coordinates
    delta_box_center = bcv2 - bcv
    new_lidar[:,0:3] += delta_box_center

    # Apply shift to label (3D box)
    box_center_velo += delta_box_center
    box_center_rect = calib.project_velo_to_rect(np.expand_dims(box_center_velo,0))[0,:]
    new_label.box3d.t = box_center_rect

    # If rescaling, scale points toward centerline to make it fit in same frustum
    if rescale:
        rr = r_new / r_0
        # Rotate into a boresight coordinate frame and scale cross-range
        center_vec = box_center_velo
        R = tforms.align_x_to_vec(center_vec)

        # New points: Rinv @ S @ R @ PC (associative property holds)
        # Where S is [[rr, 0, 0], [0, rr, 0], [0, 0, rr]]
        if scale_range:
            S = np.array([[rr, 0, 0], [0, rr, 0], [0, 0, rr]])
        else:
            S = np.array([[1, 0, 0], [0, rr, 0], [0, 0, rr]])

        # Apply scaling
        M = np.transpose(R) @ S @ R
        new_lidar[:,0:3] = (new_lidar[:,0:3]-box_center_velo) @ np.transpose(M) + \
                            box_center_velo

        # Scale the bounding box corners
        # NOTE: this is broken right now...
        box_corners = new_label.box3d.get_corners()
        box_corners_velo = calib.project_rect_to_velo(box_corners)
        corners_scaled_velo = (box_corners_velo - box_center_velo) @ np.transpose(M) +\
                                    box_center_velo
        corners_scaled_rect = calib.project_velo_to_rect(corners_scaled_velo)
        l,w,h = bbox.compute_box_size(corners_scaled_rect, new_label.box3d.ry)
        new_label.box3d.l = l
        new_label.box3d.w = w
        new_label.box3d.l = h

    # Apply shift to label (2D box)
    new_label.box2d = bbox.proj_3d_bbox_to_2d_bbox(new_label.box3d, calib.P)

    return new_lidar, new_label


def azimuth_rotate_points(lidar, delta_az, label=None, calib=None):
    """
    Take point cloud and rotate by some azimuth angle

    Define the "delta_az" as the angle that the point cloud needs to rotate

    NOTE: lidar point cloud is defined as (right-handed)
    x-forward
    y-left
    z-up
    so an "azimuth" rotation is about z using the right-hand-rule for direction

    :lidar - nx3/4 point cloud matrix
    :delta_az - the desired azimuth angle to rotate the point cloud
    :label - label associated with that lidar point cloud
    :calib - calibration data
    """
    import avutils.tforms as tforms

    # Build rotation matrix
    Rz = tforms.rotz(delta_az)

    # ------------------------
    # Update points
    # ------------------------
    new_lidar = copy(lidar)
    new_lidar[:,0:3] = lidar[:,0:3] @ Rz.T

    # ------------------------
    # Update label
    # ------------------------
    if label is not None:
        new_label = deepcopy(label)
        assert(calib is not None)

        # --- 3D box center
        t = calib.project_rect_to_velo(label.box3d.get_center())
        t = calib.project_velo_to_rect(Rz @ t)

        # --- 3D bounding box
        box_corners = calib.project_rect_to_velo(label.box3d.get_corners())
        box_corners = calib.project_velo_to_rect(box_corners @ Rz.T)
        l, w, h = bbox.compute_box_size(box_corners)
        ry = bbox.get_heading() - delta_az
        new_box3d = bbox.Box3D(np.array((h, w, l, t[0], t[1], t[2], ry)))

        # --- 2D bounding box
        new_box2d = new_box3d.project_to_2d_bbox(calib)

        # --- make new label
        new_label.box3d = deepcopy(new_box3d)
        new_label.box2d = deepcopy(new_box2d)

        return new_lidar, new_label
    else:
        return new_lidar


class LidarAttack():
    """Framework for attacking perception"""

    @abc.abstractmethod
    def attack(self):
        """How to perform the attack"""


class NaiveSpoofing(LidarAttack):
    """Executing naive spoofing"""
    def __init__(self, trace_directory):
        self.trace_directory = trace_directory

    def attack(self, point_cloud, dist, npts, itrace):
        """Append spoof points to point cloud"""
        spoof_points, meta_data = self.get_spoof_points(point_cloud, dist, npts, itrace)
        return np.vstack((point_cloud, spoof_points)), meta_data

    def get_spoof_points(self, dist, npts, itrace):
        """Getting naive spoof points"""
        meta_data = {}
        meta_data['dist'] = dist
        meta_data['npts'] = npts
        occ_obj_data = self.load_attack_trace(npts, itrace)
        obj_lidar, obj_bbox = move_to_front(occ_obj_data['lidar'], occ_obj_data['bbox'], dist)

    def load_attack_trace(self, npts, itrace):
        return np.load(f'{self.trace_directory}/{npts}/{itrace}.npz')


# ---------------------------------------
# HELPERS FOR LARGE-SCALE DATA COLLECTION
# ---------------------------------------

def make_default_file_names(alg_list, attack_type, inject_type, inject_shape, defense):
    fnames = {}
    for alg in alg_list:
        fnames[alg] = f'{attack_type}-{alg}-full-{inject_type}'
        if inject_type == 'random-box':
            fnames[alg] += f'-{inject_shape}'
        if defense is not None:
            fnames[alg] += '-{}'.format(defense+'_defense')
        fnames[alg] += '.p'
    return fnames


def make_data_struct():
    """Creates a data structure useful for large-scale captures and analysis"""
    return {'frame':[], 'obj':[], 'occlusion_num':[],
            'target_dist':[], 'target_xyz':[],
            'npts_bbox':[], 'npts_frust':[],
            'attack_xyz':[], 'attack_trace':[],
            'attack_pts':[], 'attack_dist':[],
            'nAssigned':[], 'nFP':[], 'nFN':[],
            'nFP_over_baseline':[], 'nFN_over_baseline':[],
            'FP_near_inject':[], 'FN_of_target':[]}


def run_full_spoofing_attack(DM, attack_type, inject_type, inject_shape,
        exp_folder, exp_split, alg_names,
        alg_file_names_map=None, out_folder='output_data', use_checkpoint=True,
        results_baseline=None, defense=None, seed=None):

    # Make output folder
    if out_folder is not None:
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

    # Make file names if not passed
    if (alg_file_names_map is None) or (len(alg_file_names_map)<len(alg_names)):
        alg_file_names_map = make_default_file_names(alg_names, attack_type, inject_type,
                                            inject_shape, defense)

    # Preliminaries
    attack_params = get_attack_params(DM, attack_type)
    DEF_POSTPROC, svf, svf_method = get_defenses(defense)
    istarts, DCs_master = handle_checkpoint(out_folder, alg_file_names_map,
                                            attack_type, attack_params, use_checkpoint)

    # Figure out lowest starting index
    istart_min = min(istarts.values())

    # Run the experiment loop
    first = True
    nExcepts_data = 0
    nExcepts_alg = 0
    exp_path = os.path.join(exp_folder, exp_split)

    for i_iter in range(istart_min, attack_params['params_loop'].shape[1], 1):
        print('ITERATION:', i_iter)
        # Check if need to skip
        if (attack_type == 'frustum') and \
            (len(DM.get_labels(attack_params['params_loop'][:,i_iter][0], whitelist_types=['Car']))==0):
            print('No suitable objects')
            continue

        # Call the perception algorithms -- only ones that haven't been called
        algs_to_call = [alg for alg in alg_names if istarts[alg] <= i_iter]
        if svf:
            assert algs_to_call == ['pillars'], 'Algorithm must only be pillars for svf'

        # Make data
        DC_ADD_blank = make_data(DM, exp_split, i_iter, attack_params,
            attack_type, inject_type, inject_shape)

        # Loop over algorithms to call
        for alg in algs_to_call:
            print(f'RUNNING ALGORITHM: {alg}')
            DC_ADD = {alg:deepcopy(DC_ADD_blank)}

            # --- call algorithm
            result_dir = call_percep_on_data(DM, exp_folder, exp_split, [alg],
                                             svf=svf, svf_method=svf_method)

            # --- run defense
            result_dir = call_postproc_defenses(DM, DEF_POSTPROC, exp_split, result_dir)

            # --- run analysis
            DC_ADD = analysis_on_percep_results(DM, attack_type, exp_split, result_dir,
                                                {alg:DC_ADD[alg]}, results_baseline, [alg])

            # --- aggregate_results and save
            DCs_master = aggregate_results(DCs_master, {alg:DC_ADD[alg]}, alg_file_names_map, out_folder, save=True)


def make_data(DM, exp_split, i_iter, attack_params,
            attack_type, inject_type, inject_shape):
    if attack_type == 'naive':
        pts, dist, trace, nframes = attack_params['params_loop'][:,i_iter]
        print('\n\nPARAMS: %i pts, %i m, %i trace' % (pts, dist, trace))
        DC_ADD = make_naive_spoofing_data(
            DM=DM,
            exp_split=exp_split,
            npts=pts,
            itrace=trace,
            distance=dist,
            nframes=nframes)
    elif attack_type == 'frustum_mesh':
        iframe = attack_params['params_loop'][:,i_iter][0]
        npts, dist_rel = attack_params['npts'], attack_params['dist_rel']
        idx_attack_indicators = []
        labs = DM.get_labels(iframe, whitelist_types=['Car'])
        for iobj in range(len(labs)):
            # TODO: eliminate if not good baseline result
            idx_attack_indicators.append((iframe, iobj))
        DC_ADD = make_frustum_mesh_spoofing_data(
            DM=DM,
            exp_split=exp_split,
            idx_attack_indicators=idx_attack_indicators,
            npts=npts,
            dist_rel=dist_rel,
            inject_type=inject_type,
            inject_shape=inject_shape)
    elif attack_type == 'frustum_sample':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return DC_ADD

# ========================================================
# EXTRAS -- common
# ========================================================

def get_attack_params(DM, spoof_type, **kwargs):
    if spoof_type == 'naive':
        npts = np.arange(10, 210, 10)
        distance = np.array([8, 15, 30])
        traces = np.arange(0,5,1)[:kwargs.get('ntraces', 5)]
        nframes = 200
        g = np.meshgrid(npts, distance, traces, nframes)
        params_loop = np.vstack(map(np.ravel, g))
        spoof_params = {'npts':npts, 'distance':distance, 'traces':traces, 'nframes':nframes,
                        'params_loop':params_loop}
    elif spoof_type == 'frustum':
        npts = np.concatenate(([2], np.arange(5, 100, 5),
                                 np.arange(100, 210, 10)), axis=0)
        distance_rel = np.arange(-10, 30, 2)
        # distance_rel = np.arange(-10, 30, 10)
        idx_frames = DM.idxs
        params_loop = idx_frames[:,None].T
        spoof_params = {'npts':npts, 'dist_rel':distance_rel, 'frames':idx_frames, 'params_loop':params_loop}
    else:
        raise NotImplementedError

    return spoof_params


def aggregate_results(DCs_master, DCs_add, alg_file_names_map, save_folder, save=True):
    for alg, D_add in DCs_add.items():
        for key in DCs_master[alg]:
            DCs_master[alg][key].extend(D_add[key])
        fsave = os.path.join(save_folder, alg_file_names_map[alg])
        pickle.dump(DCs_master[alg], open(fsave, 'wb'))
    print('SAVED OUTCOMES\n\n')
    return DCs_master


def analysis_on_percep_results(DM, attack_type, exp_split, result_dirs, DC_ADD, results_baseline, algs):
    DM_TEMP = DM.make_new(DM.data_dir, exp_split)
    for i, (alg, res_dir) in enumerate(zip(algs, result_dirs)):
        res_base = results_baseline[alg] if results_baseline is not None else None
        if attack_type == 'naive':
            DC_ADD[alg], _ = analysis.run_naive_spoof_analysis(
                DM_TEMP, res_dir, DC_ADD[alg], res_base)
        else:
            DC_ADD[alg], _ = analysis.run_frustum_spoof_analysis(
                DM_TEMP, res_dir, DC_ADD[alg], res_base)
    return DC_ADD


def call_postproc_defenses(DM, DEF_POSTPROC, exp_split, result_dirs):
    """Run defenses on the output of the percep models"""
    if DEF_POSTPROC is not None:
        DM_TEMP = DM.make_new(DM.data_dir, exp_split)
        for i, res_dir in enumerate(result_dirs):
            DEF_POSTPROC.set_data_manager(DM_TEMP)
            _, _, result_dirs[i] = DEF_POSTPROC.evaluate_from_folder(res_dir, write=True)
    return result_dirs


def call_percep_on_data(DM, exp_folder, exp_split, alg_names, **kwargs):
    """Loop over all algorithms and run on percep data"""
    result_dirs = []
    if isinstance(alg_names, str):
        alg_names = [alg_names]
    exp_path = os.path.join(exp_folder, exp_split)
    for alg in alg_names:
        P_ALG = algorithms.get_algorithm_class(alg)
        P_ALG.inference_on_folder_run_command(exp_path, **kwargs)
        result_dirs.append(P_ALG.get_save_dir(exp_split))
    return result_dirs


def handle_checkpoint(out_folder, alg_file_names_map, attack_type, attack_params, use_checkpoint):
    """Load in data from a saved checkpoint"""
    if use_checkpoint:
        istarts = {}
        DCs_master = {}
        for alg, fname in alg_file_names_map.items():
            fpath = os.path.join(out_folder, fname)
            if os.path.exists(fpath):
                print(f'Loading {fpath} checkpoint results')
                this_dc_master = pickle.load(open(fpath, "rb"))
                if attack_type == 'naive':
                    pts_last_done = this_dc_master['attack_pts'][-1]
                    dist_last_done = this_dc_master['attack_dist'][-1]
                    trace_last_done = this_dc_master['attack_trace'][-1]
                    row = np.array([pts_last_done, dist_last_done, trace_last_done])
                    aeq = np.all(attack_params['params_loop'][:3, :].T == row, axis=1)
                elif attack_type == 'frustum':
                    frame_last_done = np.asarray(this_dc_master['frame'][-1])
                    aeq = np.all(attack_params['params_loop'].T == frame_last_done, axis=1)
                else:
                    raise NotImplementedError
                idx_last = np.where(aeq)[0][0]
                istarts[alg] = idx_last + 1
                DCs_master[alg] = this_dc_master
            else:
                print(f'Could not find checkpoint results {fpath} -- making new')
                istarts[alg] = 0
                DCs_master[alg] = make_data_struct()
    else:
        # Get a new data structure
        istarts = {alg:0 for alg in alg_file_names_map}
        DCs_master = {alg:make_data_struct() for alg in alg_file_names_map}
    print('\nAlgorithms starting at following checkpoint indices:')
    for k, v in istarts.items():
        print('--', k, ' : ', v)
    return istarts, DCs_master


def get_folder(file_name):
    out_folder = os.path.join('output_data', file_name.replace('.p', ''))
    if not os.path.exists(out_folder):
        try:
            os.makedirs(out_folder)
        except FileExistsError as e:
            pass  # for multiprocess cases
    fname_res = os.path.join(out_folder, file_name)
    return out_folder, fname_res


def get_defenses(defense):
    # Make defenses
    svf = False
    svf_method = None
    DEF_POSTPROC = None
    if defense in ['lpd', 'gbsc']:
        DEF_POSTPROC = defend.factory_defense(defense, DM=[])
    elif defense in ['svf']:
        svf=True
        svf_method='force'
    return DEF_POSTPROC, svf, svf_method

# ========================================================
# EXTRAS -- naive attack
# ========================================================

def move_to_front(lidar, bbox, d):
    """
    Moves lidar and box data to a certain distance (range)
    """
    lidar_copy = deepcopy(lidar)
    bbox_copy = deepcopy(bbox)

    # Center object in the frame
    box_center_y = np.mean(bbox_copy[:,1])
    lidar_copy[:,1] -= box_center_y
    bbox_copy[:,1] -= box_center_y

    # Get unit vector
    center_loc = np.mean(lidar_copy[:,0:3], axis=0)
    center_dist = np.linalg.norm(center_loc)
    unit_vec = center_loc / center_dist

    # Get what to add by delta in unit vector
    delta_dist = d - center_dist
    delta_vec = delta_dist * unit_vec

    # Add to lidar and bbox
    lidar_copy[:,:3] += delta_vec
    bbox_copy += delta_vec

    return lidar_copy, bbox_copy


# ========================================================
# EXTRAS
# ========================================================

def _handle_frustum_attack_inputs(DM, file_name, exp_name, algorithm, defense,
                          use_checkpoint, out_folder, frustum_only=False):
    # =======================================
    # Validation of inputs
    # =======================================
    out_folder_path = os.path.join(out_folder, file_name.replace('.p', ''))
    if not os.path.exists(out_folder_path):
        try:
            os.makedirs(out_folder_path)
        except FileExistsError as e:
            pass  # for multiprocess cases
    fname_res = os.path.join(out_folder_path, file_name)

    # HANDLE DEFENSE
    if defense is not None:
        defense = defense.lower()
        assert(defense in ['svf', 'lpd', 'gbsc'])

    # Make defenses
    svf = False
    svf_method = None
    if defense in ['lpd', 'gbsc']:
        DEF_POSTPROC = defend.factory_defense(defense, DM=[])
    elif defense in ['svf']:
        DEF_POSTPROC = None
        assert(algorithm.lower() == 'pillars')
        svf=True
        svf_method='force'


    # HANDLE CHECKPOINT
    if use_checkpoint and (os.path.exists(fname_res)):
        # Load the old data structure
        data_collect_master = pickle.load(open(fname_res, "rb"))

        if 'defense' in fname_res:
            last_pts_finished = data_collect_master['attack_pts'][-1]
            last_dist_finished = data_collect_master['attack_dist'][-1] - data_collect_master['target_dist'][-1]
            idx_start = (last_pts_finished, last_dist_finished)
        else:
            max_frame_done = max(data_collect_master['frame'])
            idx_start = np.argwhere(DM.idxs == max_frame_done)[0][0]
    else:
        # Get a new data structure
        data_collect_master = make_data_struct()
        if 'defense' in fname_res:
            idx_start = (-1, -1)
        else:
            idx_start = 0

    # HANDLE ALGORITHM
    assert(algorithm.lower() in ['fpn', 'fcn', 'avod', 'epnet', 'pillars', 'pointrcnn', 'pixor'])

    if frustum_only:
        assert(algorithm.lower() in ['fpn', 'fcn'])

    # SET UP FILE PATHS
    exp_split = '%s-experiment-%s' % (algorithm.lower(), exp_name)

    return fname_res, out_folder_path, DEF_POSTPROC, exp_split, data_collect_master, idx_start, svf, svf_method


def write_idxs(DM, dataset_path, idx_list):
    # Write indices to the file
    if 'kitti' in DM.__class__.__name__.lower():
        _, file_tail = os.path.split(dataset_path)
        path_to_set_info = os.path.join(DM.data_dir, 'ImageSets', file_tail) + '.txt'
        with open(path_to_set_info, 'w') as fw:
            for idx in idx_list:
                fw.write('%06d\n' % idx)


# ========================================================
# NAIVE SPOOFING
# ========================================================

def make_naive_spoofing_data(DM,
                            exp_split,
                            npts,
                            itrace,
                            distance,
                            nframes=400,
                            frame_list=None,
                            seed=None,
                            verbose=False,
                            occluded_obj_dir='/data/spencer/KITTI/Occluded_Objects'):
    """
    Make one batch of naive spoofing data

    A batch of data is defined by:
    - npts
    - ntraces
    - distance
    - frame_list or nframes
    """
    # Initialize dataframe
    data_collect = make_data_struct()
    random.seed(seed)
    dataset_path = os.path.join(DM.data_dir, exp_split)
    print(dataset_path, flush=True)
    # Wipe the existing data in the experiment folder
    if 'kitti' in DM.name.lower():
        DM.wipe_experiment(dataset_path)
    else:
        raise NotImplementedError

    # Check inputs
    acceptable_points = list(range(10, 210, 10))
    assert npts in acceptable_points
    if itrace > 5:
        raise RuntimeError('Currently cannot test more than 5 traces')

    # Get number of frames
    assert(not ((nframes is None) and (frame_list is None)))
    if nframes is not None:
        frame_list = DM.idxs[:nframes]
    elif frame_list is not None:
        pass
    else:
        frame_list = DM.idxs

    # Make data with loops
    icount = 0

    # Load the occluded object -- need this weird hack for multiprocess pool??
    occ_obj_data = np.load(f'{occluded_obj_dir}/{npts}/{itrace}.npz')
    occ_obj_data = {key:occ_obj_data[key] for key in occ_obj_data}

    print(f'Occluded obj. points: {npts}')
    print(f'd = {distance}')
    if verbose:
        print(f'Creating configuration:\n---Trace: {itrace}\n---Pts: {npts}\n---Distance: {distance}')

    # Copy data to new folder
    src_base = DM.split_path
    dest_base = dataset_path
    print('Copying experiment data')
    DM.copy_experiment_from_to(src_base, dest_base, frame_list=frame_list)

    # Use multi-process to make the data
    print('Making naive spoof data with multiprocess...')
    part_func = partial(_make_single_naive_data, DM, dataset_path, occ_obj_data, itrace, npts, distance)
    pool = Pool(min(int(cpu_count()/2), 16))
    inputs = [(i, fr) for i, fr in enumerate(frame_list)]
    res = tqdm(pool.imap(part_func, inputs), total=len(frame_list))
    pool.close()
    pool.join()
    print('Done making data!')

    # Add to data structures
    for new_dc in res:
        for key in new_dc:
            data_collect[key].append(new_dc[key])

    # Write indices to the file
    write_idxs(DM, dataset_path, frame_list)
    return data_collect.copy()


def _make_single_naive_data(DM, dataset_path, occ_obj_data, itrace, npts, distance, count_frame_tup):
    icount, frame = count_frame_tup
    calib = DM.get_calibration(frame)
    lidar_altered, attack_xyz_velo = create_inject_frame(DM, occ_obj_data, frame, npts, distance)
    attack_xyz = calib.project_velo_to_rect(attack_xyz_velo[:,None].T)
    # Save and add to data structure
    DM.save_lidar(lidar_altered, os.path.join(dataset_path, 'velodyne'), frame)
    new_data_struct = {'frame':frame,
                       'attack_xyz':attack_xyz,
                       'attack_pts':npts,
                       'attack_dist':distance,
                       'attack_trace':itrace}
    return new_data_struct


def create_inject_frame(DM, occ_obj_data, ifr, npts, distance):
    """Create naive spoofing frame"""
    # Move object
    obj_lidar, obj_bbox = move_to_front(occ_obj_data['lidar'], occ_obj_data['bbox'], distance)
    attack_xyz = np.mean(obj_bbox, axis=0)

    # Add to point cloud
    lidar_altered = np.vstack((DM.get_lidar(ifr), obj_lidar))

    return lidar_altered, attack_xyz


# ========================================================
# FRUSTUM SPOOFING
# ========================================================

def _validate_frustum_inputs(DM, exp_split, inject_type, seed):
    assert inject_type in ['orig-bbox', 'orig-frustum', 'random-box']
    data_collect = make_data_struct()
    random.seed(seed)
    dataset_path = os.path.join(DM.data_dir, exp_split)
    print(dataset_path, flush=True)
    if 'kitti' in DM.name.lower():
        DM.wipe_experiment(dataset_path)
    else:
        raise NotImplementedError
    return dataset_path, data_collect


def make_frustum_sample_spoofing_data(DM, exp_split, npts, dist_rel, nframes,
                                      inject_type, inject_shape,
                                      rescale=False, scale_range=False, gp_consistency=False,
                                      allow_upsample=True, random_frames=True,
                                      seed=None, verbose=False):
    """
    Make a batch of frustum spoofing over a sampled number of frames with single params
    """
    dataset_path, data_collect = _validate_frustum_inputs(DM, exp_split, inject_type, seed)

    # Make samples with multiprocess pool
    idxs_data = DM.idxs
    print('Making frustum sample spoof data using multiprocess')
    params = list(range(nframes))
    part_func = partial(_make_sample_frustum_data, DM, idxs_data, dataset_path, inject_type, npts, dist_rel, random_frames, seed)
    pool = Pool(min(int(cpu_count()/2), 16))
    res = tqdm(pool.imap(part_func, params), total=len(params))
    pool.close()
    pool.join()
    print('Done making the data!')

    # Get outputs
    assert len(res) == len(params)
    for r in res:
        # Add to dataframe
        new_dc = {'frame':r[0],
                  'obj':r[1],
                  'occlusion_num':r[2].occlusion if r[2] is not None else None,
                  'target_xyz':deepcopy(r[2].box3d.t if r[2] is not None else None),
                  'target_dist':r[2].range if r[2] is not None else None,
                  'npts_bbox':r[3],
                  'npts_frust':r[4],
                  'attack_xyz':deepcopy(r[5]),
                  'attack_pts':npts,
                  'attack_dist':dist_rel}
        for key in new_dc:
            data_collect[key].append(new_dc[key])

    if os.path.exists(os.path.join(DM.split_path, 'timestamps.txt')):
        shutil.copyfile(os.path.join(DM.split_path, 'timestamps.txt'),
                        os.path.join(dataset_path, 'timestamps.txt'))
    write_idxs(DM, dataset_path, range(nframes))
    return data_collect


def _make_sample_frustum_data(DM, idxs_data, dataset_path, inject_type, npts, dist_rel, random_frames, seed, irun):
    # Sample frame-object pair
    assert len(idxs_data) > 0

    if random_frames:
        # --- for a random frame, ensure there is a suitable attack point
        while True:
            idx_frame = np.random.RandomState().choice(idxs_data, size=1)[0]
            labels = DM.get_labels(idx_frame, whitelist_types=['Car'])
            if len(labels)>0:
                idx_object = np.random.RandomState().choice(range(len(labels)), size=1)[0]
                # Check number of points
                lidar_pc = DM.get_lidar(idx_frame)
                calib = DM.get_calibration(idx_frame)
                mask_bbox = maskfilters.filter_points_in_object_bbox(lidar_pc, labels[idx_object].box3d, calib)
                if sum(mask_bbox) > 0:
                    break
    else:
        # --- for a predetermined frame, if there isn't a suitable attack, do nothing
        idx_frame = idxs_data[irun]
        idx_object = 0  # TODO: MAKE THIS A POSSIBLE INPUT
        print('always choosing object 0!! improve this')
        calib = DM.get_calibration(idx_frame)
        labels = DM.get_labels(idx_frame, whitelist_types=['Car'])

    if (not random_frames) and (len(labels) < (idx_object+1)):
        # --- NO FRUSTUM ATTACK
        label_orig_ex = None
        npts_bbox = None
        npts_frust = None
        attack_xyz = None
        ground = None
        image = DM.get_image(idx_frame)
        lidar = DM.get_lidar(idx_frame)
        save_data(DM, calib, lidar, labels, image, ground, dataset_path, idx_frame)
    else:
        # --- YES FRUSTUM ATTACK
        # Get base spoofing data using this sample
        image, calib, labels, labels_all, ground, lidar_pc_in, label_orig_ex, mask_frustum, mask_bbox, lidar_orig_ex = _get_single_data_sample(DM, idx_frame, idx_object, inject_type)
        lidar_attack_base = _make_frustum_attack_base(lidar_orig_ex, label_orig_ex, calib, inject_type, npts)

        # Make spoofing data (includes saving)
        params = (-1, irun, npts, dist_rel)
        _, attack_xyz = _make_single_frustum_data(DM, dataset_path, ground, labels_all, image, lidar_pc_in, lidar_attack_base, label_orig_ex, calib, params)
        npts_bbox, npts_frust = sum(mask_bbox), sum(mask_frustum)

    return idx_frame, idx_object, label_orig_ex, npts_bbox, npts_frust, attack_xyz


def make_frustum_mesh_spoofing_data(DM, exp_split, idx_attack_indicators,
                              npts, dist_rel,
                              inject_type, inject_shape,
                              rescale=False, scale_range=False, gp_consistency=False,
                              allow_upsample=True, seed=None, verbose=False):
    """
    Make a batch of frustum spoofing data using a meshgrid of parameters

    :DM - the data manager
    :exp_split - name for saving
    :idx_attack_indicators - [[frame, object], ...] for running attack
    :algorithm - if making it to run on a particular algorithm, this handles the extra steps
    :lidar_preproc - a function for additional preprocessing of lidar data
    :inject_type - choose from {'orig-bbox', 'random-box'}
    :rescale - whether to scale according to range of injections
    :scale_range - whether to scale specifically along range dimension
    :gp_consistency - whether to move vertically so to be consistent with ground plane
    :allow_upsample - when sampling points from the vehicle itself, whether to allow sampling more points
    :inject_shape - distribution shape when using random injections
    :distance - iterable of distance values for injections
    :downsample - iterable of either the fraction of points (if < 1) or number of points (if > 1) to inject
    :frustum_only - whether to only consider points in the frustums or to still generate full frame
    """
    assert len(idx_attack_indicators) > 0
    dataset_path, data_collect = _validate_frustum_inputs(DM, exp_split, inject_type, seed)

    # Loop over frames
    icount = 0
    print('Looping over %d attack indicators' % len(idx_attack_indicators))
    for i_comb, (idx_frame, idx_object) in enumerate(idx_attack_indicators):
        print(f'{i_comb+1} of {len(idx_attack_indicators)}')

        # Get base sample of lidar data
        image, calib, labels, labels_all, ground, lidar_pc_in, label_orig_ex, mask_frustum, mask_bbox, lidar_orig_ex = _get_single_data_sample(DM, idx_frame, idx_object, inject_type)
        lidar_attack_base = _make_frustum_attack_base(lidar_orig_ex, label_orig_ex, calib, inject_type, npts)

        # Create grid of parameters
        distance = dist_rel + label_orig_ex.range
        distance = distance[distance > 0]
        xm, ym = np.meshgrid(npts, distance)
        n_create = np.prod(xm.shape)

        # Loop over parameters
        print('Making frustum mesh spoof data using multiprocess')
        # idx_icount = [(i_idx, i_idx + icount) for i_idx in range(n_create)]
        params = [(i_idx, i_idx+icount, npts, dist) for i_idx, (npts, dist) in enumerate(zip(xm.ravel(), ym.ravel()))]
        icount += n_create
        part_func = partial(_make_single_frustum_data, DM, dataset_path, ground,
                            labels_all, image, lidar_pc_in, lidar_attack_base, label_orig_ex, calib)
        pool = Pool(min(int(cpu_count()/2), 16))
        res = tqdm(pool.imap(part_func, params), total=len(params))
        pool.close()
        pool.join()
        print('Done making data!')

        # Get outputs
        assert len(res) == len(params)
        attack_xyz = np.zeros(xm.shape, dtype=(float,3))
        for r in res:
            idx_tup = np.unravel_index(r[0], xm.shape)
            attack_xyz[idx_tup] = r[1]

        # Append to dataframe
        new_dc = {'frame':idx_frame,
                  'obj':idx_object,
                  'occlusion_num':label_orig_ex.occlusion,
                  'target_xyz':deepcopy(label_orig_ex.box3d.t),
                  'target_dist':label_orig_ex.range,
                  'npts_bbox':sum(mask_bbox),
                  'npts_frust':sum(mask_frustum),
                  'attack_xyz':deepcopy(attack_xyz),
                  'attack_pts':deepcopy(xm),
                  'attack_dist':deepcopy(ym)}
        for key in new_dc:
            data_collect[key].append(new_dc[key])

    if os.path.exists(os.path.join(DM.split_path, 'timestamps.txt')):
        print('Cannot copy timestamps in mesh experiment yet')
    write_idxs(DM, dataset_path, range(icount))
    return data_collect


def _get_single_data_sample(DM, idx_frame, idx_object, inject_type):
    # Get data from inject frame
    region = inject_type.replace('orig-', '') if inject_type in ['orig-bbox', 'orig-frustum'] else 'bbox'
    try:
        ground = DM.get_ground(idx_frame).p_lidar
        save_planes = True
    except FileNotFoundError as e:
        ground = None
        save_planes = False
    image = DM.get_image(idx_frame)
    calib = DM.get_calibration(idx_frame)
    labels = DM.get_labels(idx_frame, whitelist_types=['Car'])
    labels_all = DM.get_labels(idx_frame, whitelist_types='all', ignore_types=None)
    lidar_pc_in = DM.get_lidar(idx_frame)
    label_orig_ex = deepcopy(labels[idx_object])
    mask_frustum = maskfilters.filter_points_in_frustum(lidar_pc_in, label_orig_ex.box2d, calib)
    mask_bbox = maskfilters.filter_points_in_object_bbox(lidar_pc_in, label_orig_ex.box3d, calib)
    lidar_orig_ex = lidar_pc_in[mask_frustum,:] if region == 'frustum' else lidar_pc_in[mask_bbox,:]
    return image, calib, labels, labels_all, ground, lidar_pc_in, label_orig_ex, mask_frustum, mask_bbox, lidar_orig_ex


def _make_frustum_attack_base(lidar_orig_ex, label_orig_ex, calib, inject_type, npts):
    if inject_type in ['random-box']:
        try:
            max_pts = max(npts)
        except:
            max_pts = npts
        if rescale or gp_consistency:
            raise NotImplementedError
        # Take point sample
        if inject_shape == 'moment-match':
            sig_range = 1.02e-01 / 2
            sig_az = 4.54e-01
            sig_el = 1.82e-01
            mean_loc = np.array([-1.10, 0, 0.8])
        elif inject_shape == 'moment-smooth':
            sig_range = 1e-01
            sig_az = 5e-01
            sig_el = 2e-01
            mean_loc = np.array([-1, 0, 1])
        elif inject_shape == 'moment-smooth-2':
            sig_range = 1e-01
            sig_az = 5e-01
            sig_el = 5e-01
            mean_loc = np.array([-1, 0, 1])
        elif inject_shape == 'moment-elongated':
            sig_range = 2
            sig_az = 5e-01
            sig_el = 2e-01
            mean_loc = np.array([-1, 0, 1])
        else:
            raise NotImplementedError

        R_chol = np.diag([sig_range, sig_az, sig_el])
        t_target_velo = calib.project_rect_to_velo(deepcopy(label_orig_ex.box3d.t)[:,None].T)[0,:]
        noise = (R_chol @ np.random.RandomState().randn(3, max_pts)).T
        lidar_pre_range = mean_loc + noise
        if len(lidar_orig_ex[:,3]) > 0:
            lidar_intensity = np.random.RandomState().choice(lidar_orig_ex[:,3],
                                               max_pts, replace=True)[:,None]
        else:
            lidar_intensity = np.random.RandomState().rand(max_pts, 1)

        lidar_attack = np.concatenate((lidar_pre_range, lidar_intensity), axis=1)
        # Rotate points to the angle of the target
        ang_z = np.tan(t_target_velo[1] / t_target_velo[0])
        Rz = tforms.rotz(ang_z)

        # Add on the center of the target
        lidar_attack[:,:3] = lidar_attack[:,:3] @ Rz.T + t_target_velo
        # NOTE: Casting to the dtype is EXTREMELY IMPOTANT
        lidar_attack_base = lidar_attack.astype(dtype=lidar_orig_ex.dtype)
    else:
        lidar_attack_base = lidar_orig_ex
    return lidar_attack_base


def _make_single_frustum_data(DM, dataset_path, ground, labels_all, image,
                              lidar_pc_in, lidar_attack_base, label_orig_ex, calib, params,
                              rescale=False, gp_consistency=False, scale_range=False, allow_upsample=True):
    idx, icount, npts, dist_rel = params
    # idx_tup = np.unravel_index(idx, xm.shape)
    # --- get points
    lidar_pc_ex = deepcopy(lidar_attack_base)
    label_ex = deepcopy(label_orig_ex)
    lidar_pc_ex = percep_attacks.lidar.sample_pc(lidar_pc_ex, keep=npts,
                                upsample=allow_upsample)
    # --- move points
    dist = dist_rel + label_ex.range
    lidar_pc_ex, label_ex = percep_attacks.lidar.range_out_points(
                                             lidar_pc_ex, label_ex, calib,
                                             r_new=dist, rescale=rescale,
                                             gp_consistent=gp_consistency,
                                             scale_range=scale_range)
    # --- merge data
    attack_xyz = calib.project_velo_to_rect(np.mean(
        lidar_pc_ex[:,0:3], axis=0)[:,None].T)
    lidar_pc_merged = np.concatenate((lidar_pc_in, lidar_pc_ex), axis=0)
    save_data(DM, calib, lidar_pc_merged, labels_all, image, ground, dataset_path, icount)
    return idx, attack_xyz


def save_data(DM, calib, lidar, labels, image, ground, dataset_path, icount):
    DM.save_calibration(calib, os.path.join(dataset_path, 'calib'), icount)
    DM.save_lidar(lidar, os.path.join(dataset_path, 'velodyne'), icount)
    DM.save_labels(labels, os.path.join(dataset_path, 'label_2'), icount, False)
    DM.save_image(image, os.path.join(dataset_path, 'image_2'), icount)
    if ground is not None:
        DM.save_ground([-ground[1], -ground[2], ground[0], ground[3]-0.08], os.path.join(dataset_path, 'planes'), icount)
