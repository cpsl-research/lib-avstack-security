# @Author: Spencer Hallyburton <spencer>
# @Date:   2021-06-04
# @Filename: defenses.py
# @Last modified by:   spencer
# @Last modified time: 2021-09-09

import sys
import os
import shutil
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from copy import copy, deepcopy

import numpy as np


path = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(path)))

# -------------
# AV utilities
from avstack.utils import maskfilters
from avstack.geometry import bbox

from avapi import evaluation

# =================================================
# DEFENSE CLASSES
# =================================================

class PerceptionDefense():
    def __init__(self, DM, defense_eval_func, defense_name, defense_type):
        """Set up a defense evaluator and type of defense"""
        self.DM = DM
        self.defense_name = defense_name.lower()
        self.defense_type = defense_type.lower()
        self.defense_eval_func = defense_eval_func

    def set_data_manager(self, DM):
        self.DM = DM

    def get_detection_results_from_folder(self, path_to_results, idxs=None):
        """Load detection results from a folder"""
        res_all, idx_all = self.DM.get_labels_from_directory(dir=path_to_results, idxs=idxs)
        results = {idx:res for idx, res in zip(idx_all, res_all)}
        return results

    def get_detection_results_from_batch(self):
        """Load detection results from batch pickles"""
        raise NotImplementedError

    def get_write_path(self, result_path, write_path):
        if write_path is None:
            write_path = result_path
            if 'defense' not in result_path:
                if write_path.endswith('/'):
                    write_path = write_path[:-1]
                write_path += '-{}_defense'.format(self.defense_name)
        return write_path

    def write_detection_results_from_folder(self, results, result_path, write_path):
        write_path = self.get_write_path(result_path, write_path)
        if not os.path.exists(write_path):
            os.makedirs(write_path)

        # Check for index match
        for idx, res in results.items():
            self._write_detection_single(res, idx, write_path)
        return write_path

    def _write_detection_single(self, result, idx, path):
        self.DM.save_labels(result, path, idx, add_label_folder=False)

    def evaluate(self):
        """Run defense on a set of information"""
        raise NotImplementedError

    def evaluate_from_folder(self, path_to_results, idxs=None, write=False, write_path=None):
        """Evaluate postprocess lidar defense"""
        results = self.get_detection_results_from_folder(path_to_results, idxs=idxs)
        passed_all = {}

        # Check and delete
        print('Running defense on detection results -- using multiprocess')
        pool = Pool(min(int(cpu_count()/2), 16))
        part_func = partial(self.defense_eval_func, self.DM, self.evaluate)
        idxs = [int(idx) for idx in results]
        inputs = [(idx, res) for idx, res in results.items()]
        res = tqdm(pool.imap(part_func, inputs), total=len(inputs))
        pool.close()

        # Stitch back together
        new_results = {}
        passed_all = {}
        for idx, r in zip(idxs, res):
            new_results[idx] = r[0]
            passed_all[idx] = r[1]

        # write detection results
        if write:
            write_path = self.write_detection_results_from_folder(new_results, path_to_results, write_path)
        else:
            write_path = None

        return new_results, passed_all, write_path


def _run_postproc_lidar_def(DM, evaluate_func, input):
    """
    Run a lidar-based postprocessing defense

    :input -- idx=frame index; result=list of detection labels in this frame
    """
    idx, result = input
    new_passed = []
    lidar = DM.get_lidar(idx)
    calib = DM.get_calibration(idx)
    idx_remove = []
    removed = False
    idx_keep = list(range(len(result)))
    for ilab, label in enumerate(result):
        passed, ratio = evaluate_func(lidar, label, calib)
        new_passed.append(passed)
        if not passed:
            removed = True
            idx_keep.remove(ilab)
    if len(idx_keep) > 0:
        new_result = np.asarray(result)[np.asarray(idx_keep)]
    else:
        new_result = []
    return new_result, new_passed


def _run_postproc_voting_def(DM, evaluate_func, input):
    """
    Run a voting-based postprocessing defense

    :input -- idx=frame index; result_lidar=list of lidar detections, result_image=list of image detections
    """
    idx, (labels_lidar, labels_image) = input
    # calib = DM.get_calibration(idx)
    passed, nvotes = evaluate_func(labels_lidar, labels_image)
    if len(labels_lidar)==0 or len(passed)==0:
        new_result = []
    else:
        new_result = np.asarray(labels_lidar)[passed]
    return new_result, passed


def _run_postproc_camera_lidar_def(DM, evaluate_func, input):
    """
    Run a camera-lidar-based postprocessing defense
    """
    raise NotImplementedError


class LpdDefense(PerceptionDefense):
    def __init__(self, DM):
        super().__init__(DM, _run_postproc_lidar_def, 'lpd', 'postprocess')
        self.threshold = 0.8

    def evaluate(self, lidar, label, calib):
        passed, ratio = LPD(lidar, label, calib, self.threshold)
        return passed, ratio


class GbscDefense(PerceptionDefense):
    def __init__(self, DM):
        super().__init__(DM, _run_postproc_lidar_def, 'gbsc', 'postprocess')
        self.threshold = 0.2
        self.alpha = 0.4

    def get_adaptive_threshold(self, lidar, labels):
        """
        Gets an adaptive threshold from the batch of data
        """
        pass

    def evaluate(self, lidar, label, calib):
        passed, score = GBSC(lidar, label, calib, alpha=self.alpha, threshold=self.threshold)
        return passed, score


class ImageLidar3DCheck(PerceptionDefense):
    """
    Post-processing defense that compares detections from camera and lidar
    """
    def __init__(self, DM, path_to_im_results, assume_image_good=False):
        super().__init__(DM, _run_postproc_voting_def, 'imagelidar3d', 'postprocess')
        self.metric='3D_IoU'
        self.assume_image_good = assume_image_good
        self.path_to_im_results = path_to_im_results
        if self.assume_image_good:
            raise NotImplementedError

    def get_detection_results_from_folder(self, path_to_results, idxs=None):
        """overwrite the loading of detections to handle the multiple paths"""
        # Get lidar results
        res_lidar, idx_all = self.DM.get_labels_from_directory(dir=path_to_results, idxs=idxs)
        # Get image results
        assert os.path.exists(self.path_to_im_results), 'Need to create 3D image results at %s'%path_to_im_results
        res_image, idx_all = self.DM.get_labels_from_directory(dir=self.path_to_im_results, idxs=idxs)
        assert len(res_lidar)==len(res_image), 'Lidar (%i) and image (%i) results need to have same number of frames'%(len(res_lidar), len(res_image))
        return {idx:[res_li, res_im] for idx, res_li, res_im in zip(idx_all, res_lidar, res_image)}

    def evaluate(self, labels_lidar, labels_image):
        """evaluate by associating detections"""
        assignments, A = evaluation.associate_detections_truths(labels_lidar, labels_image, metric=self.metric)

        # Get passed list
        idx_lidar_good = [a[0] for a in assignments]
        passed = np.array([i in idx_lidar_good for i in range(len(labels_lidar))])
        nvotes = np.array([int(p) for p in passed])
        if self.assume_image_good:
            raise NotImplementedError
        return passed, nvotes

#
# class LidarMonoMonoVoting_v1():
#     """
#     Specific defense technique comparing LiDAR and two 3D-from-mono algorithms
#
#     e.g., use --Frustum Pointnet (3D from Fusion)
#               --Pseudo-LiDAR (3D from Mono)
#               --3D-Deepbox (3D from Mono)
#
#     Particularly useful to guard against attacks on LiDAR with an unattacked camera module
#     """
#     def __init__(self):
#         super().__init__()
#
#     def merge_multiple_detections(self, lidar_det=None, mono1_det=None, mono2_det=None):
#         # One must be populated
#         if lidar_det is not None:
#             new_det = lidar_det
#         elif (mono1_det is not None) or (mono2_det is not None):
#             method = 'first'
#             if method == 'first':
#                 new_det = mono1_det
#             elif method == 'second':
#                 new_det = mono2_det
#             elif method == 'center_avg':
#                 new_det = deepcopy(mono1_det)
#                 new_det.box3d.t = (new_det.box3d.t + mono2_det.box3d.t) / 2
#             else:
#                 raise NotImplementedError
#         else:
#             raise RuntimeError
#             new_det = None
#
#         return new_det
#
#     def test(self, lidar_dets, mono1_dets, mono2_dets, allow_chain_overlap=True):
#         """
#         Main test routine for the defense
#
#         NOTE: THIS IS VERY UNOPTIMIZED AT THE MOMENT. OPTIMIZATION COMES LATER
#         NOTE: HAVE NOT DECIDED WHETHER TO REMOVE FROM BUFFER ON ASSIGNMENT OR TO KEEP HANGING AROUND FOR LATER....
#
#         INPUTS:
#         --lidar_dets --> list of detections from lidar
#         --mono1_dets --> list of detections from first mono alg
#         --mono2_dets --> list of detections from second mono alg
#         --allow_chain_overlap --> boolean of whether to allow chained overlaps
#         (True) i.e., associative property, or to force >=3 overlaps to all have to overlap with each other (False)
#
#         Idea is to go through LiDAR detections first and determine mono
#         overlaps. Then go through remaining mono detections.
#         """
#         assert(allow_chain_overlap)  # for now...
#
#         # Data structures for outcomes
#         res_valid = []
#         res_fp = []
#
#         # In unoptimized fashion, just pre-compute all association matrices
#         assignments_lidar_mono1, idx_solo_lidar_vs_mono1, idx_solo_mono1_vs_lidar, A1 = \
#             eval.associate_detections_truths(lidar_dets, mono1_dets, metric='3D_IoU')
#
#         assignments_lidar_mono2, idx_solo_lidar_vs_mono2, idx_solo_mono2_vs_lidar, A2 = \
#             eval.associate_detections_truths(lidar_dets, mono2_dets, metric='3D_IoU')
#
#         assignments_mono1_mono2, idx_solo_mono1_vs_mono2, idx_solo_mono2_vs_mono1, A3 = \
#                         eval.associate_detections_truths(mono1_dets, mono2_dets, metric='3D_IoU')
#
#         # Now loop------
#         for i, li_det in enumerate(lidar_dets):
#             # Check if we have overlaps
#             overlap_1 = i not in idx_solo_lidar_vs_mono1
#             overlap_2 = i not in idx_solo_lidar_vs_mono2
#
#             # Get overlap object
#             if overlap_1:
#                 mono1_overlap = mono1_dets[dict(assignments_lidar_mono1)[i]]
#             else:
#                 mono1_overlap = None
#
#             if overlap_2:
#                 mono2_overlap = mono2_dets[dict(assignments_lidar_mono2)[i]]
#             else:
#                 mono2_overlap = None
#
#             # Get string
#             if overlap_1 and overlap_2:
#                 merge_str = 'all'
#             elif overlap_1 or overlap_2:
#                 merge_str = 'lidar_and_mono{}'.format(1 if overlap_1 else 2)
#
#             # Append result
#             if overlap_1 or overlap_2:
#                 det = self.merge_multiple_detections(lidar_det=li_det,
#                                                      mono1_det=mono1_overlap,
#                                                      mono2_det=mono2_overlap)
#                 res_valid.append((det, merge_str))
#             else:
#                 res_fp.append((li_det, 'lidar'))
#
#         # Go through first mono detections that didn't align with lidar
#         for i_mon1 in idx_solo_mono1_vs_lidar:
#             if i_mon1 in idx_solo_mono1_vs_mono2:
#                 res_fp.append((mono1_dets[i_mon1], 'mono1'))
#             else:
#                 # Merge
#                 i_mon2 = dict(assignments_mono1_mono2)[i_mon1]
#                 det = self.merge_multiple_detections(mono1_det=mono1_dets[i_mon1],
#                                                      mono2_det=mono2_dets[i_mon2])
#                 res_valid.append((det, 'mono1_and_mono2'))
#
#         # Only left are invalid mono2 detections
#         for i in idx_solo_mono2_vs_lidar:
#             if i in idx_solo_mono2_vs_mono1:
#                 res_fp.append((mono2_dets[i], 'mono2'))
#
#         return res_valid, res_fp


def factory_defense(defense_type, DM):
    if defense_type.lower() == 'lpd':
        DEF = LpdDefense(DM)
    elif defense_type.lower() == 'gbsc':
        DEF = GbscDefense(DM)
    else:
        raise NotImplementedError
    return DEF


# =================================================
# DEFENSE FUNCTIONS
# =================================================

def GBSC(lidar, label, calib, alpha=0.4, threshold=0.2):
    """Ghostbuster-Shadowcatcher defense function"""
    # Get shadow points
    shadow_filter = maskfilters.filter_points_in_shadow(lidar, label.box2d, label.box3d, calib)

    # Run anomaly score
    score = get_shadow_anomaly_score(lidar[shadow_filter,:], label, calib, alpha=alpha)
    return score <= threshold, score


def LPD(lidar, label, calib, threshold=0.8):
    # Get number of points in frustum (denom)
    frustum_filter = maskfilters.filter_points_in_image_frustum(lidar, label.box2d, calib)
    npts_frustum = sum(frustum_filter)

    # Get number of points behind
    half_box_len = label.box3d.l / 2
    npts_behind = sum((np.linalg.norm(lidar[frustum_filter,0:3], axis=1) - \
                   np.linalg.norm(label.box3d.t)) > half_box_len)

    # Threshold to say if passed
    if npts_frustum == 0:
        ratio = 0
    else:
        ratio = npts_behind / npts_frustum
    passed = ratio < threshold

    return passed, ratio


def compare_3d_detections_image_lidar(detections_image=None, detections_lidar=None, detections_fusion=None, metric='center_dist', radius=5):
    """
    Compares the 3D detections from different selections of sensors

    Arguments
    :detections_image - detections from image-only algorithms
    :detections_lidar - detections from lidar-only algorithms
    :detections_fusion - detections from fusion algorithms
    :metric - the metric to use to compare detections during assignment
    :radius - (if applicable) a radius to use for comparing detections

    Returns
    :consistent - array of detections that are consistent between modalities
    :inconsistent - list of detections from each modality that are inconsistent
    """

    # Only 2 things should be passed in
    assert(sum([detections_image is not None, detections_lidar is not None, detections_fusion is not None]) == 2)

    # Discover which ones we are comparing
    if detections_image is not None:
        detections_A = detections_image
        if detections_lidar is not None:
            detections_B = detections_lidar
        else:
            detections_B = detections_fusion
    else:
        detections_A = detections_lidar
        detections_B = detections_fusion

    # Run assignment algorithm
    assignments, A = evaluation.associate_detections_truths(
                                    detections=detections_A,
                                    truths=detections_B,
                                    metric=metric,
                                    radius=radius)

    # Return which are consistent and which are inconsistent
    consistent = assignments
    inconsistent = [detections_A[assignments.unassigned_rows], detections_B[assignments.unassigned_cols]]
    return consistent, inconsistent


def compare_2d_3d_image_bbox(bbox_3d, bbox_2d, calib, only_in_image=False, image_size=[375, 1242]):
    """
    Compare the 3D detection to the 2D detection in image and check consistency

    Returns:
    - inter -- the intersection of the 2D bboxes
    - inter_ratio -- the ratio of intersection to the bbox_2d area
    - union -- the union of the 2D bboxes
    - union_ratio -- the ratio of union to the bbox_2d area
    - iou -- the ratio of intersection to union
    """
    # Project the 3d box into 2d
    bbox_2d_from_3d = bbox_3d.project_to_2d_bbox(calib)
    bbox_2d_from_3d_area = bbox.box_area(bbox_2d_from_3d.box2d)
    bbox_2d_area = bbox.box_area(bbox_2d.box2d)

    # Only consider the part of the projected box that's in the image, if enabled
    if only_in_image:
        xmin = max(0, bbox_2d_from_3d.box2d[0])
        ymin = max(0, bbox_2d_from_3d.box2d[1])
        xmax = max(0, min(image_size[1], bbox_2d_from_3d.box2d[2]))
        ymax = max(0, min(image_size[0], bbox_2d_from_3d.box2d[3]))
        bbox_2d_from_3d = bbox.Box2D([xmin, ymin, xmax, ymax])

    # Compare the detections
    inter = bbox.box_intersection(bbox_2d.box2d, bbox_2d_from_3d.box2d)
    union = bbox.box_union(bbox_2d.box2d, bbox_2d_from_3d.box2d)
    iou = inter / union
    area_ratio = bbox_2d_from_3d_area / bbox_2d_area

    # TODO: add in threshold calculation on some metric

    return inter, inter/bbox_2d_area, union, union/bbox_2d_area, iou, area_ratio


# =======================================================
# MULTI-ALGORITHM VOTING AND ASSIGNMENT
# =======================================================


# =======================================================
# Utilities
# =======================================================

def normalize(vec):
    return vec / np.linalg.norm(vec)


def get_dist_point_line_vec(x_pts, line):
    """
    vectorized point to line distance for 2D

    x_pts -- n x 2
    line -- 2x2 where each line is a column
    """
    P1 = line[:,0]
    P2 = line[:,1]
    x0, y0 = x_pts[:,0], x_pts[:,1]
    x1, y1 = P1
    x2, y2 = P2
    numer = np.abs((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1))
    denom = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    return numer / denom


def get_shadow_vectors(label, calib):
    """
    Gets shadow vectors in the lidar frame of reference
    """
    # Project bounding box corners to BEV
    corners_rect = label.box3d.get_corners()
    corners_velo = calib.project_rect_to_velo(corners_rect)
    corners_velo_bev = calib.project_velo_to_bev(corners_velo)

    # Get BEV geometry of the shadow region
    slopes = corners_velo_bev[:,0] / corners_velo_bev[:,1]
    shadow_vectors = [normalize(corners_velo_bev[np.argmax(slopes),:]),
                      normalize(corners_velo_bev[np.argmin(slopes),:])]

    # Compute shadow length
    H_laser = 1.73
    h_obj = label.box3d.t[1]
    shadow_length = label.get_range() * h_obj / max(1e-4, abs(H_laser - h_obj))
    max_shadow = 20
    shadow_length = min(shadow_length, max_shadow+label.get_range())

    # Package up vectors in BEV
    # -- 1: start-line top left
    # -- 2: end-line top/bottom left
    # -- 3: end-line bottom/top right
    # -- 4: start-line bottom right
    v1 = corners_velo_bev[np.argmax(slopes),:]
    v2 = shadow_length*shadow_vectors[0]
    v3 = shadow_length*shadow_vectors[1]
    v4 = corners_velo_bev[np.argmin(slopes),:]
    return np.array([v1, v2, v3, v4]).T


def get_shadow_anomaly_score(points_shadow, label, calib, alpha=0.4):
    """
    Gets the shadow-catcher score

    the shadow region is defined by 4 lines:
    --start_line = the line right behind the object
    --end_line = the line definining the maximum length of the shadow region
    """
    shadow_vectors = get_shadow_vectors(label, calib)
    points_shadow_bev = calib.project_velo_to_bev(points_shadow)[:,:2]

    # Only do in hull points
    in_sub_shadow = bbox.in_hull(points_shadow_bev, shadow_vectors.T)
    point_sub_shadow_bev = points_shadow_bev[in_sub_shadow,:]
    T = point_sub_shadow_bev.shape[0]

    if T == 0:
        score = 0
    else:
        log_inner = np.log(0.5)/alpha

        # Define lines
        line_start = shadow_vectors[:,[0,3]]
        line_end = shadow_vectors[:,[1,2]]
        line_top = shadow_vectors[:,[0,1]]
        line_bottom = shadow_vectors[:,[3,2]]
        line_mid = np.mean(np.concatenate((line_top[:,:,None], line_bottom[:,:,None]), axis=2), axis=2)

        # Get distances
        x_start = get_dist_point_line_vec(point_sub_shadow_bev, line_start)
        x_mid = get_dist_point_line_vec(point_sub_shadow_bev, line_mid)
        x_end = get_dist_point_line_vec(point_sub_shadow_bev, line_end)
        x_bound_1 = get_dist_point_line_vec(point_sub_shadow_bev, line_top)
        x_bound_2 = get_dist_point_line_vec(point_sub_shadow_bev, line_bottom)
        x_bound = np.minimum(x_bound_1, x_bound_2)

        # Compute weightings
        w_start = np.exp(log_inner * x_start/(x_start+x_end))
        w_mid = np.exp(log_inner * x_mid/(x_mid+x_bound))
        w_min = np.exp(log_inner)
        score = (sum(w_start*w_mid) - T*w_min**2) / (T*(1-w_min**2))

    return score
