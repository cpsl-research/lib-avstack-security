# -*- coding: utf-8 -*-
# @Author: spencer@primus
# @Date:   2022-05-30
# @Last Modified by:   spencer@primus
# @Last Modified time: 2022-09-16

import numpy as np
from copy import copy, deepcopy
from functools import partial
import math

from seca.attack.types import Monitor
from avstack import geometry, GroundTruthInformation, sensors, calibration
from avstack import transformations as tforms
from avstack.utils import mean_confidence_interval, maskfilters
from avstack.modules.perception.detections import BoxDetection, CentroidDetection
from avstack.modules.perception.object2dbev import Lidar2dCentroidDetector
from avstack.modules import perception, tracking
from avstack.datastructs import DataContainer


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


class SceneMonitor(Monitor):
    def __init__(self, co_monitor, gp_monitor, obj_monitor):
        self.coordinate_monitor = co_monitor
        self.ground_plane_monitor = gp_monitor
        self.object_monitor = obj_monitor

        self.found_coordinates = False
        self.found_ground = False

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'Scene Monitor with:\n----{self.coordinate_monitor}\n----{self.ground_plane_monitor}\n----{self.object_monitor}'
    
    @property
    def ready(self):
        return self.found_coordinates and self.found_ground

    def get_track_features(self, track):
        """
        get features for scoring
        """
        try:
            position = track.filter.x_vector[:3]
            velocity = track.filter.x_vector[3:6]
        except AttributeError as e:
            position = track.position
            velocity = track.velocity
        features = {'updates':track.n_updates,
                    'angle':abs(np.arctan2(position[1], position[0])),
                    'distance':np.linalg.norm(position),
                    'l-r-vel':velocity[1],
                    'f-b-vel':velocity[0]}
        return features

    def score_track_candidates(self, tracks,
            trk_alive_min=3,
            ang_weight=1, ang_thresh=20*math.pi/180,
            d_weight=1, d_min=5, d_max=40,
            v_l_weight=1, v_l_min=-1.5, v_l_max=1.5,
            v_f_weight=1, v_f_min=-2, v_f_max=4):
        """Find candidate tracks that could be used for attack

        lower score is better
        assumes we are in standard coordinates centered on the ego

        ang
        d
        v
        """
        scores = [1]*len(tracks)

        for i, trk in enumerate(tracks):
            # -- get features
            features = self.get_track_features(trk)

            # -- track alive time
            scores[i] *= np.inf if features['updates'] < trk_alive_min else 1
            if scores[i] == np.inf: continue

            # -- angle off of forward
            ang = features['angle']
            a_fac = np.inf if ang > ang_thresh else ang_weight * sigmoid(ang/ang_thresh)
            scores[i] *= (a_fac)
            if scores[i] == np.inf: continue

            # -- distance bounds
            d = features['distance']
            d_fac = np.inf if not (d_min<=d<=d_max) else d_weight * sigmoid((d-d_min)/(d_max-d_min))
            scores[i] *= d_fac
            if scores[i] == np.inf: continue

            # -- velocity left-right
            v_l = features['l-r-vel']
            v_l_scaled = v_l / v_l_max if ((v_l >= 0) and (v_l_min <= 0)) else v_l / v_l_min
            v_l_fac = np.inf if v_l_scaled > 1 else v_l_weight * sigmoid(v_l_scaled)
            scores[i] *= v_l_fac
            if scores[i] == np.inf: continue

            # -- velocity forward-back
            v_f = features['f-b-vel']
            v_f_scaled = v_f / v_f_max if ((v_f >= 0) and (v_f_min <= 0)) else v_f / v_f_min
            v_f_fac = np.inf if v_f_scaled > 1 else v_f_weight * sigmoid(v_f_scaled)
            scores[i] *= v_f_fac
            if scores[i] == np.inf: continue

        return scores

    def _ingest(self, data):
        self.diagnostics = {'ground_plane':None, 'object_monitor':None}

        # -- transform monitor
        if self.coordinate_monitor is not None:
            C = self.coordinate_monitor(data)
            self.found_coordinates = C is not None
        else:
            C = None

        # -- ground plane monitor
        if self.found_coordinates:
            data_std = C.convert(data, geometry.StandardCoordinates)
            P_ground, self.diagnostics['ground_plane'] = self.ground_plane_monitor(data_std)
            self.found_ground = P_ground is not None
        else:
            P_ground = None
            if self.object_monitor is not None:
                self.object_monitor.n_frames += 1
                self.object_monitor.t += self.object_monitor.dt

        # -- object monitor
        if self.found_ground and (self.object_monitor is not None):
            T_ground = P_ground.as_transform()
            origin_sensor = geometry.Origin(T_ground.translation.vector, geometry.StandardCoordinates.get_conversion_matrix(C))
            tracks, self.diagnostics['object_monitor'] = self.object_monitor(data, origin_sensor=origin_sensor)
            scores = self.score_track_candidates(tracks)
        else:
            tracks = None
            scores = None

        # -- package output
        self.output = {'C':C, 'P_ground':P_ground, 'tracks':tracks, 'track_scores':scores}

    def _distill(self):
        return self.ready, self.output, self.diagnostics


class NaiveSceneMonitor(SceneMonitor):
    def __init__(self, dataset):
        co_monitor = TruthCoordinateMonitor(dataset)

        # gp_monitor = TruthGroundPlaneMonitor(dataset)
        gp_monitor = GroundPlaneViaSensorHeightMonitor()

        obj_monitor = None
        super().__init__(co_monitor, gp_monitor, obj_monitor)


class FullSceneMonitor(SceneMonitor):
    def __init__(self, dataset, framerate, awareness, gpu_ID, save_folder, save=False):
        co_monitor = TruthCoordinateMonitor(dataset)
        gp_monitor = GroundPlaneViaSensorHeightMonitor()

        # obj_monitor = ObjectMonitorWithTruthDetections(data_manager)
        obj_monitor = ObjectMonitorWithLidarDetections(awareness=awareness,
            dataset=dataset, framerate=framerate,
            gpu_ID=gpu_ID, save_folder=save_folder, save=save)
        super().__init__(co_monitor, gp_monitor, obj_monitor)


class PassthroughMonitor(SceneMonitor):
    def __init__(self):
        super().__init__(co_monitor=None, gp_monitor=None, obj_monitor=None)

    @property
    def ready(self):
        return True


# ==============================================================
# COORDINATE MONITOR
# ==============================================================

class TruthCoordinateMonitor(Monitor):
    def __init__(self, dataset):
        self.C = truth_coordinates(dataset)

    def _ingest(self, data):
        pass

    def _distill(self):
        return self.C


class InstantaneousCoordinateMonitor(Monitor):
    pass


class LongitudinalCoordinateMonitor(Monitor):
    pass


# ==============================================================
# GROUND PLANE MONITOR
# ==============================================================

class TruthGroundPlaneMonitor(Monitor):
    def __init__(self, dataset):
        self.plane = truth_ground_plane(dataset)

    def _ingest(self, data):
        pass

    def _distill(self):
        return self.plane


class GroundPlaneViaSensorHeightMonitor(Monitor):
    """Determines the ground plane in the Nominal coordinates
    centered at the lidar sensor"""
    def __init__(self, percentile=0.25, conf_threshold=0.25, n_frames_min=5):
        self.n_frames = 0
        self.height_history = []
        self.conf_threshold = conf_threshold
        self.n_frames_min = n_frames_min
        self.percentile = percentile

    def _ingest(self, data, range_min=0, range_max=40):
        self.n_frames += 1
        pc_z = data[maskfilters.filter_points_range(data, range_min, range_max), 2]
        self.height_history.append(np.quantile(pc_z, self.percentile))

    def _distill(self):
        if self.n_frames > self.n_frames_min:
            m, h = mean_confidence_interval(self.height_history, confidence=0.95)
            if h < self.conf_threshold:
                return geometry.GroundPlane([0, 0, 1, -m], geometry.NominalOriginStandard), {'height':-m}
        return None, {}


# ==============================================================
# OBJECT MONITOR
# ==============================================================

class ObjectMonitor(Monitor):
    def __init__(self, detector, tracker, framerate=10, prob_threshold=0.99, is_bev=False):
        self.detector = detector
        self.tracker = tracker
        self.t = 0
        self.dt = 1/framerate
        self.n_frames = 0
        self.prob_threshold = prob_threshold
        self.is_bev = is_bev
        self.diagnostics = {}

    def _ingest(self, data, origin_sensor, **kwargs):
        self.diagnostics = {}
        if self.is_bev:
            assert data.shape[1] == 2
        else:
            assert data.shape[1] in [3, 4, 5]
        self.n_frames += 1
        calib = calibration.Calibration(origin_sensor)  # not necessary for now
        D = sensors.LidarData(self.t, self.n_frames, data, calib, source_ID=0)
        dets = self.detector(self.n_frames, D, 'attacker_lidar_detections')
        for d in dets:
            d.change_origin(geometry.NominalOriginStandard)
        self.diagnostics['detections'] = deepcopy(dets)
        self.tracker(self.n_frames, dets)
        self.t += self.dt

    def _filter_track(self, track):
        return track
        # return track.confirmed and track.probability >= self.prob_threshold

    def _distill(self, **kwargs):
        tracks = [trk for trk in self.tracker.confirmed_tracks if self._filter_track(trk)]
        self.diagnostics['n_tracks'] = len(self.tracker.tracks)
        self.diagnostics['n_tracks_confirmed'] = len(self.tracker.confirmed_tracks)
        self.diagnostics['confirmed_tracks_filtered'] = deepcopy(tracks)
        return tracks, self.diagnostics


def truth_detector(DM, D):
    frame, timestamp, data = D.frame, D.timestamp, D.data
    dets = [BoxDetection('sensor-0', d.box3d, obj_type=d.obj_type) for d in DM.get_labels(frame)]
    return DataContainer(frame, timestamp, dets)


class ObjectMonitorWithTruthDetections(ObjectMonitor):
    def __init__(self, data_manager, dt=0.1):
        detector = partial(truth_detector, data_manager)
        tracker = tracking.tracker3d.BasicBoxTracker(framerate=1/dt, threshold_coast=6)
        super().__init__(detector, tracker, dt=dt)
        self.DM = data_manager


class ObjectMonitorWithLidarDetections(ObjectMonitor):
    def __init__(self, awareness, dataset, gpu_ID, save_folder, framerate=10, save=False):
        if awareness == 'high':
            save_output = (save_folder != '') and (save)
            detector = perception.object3d.MMDetObjectDetector3D(model='pointpillars',
                dataset=dataset, gpu=gpu_ID, save_output=save_output, save_folder=save_folder)
        elif awareness == 'low':
            raise
        else:
            raise NotImplementedError(f'Awareness of {awareness} not implemented')
        save_output = save_folder != ''
        tracker = tracking.tracker3d.BasicBoxTracker(framerate=framerate,
            save_output=save_output, save_folder=save_folder)
        super().__init__(detector, tracker, framerate=framerate)

# ==============================================================
# UTILITIES
# ==============================================================

def truth_coordinates(dataset):
    if dataset.lower() == 'kitti':
        C = geometry.LidarCoordinates
    elif dataset.lower() == 'nuscenes':
        C = geometry.LidarCoordinatesYForward
    else:
        raise NotImplementedError
    return C


def truth_ground_plane(dataset):
    """Ground planes are defined in the sensor frame"""
    if dataset.lower() == 'kitti':
        # just picked a random plane...
        p = np.array([3.493957e-03, -9.999935e-01, 9.128743e-04, 1.689673e+00])
        gp = geometry.GroundPlane(p,
            geometry.CameraCoordinates).convert(geometry.LidarCoordinates, dz=0.08)
    else:
        raise NotImplementedError(dataset)
    return gp
