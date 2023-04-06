import numpy as np
from avstack import geometry
from avstack import transformations as tforms
from seca.attack.types import Scheduler
from copy import copy, deepcopy


class ReplayScheduler(Scheduler):
    """Replay the sensor data
    
    The "stable" portion is when we build up the replay buffer

    The "attack" portion is when we apply the buffer
    """
    def __init__(self, dt_burnin=0, dt_stable=3, dt_attack=3, framerate=5, dt_repeat=0.5):
        super().__init__(dt_burnin, dt_stable+dt_repeat, dt_attack, framerate)
        self.dt_repeat = dt_repeat
        self.n_frames_buffer = round(dt_stable * self.framerate)

    def _schedule_stable(self, info):
        diagnostics = {}
        if self.n_frames_ready <= (self.n_frames_burnin+self.n_frames_buffer):
            info['mode'] = 'buffer'
        else:
            info['mode'] = 'repeat'
        diagnostics['mode'] = info['mode']
        info['attack_target'] = 'universal'
        return info, diagnostics

    def _schedule_attack(self, info):
        diagnostics = {}
        info['mode'] = 'replay'
        diagnostics['mode'] = info['mode']
        info['attack_target'] = 'universal'
        return info, diagnostics


class _NaiveMoveObjectScheduler(Scheduler):
    """Make an object move somewhere"""
    def __init__(self, init_razel, final_razel, dt_attack,
            dt_burnin, dt_stable, framerate, attack_profile):
        super().__init__(dt_burnin, dt_stable, dt_attack, framerate)
        self.init_razel = init_razel
        self.final_razel = final_razel
        self.razel_noise = np.array([1e-1, 1e-8, 0])
        self.attack_profile = attack_profile
        self.last_pts_ground = None

        # -- extras...for "acceleration" or "jerk" type attack
        self._v_i = 0
        self._v_f = 0
        self._a_i = 0
        self._a_f = 0
        self._a = None
        self._j = None

    def _schedule_stable(self, info):
        """Schedule the 'stabilizing' part of the attack"""
        diagnostics = {}
        # C = info['C']
        pts_razel = self.init_razel + np.sqrt(self.razel_noise) * np.random.randn(3)
        pts_ground = tforms.spherical_to_cartesian(pts_razel, geometry.StandardCoordinates)  # always in standard coordinates
        self.last_pts_ground = pts_ground
        info['pts_ground'] = [pts_ground]
        info['attack_target'] = [pts_ground]
        return info, diagnostics

    def _schedule_attack(self, info):
        """Schedule the 'attacking' part of the attack"""
        diagnostics = {}
        C = info['C']
        pts_ground_last = self.last_pts_ground
        pts_ground_final = tforms.spherical_to_cartesian(self.final_razel, geometry.StandardCoordinates)  # always in standard coordinates
        n_frames_to_go = self.n_frames_attack - (self.n_frames_ready-self.n_frames_burnin-self.n_frames_stable)
        if n_frames_to_go <= 0:
            pts_ground = pts_ground_final
        else:
            delta = pts_ground_last - pts_ground_final
            dist = np.linalg.norm(delta)
            vector = delta / dist
            if self.attack_profile == 'linear':
                pts_ground = pts_ground_last - delta/n_frames_to_go
            elif self.attack_profile == 'quadratic':
                pts_ground = pts_ground_last - delta/n_frames_to_go**2
            elif self.attack_profile == 'piecewise-acceleration':
                if self._a is None:  # set acceleratoin based on parameters
                    self._a = -1  # initialize with just 1 m/s^2
                    self._mode = 'init'
                    self._switch_at_n_frames = n_frames_to_go - 3
                elif (n_frames_to_go <= self._switch_at_n_frames) and (self._mode == 'init'):
                    self._mode = 'final'
                    dt_left = self.dt * n_frames_to_go
                    self._a = -2*dist / dt_left**2
                self._v_i = self._v_f
                self._v_f = self._v_f + self._a * self.dt
                v = (self._v_i + self._v_f)/2  # approximate as the middle
                pts_ground = pts_ground_last + vector * (v*self.dt + 1/2*self._a*self.dt**2)
            elif self.attack_profile == 'jerk':
                # -- initialize jerk
                if self._j is None:
                    dt_left = self.dt * n_frames_to_go
                    self._j = -6*dist / dt_left**3
                    print(f'Initialized jerk at {self._j:.2f} m/s^3')

                # -- accel
                self._a_i = self._a_f
                self._a_f = self._a_f + self._j*self.dt
                a = (self._a_i + self._a_f)/2  # approximate as the middle

                # -- vel
                self._v_i = self._v_f
                self._v_f = self._v_f + a*self.dt + 1/2*self._j*self.dt**2
                v = (self._v_i + self._v_f)/2  # approximate as the middle

                # -- position
                pts_ground = pts_ground_last + vector * (v*self.dt + 1/2*a*self.dt**2 + 1/6*self._j*self.dt**3)
            else:
                raise NotImplementedError
        self.last_pts_ground = pts_ground
        info['pts_ground'] = [pts_ground]
        info['attack_target'] = [pts_ground]
        return info, diagnostics


class _FrustumMoveObjectScheduler(Scheduler):
    """Move an object using frustum attack"""
    def __init__(self, init_range, final_range, dt_attack,
        dt_burnin, dt_stable, framerate, attack_profile, maintain_frustum=False):
        super().__init__(dt_burnin, dt_stable, dt_attack, framerate)
        self.last_pts_ground = None
        self.init_range = init_range
        self.final_range = final_range
        self.range_noise = 1e-1
        self.i_frames_attack = 0
        self.delta_attack = None
        self.attack_profile = attack_profile
        self.maintain_frustum = maintain_frustum
        self.trk_pos_init = None
        self.trk_ID = None
        self.last_trk_target = None
        self.last_xyz_size = None

        # -- extras...for "acceleration" or "jerk" type attack
        self._v_i = -2
        self._v_f = -2
        self._a_i = 0
        self._a_f = 0
        self._a = None
        self._j = None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'Frustum object move scheduler: initial range:' +\
            f'{self.init_range}, final range: {self.final_range}.'+\
            f'Stable for {self.n_frames_stable}, attacking for {self.n_frames_attack}.'

    def _get_track_box(self, tracks, track_scores):
        # -- set the tracking ID
        if self.trk_ID is None:
            if (len(tracks)==0) or np.all(np.isinf(track_scores)) or np.isnan(track_scores).all():
                return None, None
            else:
                trk_sel = tracks[np.argmin(track_scores)]
                self.trk_ID = trk_sel.ID
                print(f'selected track ID {self.trk_ID} as targeted vehicle')

        # -- find the position of the track
        for trk in tracks:
            if (trk.ID == self.trk_ID) or ((self.last_trk_target is not None) and \
                    (np.linalg.norm(trk.position -self.last_trk_target.position) < 1)):
                trk_box = trk.box3d
                cg = trk_box.corners_global
                xyz_size = np.max(cg, axis=0) - np.min(cg, axis=0)
                self.last_trk_target = trk
                self.last_xyz_size = xyz_size
                self.trk_ID = trk.ID
                break
        else:
            print(f'Could not find track with ID {self.trk_ID}...using last position')
            trk_box = self.last_trk_target.box3d
            xyz_size = self.last_xyz_size
        return trk_box, xyz_size

    def _schedule_stable(self, info):
        """Schedule the 'stabilizing' part of the attack"""
        diagnostics = {}
        trk_box, xyz_size = self._get_track_box(info['tracks'], info['track_scores'])
        info['track_box'] = trk_box
        if trk_box is None:
            info = None
        else:
            trk_position = trk_box.t
            s = self.init_range / np.linalg.norm([trk_position[:2]])
            pts_ground = np.array([s*trk_position[0], s*trk_position[1], 0])  # HACK for now
            self.last_pts_ground = pts_ground
            info['pts_ground'] = [pts_ground]
            info['attack_target'] = [pts_ground]
            info['box_size'] = [xyz_size]
        return info, diagnostics

    def _schedule_attack(self, info):
        """Schedule the 'attacking' part of the attack"""
        diagnostics = {}
        pts_ground_last = self.last_pts_ground
        n_frames_to_go = self.n_frames_attack - (self.n_frames_ready-self.n_frames_burnin-self.n_frames_stable)
        info['track_box'] = None

        # -- get the delta attack between frames
        if self.delta_attack is None:
            pos = self.last_trk_target.position
            vector = np.array([pos[0], pos[1], 0]) / np.linalg.norm([pos[:2]])
            self.pts_ground_final = self.final_range * vector
            self.delta_attack = (self.pts_ground_final - pts_ground_last) / self.n_frames_attack
        dist = np.linalg.norm(pts_ground_last[:2]) - self.final_range

        # -- place relative to frustum or track
        if n_frames_to_go <= 0:
            pts_ground = pts_ground_last
            xyz_size = self.last_xyz_size
        else:
            if self.maintain_frustum:
                trk_box, xyz_size = self._get_track_box(info['tracks'], info['track_scores'])
                info['track_box'] = trk_box
                trk_position = trk_box.t
                if self.attack_profile == 'linear':
                    s = np.linalg.norm(pts_ground_last + self.delta_attack) / np.linalg.norm([trk_position[:2]])
                    pts_ground = np.array([s*trk_position[0], s*trk_position[1], 0])  # HACK for now
                elif self.attack_profile == 'acceleration':
                    raise
                elif self.attack_profile == 'jerk':
                    # -- initialize jerk
                    if self._j is None:
                        dt_left = self.dt * (n_frames_to_go)
                        self._j = -6*dist / dt_left**3
                        print(f'Initialized jerk at {self._j:.2f} m/s^3')
                    # -- accel
                    self._a_i = self._a_f
                    self._a_f = self._a_f + self._j*self.dt
                    a = (self._a_i + self._a_f)/2  # approximate as the middle
                    # -- vel
                    self._v_i = self._v_f
                    self._v_f = self._v_f + a*self.dt + 1/2*self._j*self.dt**2
                    v = (self._v_i + self._v_f)/2  # approximate as the middle
                    # -- position
                    vector = np.array([trk_position[0], trk_position[1], 0]) / np.linalg.norm(trk_position[:2]) # HACK for now
                    delta = v*self.dt + 1/2*a*self.dt**2 + 1/6*self._j*self.dt**3
                    s = dist + delta + self.final_range
                    pts_ground = s * vector
                else:
                    raise NotImplementedError(self.attack_profile)
            else:
                raise NotImplementedError
        info['box_size'] = [xyz_size]
        self.last_pts_ground = pts_ground
        info['pts_ground'] = [pts_ground]
        info['attack_target'] = [pts_ground]
        return info, diagnostics


# =================================================
# NAIVE ATTACKS
# =================================================

class NaiveObjectStopScheduler(_NaiveMoveObjectScheduler):
    """Make an object appear like it is stopping"""
    def __init__(self, dt_attack=4.5, attack_profile='jerk', dt_burnin=0, dt_stable=3, framerate=5,
            init_range=25, final_range=4):
        init_razel = np.array([init_range, 0, 0])  # defined in ground frame
        final_razel = np.array([final_range, 0, 0])  # defined in ground frame
        super().__init__(init_razel, final_razel, dt_attack, dt_burnin, dt_stable, framerate, attack_profile)


class NaiveObjectDriftScheduler(_NaiveMoveObjectScheduler):
    """Make an object appear like it is drifing laterally"""
    def __init__(self, dt_attack=4.5, attack_profile='jerk', dt_burnin=0, dt_stable=3, framerate=5):
        init_razel = np.array([10, 0.4, 0])  # defined in ground frame
        final_razel = np.array([10, 0, 0])  # defined in ground frame
        super().__init__(init_razel, final_razel, dt_attack, dt_burnin, dt_stable, framerate, attack_profile)


class NaiveObjectAccelerateScheduler(_NaiveMoveObjectScheduler):
    """Make an object appear like it is accelerating forward"""
    def __init__(self, dt_attack=4.5, attack_profile='jerk', dt_burnin=0, dt_stable=3, framerate=5):
        init_razel = np.array([8, 0, 0])  # defined in ground frame
        final_razel = np.array([20, 0, 0])  # defined in ground frame
        super().__init__(init_razel, final_razel, dt_attack, dt_burnin, dt_stable, framerate, attack_profile)


# =================================================
# FRUSTUM ATTACKS
# =================================================

class ObjectRemoveScheduler(_FrustumMoveObjectScheduler):
    """Make an object disappear"""
    def __init__(self, dt_attack=4, attack_profile='linear', dt_burnin=0, dt_stable=3, framerate=5):
        init_range = None
        final_range = None
        super().__init__(init_range, final_range, dt_attack, dt_burnin, dt_stable, framerate,
            attack_profile, maintain_frustum=True)


class FrustumObjectStopScheduler(_FrustumMoveObjectScheduler):
    """Make an object appear like it is stopping"""
    def __init__(self, dt_attack=4, attack_profile='linear', dt_burnin=0,
            dt_stable=3, framerate=5, init_range=25, final_range=4):
        super().__init__(init_range, final_range, dt_attack, dt_burnin, dt_stable, framerate,
            attack_profile, maintain_frustum=True)
