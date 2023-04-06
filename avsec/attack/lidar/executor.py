
import os, random, pickle
import numpy as np
import scipy.interpolate as si
from scipy.spatial import ConvexHull, Delaunay
from scipy.ndimage import binary_dilation, binary_closing
from sklearn.neighbors import KNeighborsRegressor

from seca.attack.types import Executor
import avstack
from avstack.utils import maskfilters
from avstack import geometry
import avstack.transformations as tforms


"""
info class:
- pts_ground --> scheduled attack centroid points in the ground frame
- P_ground --> ground plane parameters in the sensor frame
"""


def load_traces(trace_directory, n_traces, shuffle):
    """Load traces data"""
    trace_data = []
    for file in os.listdir(trace_directory):
        if file.endswith('.p'):
            with open(os.path.join(trace_directory, file), 'rb') as f:
                data = pickle.load(f)
            trace_data.append(data)
    trace_data = sorted(trace_data, key=lambda x: x['pts'])[-n_traces:]
    if shuffle:
        random.shuffle(trace_data)
    return trace_data


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0


"""
Executor Subroutine Operations

TODO: Make these all faster using Numba (if possible)
"""

def find_missing_angles(data_spherical, azimuths, elevations, do_dilate=False):
    """Find obviously missing angles using the expected angles"""
    A = data_spherical[:,1]
    E = data_spherical[:,2]
    azs = list(azimuths.values())  # expected
    els = list(elevations.values())  # expected
    idx_az = np.argsort(azs)  # indices for sorted expected
    idx_el = np.argsort(els)  # indices for sorted expected
    azs_sort = [azimuths[idx] for idx in idx_az]
    els_sort = [elevations[idx] for idx in idx_el]
    idx_az_to_bin_az = {i:idx for i, idx in enumerate(idx_az)}  # map sorted idx to original idx
    idx_el_to_bin_el = {i:idx for i, idx in enumerate(idx_el)}  # map sorted idx to original idx
    A_idxs = (np.digitize(A, azs_sort, right=False) - 1) % len(azimuths)  # indices in sorted list
    E_idxs = np.maximum(0, np.digitize(E, els_sort, right=False) - 1)  # indices in sorted list
    # now we have indices in the matrix for each point
    # **all combinations of (a_bin, e_bin) should be unique**
    AE_M = np.zeros((len(azimuths),len(elevations)), dtype=bool)
    for a_bin, e_bin in zip(A_idxs, E_idxs):  # indices in sorted list
        AE_M[a_bin, e_bin] = True
    if do_dilate:
        dil_structure = np.array([[0, 1, 0],
                                  [1, 1, 1],
                                  [0, 1, 0]], dtype=bool)
        iterations = 1
        AE_M = binary_dilation(AE_M, dil_structure, iterations)
    # map back to original indices
    az_miss, el_miss = np.where(~AE_M)
    az_miss_orig = np.array([azs[idx_az_to_bin_az[i]] for i in az_miss])
    el_miss_orig = np.array([els[idx_el_to_bin_el[i]] for i in el_miss])
    return az_miss_orig, el_miss_orig, AE_M


def get_point_mask_from_trace(data_spherical, trace_spherical, method='qhull'):
    """Get a mask from points fitting a trace"""
    if method == 'rectangle':
        az_limits = (min(trace_spherical[:,1]), max(trace_spherical[:,1]))
        el_limits = (min(trace_spherical[:,2]), max(trace_spherical[:,2]))
        point_mask = (data_spherical[:,1] > az_limits[0]) & \
                     (data_spherical[:,1] < az_limits[1]) & \
                     (data_spherical[:,2] > el_limits[0]) & \
                     (data_spherical[:,2] < el_limits[1])
    elif method == 'qhull':
        point_mask_rect = get_point_mask_from_trace(data_spherical, trace_spherical, method='rectangle')
        qhull = ConvexHull(trace_spherical[:,1:3])
        rect_in_hull = in_hull(data_spherical[point_mask_rect, 1:3], trace_spherical[qhull.vertices,1:3])
        point_mask = point_mask_rect
        point_mask[point_mask] = rect_in_hull
    else:
        raise NotImplementedError(method)
    return point_mask


def get_point_mask_from_object(data_cartesian, box3d):
    """Get a mask of points residing in a bounding box
    
    """
    return maskfilters.filter_points_in_box(data_cartesian, box3d.corners)


def inpaint_mask_as_object_from_trace(data_spherical, mask, trace_spherical, method='knn'):
    """Inpaint masked points to mimic a trace
    
    Performs operation in-place
    """
    if method == 'bspline':
        tck = si.bisplrep(trace_spherical[:,1], trace_spherical[:,2], trace_spherical[:,0])
        tx, ty, c, kx, ky = tck
        u = data_spherical[mask, 1]
        v = data_spherical[mask, 2]
        data_spherical[mask, 0] = si.dfitpack.bispeu(tx, ty, c, kx, ky, u, v)[0]
        data_spherical[mask, 3] = np.random.choice(trace_spherical[:,3], size=sum(mask), replace=True)

    elif method == 'knn':
        knn_model = KNeighborsRegressor(n_neighbors=3)
        knn_model.fit(trace_spherical[:,1:3], trace_spherical[:,0])
        data_spherical[mask, 0] = knn_model.predict(data_spherical[mask,1:3])
        data_spherical[mask, 3] = np.random.choice(trace_spherical[:,3], size=sum(mask), replace=True)

    elif method == 'append':
        data_spherical = np.concatenate((data_spherical, trace_spherical), axis=0)

    else:
        raise NotImplementedError(method)
    # replace intensity values
    return data_spherical


def inpaint_mask_as_background_from_context(data_spherical, mask, method='knn',
        sensor_height=None, continguous_mask=True):
    """Inpaint masked points to estimate a background
    
    Performs operation in-place
    """
    if method == 'geometry':
        assert sensor_height is not None, 'Must pass in sensor height for geometry approach'
        data_spherical[mask, 0] = sensor_height / np.sin(data_spherical[:,2])
    elif method.lower() == 'knn':
        knn_model = KNeighborsRegressor(n_neighbors=5)
        knn_model.fit(data_spherical[~mask,1:3], data_spherical[~mask,0])
        data_spherical[mask, 0] = knn_model.predict(data_spherical[mask,1:3])
    else:
        raise NotImplementedError(method)
    # replace intensity values
    data_spherical[mask, 3] = np.random.choice(data_spherical[~mask,3], size=sum(mask), replace=True)
    return data_spherical


"""
Executor managers
"""

def move_raw_trace(trace, pt_g, T_s2g, range_offset=-0.4, trace_method='cartesian'):
    # -- shift trace points to pts_g specification
    if trace_method == 'cartesian':
        trace_mean = np.mean(trace, axis=0)
        mean_diff = pt_g - trace_mean[:3]
        mean_diff[2] = 0.2 # do not move z by much
        trace[:,:3] += mean_diff
        trace[:, 2] += T_s2g.translation.vector[2]
        # print(np.mean(trace, axis=0))
        # trace[:,:3] = T_s2g @ (mean_diff + trace[:,:3])
    elif trace_method == 'old_method':
        trace_mean = np.mean(trace[:,:3], axis=0)
        rp = np.linalg.norm(pt_g)
        s = (rp+range_offset) / rp
        pt_g = np.array([s*pt_g[0], s*pt_g[1], pt_g[2]])  # HACK for now
        trace[:,:3] = T_s2g.T @ (pt_g + trace[:,:3])
    else:
        raise NotImplementedError
    return trace


class ReplayExecutor(Executor):
    """Execute an attack that buffers the point cloud and replays it"""
    
    def __init__(self, sensor_name, sensor_rate, reverse=False):
        super().__init__(sensor_name, sensor_rate)
        self.buffer = []
        if reverse:
            self.order = 'reverse'
            self.last_send = np.inf
            self.next_send = np.inf
            self.circular = True
        else:
            self.order = 'forward'
            self.last_send = 0
            self.next_send = 0
            self.circular = False

    def _manipulate(self, data, info):
        diagnostics = {}
        if info['mode'] == 'buffer':
            self.buffer.append(data)
        elif info['mode'] == 'repeat':
            if np.isinf(self.last_send):
                self.last_send = len(self.buffer) - 1
            data = self.buffer[self.last_send]
        elif info['mode'] == 'replay':
            if np.isinf(self.next_send):
                if np.isinf(self.last_send):
                    self.last_send = len(self.buffer) - 1
                self.next_send = self.last_send
            data = self.buffer[self.next_send]
            self.last_send = self.next_send
            if (self.order == 'forward') and (self.last_send+1 == len(self.buffer)):
                if self.circular:
                    self.order = 'reverse'
                else:
                    self.last_send = -1
            elif (self.order == 'reverse') and (self.last_send == 0):
                self.order = 'forward'
            if self.order == 'forward':
                self.next_send = self.last_send + 1
            else:
                self.next_send = self.last_send - 1
        else:
            raise NotImplementedError(info['mode'])
        diagnostics['last_send'] = self.last_send
        diagnostics['next_send'] = self.next_send
        diagnostics['order'] = self.order
        return data, diagnostics


class ReverseReplayExecutor(ReplayExecutor):
    def __init__(self, sensor_name, sensor_rate) -> None:
        super().__init__(sensor_name, sensor_rate, reverse=True)


class PointsAsBackgroundExecutor(Executor):
    """Execute an attack that moves points such that they
    appear like background. Assumes points are passed in via
    cartesian coordinates.
    
    This is effectively an "inpainting" task.
    """

    def __init__(self, sensor_name, sensor_rate):
        super().__init__(sensor_name, sensor_rate)

    def _manipulate(self, data_, info):
        """Data passed in as cartesian coordinates"""
        diagnostics = {}
        objects = [info['track_box']]  # in nominal frame
        C = info['C']
        O_s2g = info['P_ground'].as_transform().as_origin()
        data = C.convert(data_, geometry.StandardCoordinates)
        data_spherical = tforms.matrix_cartesian_to_spherical(data)
        for obj in objects:
            if obj is None:
                continue
            obj.change_origin(O_s2g)
            # -- get the mask of points for the object
            point_mask = get_point_mask_from_object(data, obj)
            # -- inpaint as background
            if sum(point_mask) == 0:
                print('No points masked...')            
            else:
                data_spherical = inpaint_mask_as_background_from_context(data_spherical, point_mask)
        
        # -- convert data back to original coordinates
        data = tforms.matrix_spherical_to_cartesian(data_spherical)
        data = geometry.StandardCoordinates.convert(data, C)
        return data, diagnostics


class PointsAsObjectExecutor(Executor):
    """Execute an attack that moves points such that they
    look like an object. Assumes points are passed in via
    cartesian coordinates

    TODO: "use missed" feature is not working!!!!!!!!!!!!!!!!!!!!!
    """
    def __init__(self, sensor_name, sensor_rate, use_missed=False, shape='trace', shuffle=False,
            trace_directory='/data/spencer/KITTI/traces'):
        super().__init__(sensor_name, sensor_rate)
        n_traces = 5
        self.use_missed = use_missed  # NOT SURE WHY THIS IS NOT WORKING
        self.shape = shape
        self.range_offset = -0.4
        self.trace_data = load_traces(trace_directory, n_traces=n_traces, shuffle=shuffle)
        self.trace_stats = {i_trace:{'min':np.min(self.trace_data[i_trace]['trace'], axis=0),
                                     'max':np.max(self.trace_data[i_trace]['trace'], axis=0)}
                            for i_trace in range(n_traces)}
        # self.trace_method = 'cartesian'
        self.idx_trace_add = 1

        # -- expected sensor angles
        firing_time = avstack.messages.get_velodyne_firing_time(sensor_name)
        az_res = sensor_rate * 2*np.pi * firing_time
        n_azimuths = int( 1.//(sensor_rate * firing_time) )
        self.exp_azimuths = {i:az_res*(i-n_azimuths/2) for i in range(0, n_azimuths, 1)}
        self.exp_elevations = {k:v*np.pi/180 for k, v in 
            avstack.messages.get_velodyne_elevation_table(sensor_name).items()}

    def _manipulate(self, data_, info):
        """Data passed in as cartesian coordinates"""
        diagnostics = {}
        pts_ground = info['pts_ground']  # in "C" frame
        P_ground = info['P_ground']  # in nominal frame
        C = info['C']
        T_s2g = P_ground.as_transform()  # in nominal frame

        # -- convert data to standard coordinates
        data = C.convert(data_, geometry.StandardCoordinates)
        data_spherical = tforms.matrix_cartesian_to_spherical(data)
        for i_obj in range(len(pts_ground)):
            # -- convert pts to standard coordinates
            pt_g = pts_ground[i_obj]   # already in standard coordinates

            # -- get the trace points (standard coordinates with nominal origin)
            t_idx = (i_obj + self.idx_trace_add) % len(self.trace_data)
            _, _, trace_raw = self.trace_data[t_idx]['range'], self.trace_data[t_idx]['pts'], self.trace_data[t_idx]['trace']
            trace = move_raw_trace(trace_raw.copy(), pt_g, T_s2g, self.range_offset)  # self.trace_method)
            trace_spherical = tforms.matrix_cartesian_to_spherical(trace)

            # -- temporarily add in missing points in case object would use
            if self.use_missed:
                az_miss, el_miss, AE_M = find_missing_angles(data_spherical,
                    self.exp_azimuths, self.exp_elevations, do_dilate=True)
                # NOTE: range doesn't matter here for now
                pts_miss_spherical = np.array([50*np.ones((len(az_miss),)), az_miss, el_miss,
                    np.random.rand(len(az_miss))]).T
                point_mask_miss = get_point_mask_from_trace(pts_miss_spherical, trace_spherical)
                # -- add ring index if needed (for nuscences dataset)
                if (data.shape[1] == 5) and (pts_miss_spherical.shape[1]==4):
                    ring_index = np.random.choice(data[:,4], pts_miss_spherical.shape[0])[:,None]  # TODO fix this
                    pts_miss_spherical = np.concatenate((pts_miss_spherical, ring_index), axis=1)
                if sum(point_mask_miss) < 100:
                    data_spherical = np.concatenate((data_spherical, pts_miss_spherical[point_mask_miss,:]), axis=0)
                raise RuntimeError('This is not working right now...do not use')

            # -- get point mask for the trace in the point cloud
            point_mask = get_point_mask_from_trace(data_spherical, trace_spherical)
            if sum(point_mask) < 0.10 * data_spherical.shape[0]:
                # -- run interpolator over the masked points
                try:
                    data_spherical = inpaint_mask_as_object_from_trace(data_spherical, point_mask, trace_spherical)    
                except ValueError as e:
                    print('No points masked...')
            else:
                diagnostics['error_codes'] = "too many masked points"
            diagnostics['n_points_masked'] = sum(point_mask)
        # -- convert data back to original coordinates
        data = tforms.matrix_spherical_to_cartesian(data_spherical)
        data = geometry.StandardCoordinates.convert(data, C)

        return data, diagnostics


class FrustumTranslateExecutor(Executor):
    def __init__(self, sensor_name, sensor_rate, use_missed=False, shape='trace', shuffle=False,
        trace_directory='/data/spencer/KITTI/traces'):
        self.remover = PointsAsBackgroundExecutor(sensor_name, sensor_rate)
        self.adder = PointsAsObjectExecutor(sensor_name, sensor_rate, use_missed=use_missed,
            shape=shape, shuffle=shuffle, trace_directory=trace_directory)

    def _manipulate(self, data_, info):
        # -- first remove an object
        data_, diag_remove = self.remover(data_, info)
        # -- then add points in the frustum
        data_, diag_add = self.adder(data_, info)
        # -- merge diagnostics
        diagnostics = {**diag_remove, **diag_add}
        return data_, diagnostics
