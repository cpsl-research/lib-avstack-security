# import os
# import shutil

# import avapi
# import numpy as np

# from avsec import defend


# KITTI_data_dir = os.path.join(os.getcwd(), "data/test_data/object")


# def load_kitti_data(idx_frame):
#     assert (idx_frame == 1) or (idx_frame == 100)

#     # Kitti path things
#     KDM = avapi.kitti.KittiObjectDataset(KITTI_data_dir, "training")

#     # Make truths and detections
#     lidar = KDM.get_lidar(idx_frame)
#     image = KDM.get_image(idx_frame, sensor="main_camera")
#     calib = KDM.get_calibration(idx_frame, sensor="main_camera")
#     objects = KDM.get_objects(idx_frame, sensor="main_camera")

#     return lidar, image, calib, objects


# def make_results(screw=False):
#     """Make some data in a results folder"""
#     KDM = avapi.kitti.KittiObjectDataset(KITTI_data_dir, "training")

#     res_folder = os.path.join(os.getcwd(), "test_results")
#     if os.path.exists(res_folder):
#         shutil.rmtree(res_folder)
#     os.makedirs(res_folder)
#     if os.path.exists(res_folder + "-lpd_defense"):
#         shutil.rmtree(res_folder + "-lpd_defense")

#     # Make some test results
#     for frame in KDM.frames:
#         objects = KDM.get_objects(frame, sensor="main_camera")
#         if screw:
#             for obj in objects:
#                 obj.box3d.t[0] = 1
#                 obj.box3d.t[2] = 1
#         KDM.save_labels(objects, res_folder, frame, add_label_folder=False)
#     return res_folder


# def test_LPD_class_folder_mode():
#     """Test the class instantiation of LPD"""
#     # Make class
#     KDM = avapi.kitti.KittiObjectDataset(KITTI_data_dir, "training")

#     LPDDEF = defend.lidar.LpdDefense(KDM)

#     # Evaluate from saved result
#     # ----all results
#     path_to_results = make_results()
#     new_results, passed, _ = LPDDEF.evaluate_from_folder(
#         path_to_results, idxs=None, write=False
#     )
#     assert len(passed) == 2
#     assert np.all(passed[1])
#     assert np.all(passed[100])

#     # ----all resultsimport os
# import shutil

# import avapi
# import numpy as np

# from avsec import defend


# KITTI_data_dir = os.path.join(os.getcwd(), "data/test_data/object")


# def load_kitti_data(idx_frame):
#     assert (idx_frame == 1) or (idx_frame == 100)

#     # Kitti path things
#     KDM = avapi.kitti.KittiObjectDataset(KITTI_data_dir, "training")

#     # Make truths and detections
#     lidar = KDM.get_lidar(idx_frame)
#     image = KDM.get_image(idx_frame, sensor="main_camera")
#     calib = KDM.get_calibration(idx_frame, sensor="main_camera")
#     objects = KDM.get_objects(idx_frame, sensor="main_camera")

#     return lidar, image, calib, objects


# def make_results(screw=False):
#     """Make some data in a results folder"""
#     KDM = avapi.kitti.KittiObjectDataset(KITTI_data_dir, "training")

#     res_folder = os.path.join(os.getcwd(), "test_results")
#     if os.path.exists(res_folder):
#         shutil.rmtree(res_folder)
#     os.makedirs(res_folder)
#     if os.path.exists(res_folder + "-lpd_defense"):
#         shutil.rmtree(res_folder + "-lpd_defense")

#     # Make some test results
#     for frame in KDM.frames:
#         objects = KDM.get_objects(frame, sensor="main_camera")
#         if screw:
#             for obj in objects:
#                 obj.box3d.t[0] = 1
#                 obj.box3d.t[2] = 1
#         KDM.save_labels(objects, res_folder, frame, add_label_folder=False)
#     return res_folder


# def test_LPD_class_folder_mode():
#     """Test the class instantiation of LPD"""
#     # Make class
#     KDM = avapi.kitti.KittiObjectDataset(KITTI_data_dir, "training")

#     LPDDEF = defend.lidar.LpdDefense(KDM)

#     # Evaluate from saved result
#     # ----all results
#     path_to_results = make_results()
#     new_results, passed, _ = LPDDEF.evaluate_from_folder(
#         path_to_results, idxs=None, write=False
#     )
#     assert len(passed) == 2
#     assert np.all(passed[1])
#     assert np.all(passed[100])

#     # ----all results
#     path_to_results = make_results(screw=True)
#     new_results, passed, _ = LPDDEF.evaluate_from_folder(
#         path_to_results, idxs=None, write=False
#     )
#     assert len(passed) == 2
#     assert not np.any(passed[1])
#     assert not np.any(passed[100])

#     # ----write all
#     path_to_results = make_results(screw=True)
#     new_results, passed, write_path = LPDDEF.evaluate_from_folder(
#         path_to_results, idxs=None, write=True
#     )
#     assert len(passed) == 2
#     assert not np.any(passed[1])
#     assert not np.any(passed[100])
#     res = LPDDEF.get_detection_results_from_folder(write_path)
#     assert len(res) == 2
#     assert len(res[1]) == 0
#     assert len(res[100]) == 0

#     # ----only use index
#     path_to_results = make_results()
#     new_results, passed, _ = LPDDEF.evaluate_from_folder(
#         path_to_results, idxs=1, write=False
#     )
#     assert len(passed) == 1
#     assert np.all(passed[1])

#     # ----only use index and write
#     path_to_results = make_results(screw=True)
#     new_results, passed, write_path = LPDDEF.evaluate_from_folder(
#         path_to_results, idxs=1, write=True
#     )
#     assert len(passed) == 1
#     res = LPDDEF.get_detection_results_from_folder(write_path)
#     assert len(res) == 1
#     assert len(res[1]) == 0


# def test_LPD_func():
#     """Test the LPD algorithm from the Sun USENIX paper"""
#     # Get the data
#     lidar, _, calib, labels = load_kitti_data(1)
#     # _, _, _, labels_1 = load_kitti_data(1)

#     # Test LPD on good detections and "bad" detections
#     assert len(labels) > 0
#     for lab in labels:
#         passed, ratio = defend.lidar.LPD(lidar, lab, calib)
#         assert passed == True

#     for lab1 in labels:
#         # Force move location
#         lab1.box3d.t[0] = 5
#         lab1.box3d.t[2] = 3
#         passed, ratio = defend.lidar.LPD(lidar, lab1, calib)
#         assert not passed


# def test_compare_3d_detections():
#     """Test the comparison of independent full 3D detections"""
#     # Get the data
#     lidar, _, calib, labels = load_kitti_data(100)
#     det_cam = labels[0:2]
#     det_lid = labels[1:3]

#     # Run comparison
#     consistent, inconsistent = defend.lidar.compare_3d_detections_image_lidar(
#         detections_image=det_cam,
#         detections_lidar=det_lid,
#         metric="distance",
#         radius=3,
#     )

#     # Run comparison
#     assert len(consistent) == 1
#     # assert(len(inconsistent[0]) == 1)
#     # assert(len(inconsistent[1]) == 1)


# def test_compare_2d_3d_image_bbox():
#     """Test the defense idea of using camera's 2D bbox and projecting 3D to 2D"""

#     # Get the data
#     lidar, _, calib, objects = load_kitti_data(100)

#     # Compare ground truths
#     for obj in objects:
#         box2d = obj.box3d.project_to_2d_bbox(calib)
#         (
#             inter,
#             inter_ratio,
#             union,
#             union_ratio,
#             iou,
#             area_ratio,
#         ) = defend.lidar.compare_2d_3d_image_bbox(
#             obj.box3d, box2d, calib, only_in_image=True
#         )
#         # Tests
#         assert inter_ratio > 0.9
#         assert union_ratio > 0.9
#         assert iou > 0.7
#         assert (area_ratio > 0.7) and (area_ratio < 1.3)

#     # Compare no overlaps
#     for obj in objects:
#         obj.box3d.t[0] += 5
#         obj.box3d.t[2] += 5
#         box2d = obj.box3d.project_to_2d_bbox(calib)
#         (
#             inter,
#             inter_ratio,
#             union,
#             union_ratio,
#             iou,
#             area_ratio,
#         ) = defend.lidar.compare_2d_3d_image_bbox(
#             obj.box3d, box2d, calib, only_in_image=True
#         )
#         # Tests
#         assert inter_ratio == 0
#         assert union_ratio > 0.9
#         assert iou == 0
#         assert area_ratio < 1.5 LPDDEF.evaluate_from_folder(
#         path_to_results, idxs=None, write=True
#     )
#     assert len(passed) == 2
#     assert not np.any(passed[1])
#     assert not np.any(passed[100])
#     res = LPDDEF.get_detection_results_from_folder(write_path)
#     assert len(res) == 2
#     assert len(res[1]) == 0
#     assert len(res[100]) == 0

#     # ----only use index
#     path_to_results = make_results()
#     new_results, passed, _ = LPDDEF.evaluate_from_folder(
#         path_to_results, idxs=1, write=False
#     )
#     assert len(passed) == 1
#     assert np.all(passed[1])

#     # ----only use index and write
#     path_to_results = make_results(screw=True)
#     new_results, passed, write_path = LPDDEF.evaluate_from_folder(
#         path_to_results, idxs=1, write=True
#     )
#     assert len(passed) == 1
#     res = LPDDEF.get_detection_results_from_folder(write_path)
#     assert len(res) == 1
#     assert len(res[1]) == 0


# def test_LPD_func():
#     """Test the LPD algorithm from the Sun USENIX paper"""
#     # Get the data
#     lidar, _, calib, labels = load_kitti_data(1)
#     # _, _, _, labels_1 = load_kitti_data(1)

#     # Test LPD on good detections and "bad" detections
#     assert len(labels) > 0
#     for lab in labels:
#         passed, ratio = defend.lidar.LPD(lidar, lab, calib)
#         assert passed == True

#     for lab1 in labels:
#         # Force move location
#         lab1.box3d.t[0] = 5
#         lab1.box3d.t[2] = 3
#         passed, ratio = defend.lidar.LPD(lidar, lab1, calib)
#         assert not passed


# def test_compare_3d_detections():
#     """Test the comparison of independent full 3D detections"""
#     # Get the data
#     lidar, _, calib, labels = load_kitti_data(100)
#     det_cam = labels[0:2]
#     det_lid = labels[1:3]

#     # Run comparison
#     consistent, inconsistent = defend.lidar.compare_3d_detections_image_lidar(
#         detections_image=det_cam,
#         detections_lidar=det_lid,
#         metric="distance",
#         radius=3,
#     )

#     # Run comparison
#     assert len(consistent) == 1
#     # assert(len(inconsistent[0]) == 1)
#     # assert(len(inconsistent[1]) == 1)


# def test_compare_2d_3d_image_bbox():
#     """Test the defense idea of using camera's 2D bbox and projecting 3D to 2D"""

#     # Get the data
#     lidar, _, calib, objects = load_kitti_data(100)

#     # Compare ground truths
#     for obj in objects:
#         box2d = obj.box3d.project_to_2d_bbox(calib)
#         (
#             inter,
#             inter_ratio,
#             union,
#             union_ratio,
#             iou,
#             area_ratio,
#         ) = defend.lidar.compare_2d_3d_image_bbox(
#             obj.box3d, box2d, calib, only_in_image=True
#         )
#         # Tests
#         assert inter_ratio > 0.9
#         assert union_ratio > 0.9
#         assert iou > 0.7
#         assert (area_ratio > 0.7) and (area_ratio < 1.3)

#     # Compare no overlaps
#     for obj in objects:
#         obj.box3d.t[0] += 5
#         obj.box3d.t[2] += 5
#         box2d = obj.box3d.project_to_2d_bbox(calib)
#         (
#             inter,
#             inter_ratio,
#             union,
#             union_ratio,
#             iou,
#             area_ratio,
#         ) = defend.lidar.compare_2d_3d_image_bbox(
#             obj.box3d, box2d, calib, only_in_image=True
#         )
#         # Tests
#         assert inter_ratio == 0
#         assert union_ratio > 0.9
#         assert iou == 0
#         assert area_ratio < 1.5