import cv2
import itertools
import pickle
import numpy as np
import sys
from multiprocessing.dummy import Pool
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constant identity perspective matrix.
PI = np.concatenate((np.eye(3), np.zeros((3, 1))), 1)

def get_or_set_features(image, fgmask):
  features = image.features
  if features is None:
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    image.features = sift.detectAndCompute(image.gray_frame, fgmask)
  return image.features

def compute_matching_features(image1, image2, fgmask1, fgmask2):
  # Get image features.
  kp1, des1 = get_or_set_features(image1, fgmask1)
  kp2, des2 = get_or_set_features(image2, fgmask2)

  FLANN_INDEX_KDTREE = 0
  index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
  search_params = dict(checks=50)

  cv2.ocl.setUseOpenCL(False)
  flann = cv2.FlannBasedMatcher(index_params, search_params)
  matches = flann.knnMatch(des1, des2, k=2)
  cv2.ocl.setUseOpenCL(True)

  # store all the good matches as per Lowe's ratio test.
  good = [ m for m, n in matches if m.distance < 0.95*n.distance ]

  return good, kp1, kp2

def estimate_pose(image1, image2, fgmask1, fgmask2, K, R, t):
  good, kp1, kp2 = compute_matching_features(image1, image2, fgmask1, fgmask2)
  src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
  dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

  E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, cv2.RANSAC, 0.95, 2)
  _, R, t, __ = cv2.recoverPose(E, src_pts, dst_pts, K, R, t, mask)
  # P = np.concatenate((R, t), 1)
  # points_4d = cv2.triangulatePoints(PI, P, src_pts, dst_pts)
  # points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)
  points_3d = None

  return points_3d, R, t

def estimate_poses(scene):
  agents = scene.agents
  len_a = len(agents)
  lens_i = [len(a.images) for a in agents]

  # Approximate focal length. Tuned based on training input images.
  R = None
  t = None

  # camera_points = np.zeros((3, len(agent.images)))
  # scene_points = np.zeros((3, len(agent.images)))

  # Build up a list of relative positions to compute.
  # Each member is a tuple
  #   (agent_idx_1, agent_idx_2, frame_idx_1, frame_idx_2, weight)
  work_indices = list(itertools.chain(
      # Adjacent frames for each agent.
      itertools.chain.from_iterable(
        ((a, a, i1, i2, 1) for i1, i2 in zip(range(lens_i[a] - 1), range(1, lens_i[a])))
        for a in range(len_a)),
      # Two frames away for each agent.
      itertools.chain.from_iterable(
        ((a, a, i1, i2, 0.5) for i1, i2 in zip(range(lens_i[a] - 2), range(2, lens_i[a])))
        for a in range(len_a)),
      # Agents at corresponding frames.
      itertools.chain.from_iterable(
        ((a1, a2, ii, ii, 0.5) for ii in range(min(lens_i[a1], lens_i[a2])))
        for a1 in range(len_a) for a2 in range(a1+1, len_a))
      ))


  def do_pose_work(args):
    """Find relative positions and rotations for two frames."""
    a1, a2, i1, i2, _ = args
    image1 = agents[a1].images[i1]
    fgmask1 = agents[a1].fgmask
    image2 = agents[a2].images[i2]
    fgmask2 = agents[a2].fgmask

    points_3d, R, t = estimate_pose(image1, image2, fgmask1, fgmask2, agents[a2].K, None, None)
    return (R, t)

  # Array of (R, t)
  pool = Pool()
  results = pool.map(do_pose_work, work_indices)

  print(results)
  print(len(results))
  # tavg = np.atleast_2d(camera_points[:, -1] - camera_points[:, 0])

  # fig = plt.figure()
  # ax = fig.add_subplot(111, projection='3d')
  # ax.plot(camera_points[0, :], camera_points[1, :], camera_points[2, :])
  # ax.scatter(camera_points[0, :], camera_points[1, :], camera_points[2, :])
  # plt.show()

# def compute_K_E(agent):
  # K = np.eye(3)

  # for i in range(len(agent.images) - 1):
    # image1 = agent.images[i]
    # image2 = agent.images[i + 1]

    # good, kp1, kp2 = compute_matching_features(image1, image2, agent.fgmask, agent.fgmask)
    # src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    # dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    # E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, cv2.RANSAC, 0.9, 1)
    # # cv2.calibrateCameraExtended(object_points, src_points,
    # # E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, cv2.RANSAC, 0.9, 1)
    # print(E)
