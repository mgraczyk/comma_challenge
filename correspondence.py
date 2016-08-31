import cv2
import itertools
import pickle
import numpy as np
import numpy.matlib
import sys
from multiprocessing.dummy import Pool

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

  E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, cv2.RANSAC, 0.95, 3)
  _, R, t, __ = cv2.recoverPose(E, src_pts, dst_pts, K, R, t, mask)

  # TODO(mgraczyk): Compute and use point in scene positions.
  # P = np.concatenate((R, t), 1)
  # points_4d = cv2.triangulatePoints(PI, P, src_pts, dst_pts)
  # points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)
  points_3d = None

  return points_3d, R, t

def estimate_poses(scene):
  agents = scene.agents
  len_a = len(agents)
  lens_i = np.asarray([len(a.images) for a in agents])

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
        ((a, a, i1, i2, 0.2) for i1, i2 in zip(range(lens_i[a] - 2), range(2, lens_i[a])))
        for a in range(len_a)),
      # Between agents at corresponding frames.
      itertools.chain.from_iterable(
        ((a1, a2, ii, ii, 0.5) for ii in range(min(lens_i[a1], lens_i[a2])))
        for a1 in range(len_a) for a2 in range(a1+1, len_a)),
      # Between agents at adjacent frames.
      itertools.chain.from_iterable(
        ((a1, a2, i1, i2, 0.3) for i1, i2 in zip(range(lens_i[a1] - 1), range(1, lens_i[a2])))
        for a1 in range(len_a) for a2 in range(a1+1, len_a)),
      itertools.chain.from_iterable(
        ((a1, a2, i1, i2, 0.3) for i1, i2 in zip(range(1, lens_i[a1]), range(lens_i[a2] - 1)))
        for a1 in range(len_a) for a2 in range(a1+1, len_a))
      ))

  def do_pose_work(args):
    """Find relative positions and rotations for two frames."""
    a1, a2, i1, i2, _ = args
    image1 = agents[a1].images[i1]
    fgmask1 = agents[a1].fgmask
    image2 = agents[a2].images[i2]
    fgmask2 = agents[a2].fgmask
    t = np.atleast_2d(np.asarray([-0.2, -0.2, -0.2]))

    points_3d, R, t = estimate_pose(image1, image2, fgmask1, fgmask2, agents[a2].K, None, t)
    return R, np.squeeze(t)

  pool = Pool()

  # Array of (R, t)
  results = pool.map(do_pose_work, work_indices)

  with open('results.pickle', 'wb') as results_f:
    pickle.dump(results, results_f)
  # with open('results.pickle', 'rb') as results_f:
    # results = pickle.load(results_f)

  num_dists = len(work_indices)
  A_with_p0 = np.matlib.zeros((num_dists, np.sum(lens_i)))
  for row, indices in enumerate(work_indices):
    from_index = sum(lens_i[:indices[0]]) + indices[2]
    to_index = sum(lens_i[:indices[1]]) + indices[3]

    A_with_p0[row, from_index] = -1
    A_with_p0[row, to_index] = 1

  # Set p0 to (0, 0, 0) as a reference.
  A = A_with_p0[:, 1:]
  D = np.asmatrix([r[1] for r in results])

  # Determine the weighted least squares estimate of positions.
  #   A X = D
  weights = np.sqrt(np.squeeze(np.asarray([w[4] for w in work_indices])))
  Aw = np.einsum('i,ij->ij', weights, A)
  Dw = np.einsum('i,ij->ij', weights, D)
  X, residuals, rank, s = np.linalg.lstsq(Aw, Dw)
  X = np.concatenate(([[0, 0, 0]], X))

  # TODO(mgraczyk): Estimate the relative camera orientations.
  return X

# def compute_K_E(agent):
  # K = np.eye(3)

  # for i in range(len(agent.images) - 1):
    # image1 = agent.images[i]
    # image2 = agent.images[i + 1]

    # good, kp1, kp2 = compute_matching_features(image1, image2, agent.fgmask, agent.fgmask)
    # src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    # dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    # E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, cv2.RANSAC, 0.9, 1)
    # cv2.calibrateCameraExtended(object_points, src_points,
    # print(E)
