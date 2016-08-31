import cv2
import itertools
import pickle
import numpy as np
import numpy.matlib
import sys
from multiprocessing.dummy import Pool

# Constant identity perspective matrix.
PI = np.concatenate((np.eye(3), np.zeros((3, 1))), 1)

# Algorithm codes here (missing from python):
# http://docs.ros.org/jade/api/rtabmap/html/namespacertflann.html
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_KMEANS = 2
FLANN_INDEX_HIERARCHICAL = 5
FLANN_INDEX_LSH = 6
FLANN_INDEX_AUTOTUNED = 255

def get_or_set_features(image, fgmask):
  features = image.features
  if features is None:
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.03, edgeThreshold=12)

    # find the keypoints and descriptors with SIFT
    image.features = sift.detectAndCompute(image.gray_frame, fgmask)
  return image.features

def compute_matching_features(image1, image2, fgmask1, fgmask2):
  # Get image features.
  kp1, des1 = get_or_set_features(image1, fgmask1)
  kp2, des2 = get_or_set_features(image2, fgmask2)

  # Parameters are documented here
  # https://github.com/opencv/opencv/blob/master/modules/flann/src/miniflann.cpp
  index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=6)
  search_params = dict(checks=75)

  cv2.ocl.setUseOpenCL(False)
  flann = cv2.FlannBasedMatcher(index_params, search_params)
  matches = flann.knnMatch(des1, des2, k=2)
  cv2.ocl.setUseOpenCL(True)

  # store all the good matches as per Lowe's ratio test.
  good = [ m for m, n in matches if m.distance < 0.95*n.distance ]

  return good, kp1, kp2

def _rot_from_vector(to_vec):
  """Get a rotation matrix that sends (1, 0, 0) to the direction of to_vec."""
  # Adapted from
  # http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
  to_vec_dir  = a0_avg / np.linalg.norm(a0_avg)
  v = np.asarray([0, -to_vec_dir[2], to_vec_dir[1]])
  c = to_vec_dir[0]
  s = np.linalg.norm(v)
  vss = np.asarray([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
  P0 = np.eye(3) + vss + np.dot(vss, vss) * (1 - c) / s**2

def _compute_ls_position_estimate(pos_params, results, lens_i):
  # Assume that agent 0, frame 0 has position [0, 0, 0] and build matrices
  # A and D from measurement adjacencies and distances. Then solve for positions
  # X so that ||W(AX - D)|| is minimized. This gives us a least squares estimate
  # of the true positions X.
  num_dists = len(pos_params)
  A_with_p0 = np.matlib.zeros((num_dists, np.sum(lens_i)))
  for row, pos_param in enumerate(pos_params):
    from_index = sum(lens_i[:pos_param[0]]) + pos_param[2]
    to_index = sum(lens_i[:pos_param[1]]) + pos_param[3]

    A_with_p0[row, from_index] = -1
    A_with_p0[row, to_index] = 1

  # Set p0 to (0, 0, 0) as a reference.
  A = A_with_p0[:, 1:]
  D = np.asmatrix([r[1] for r in results])

  # Determine the weighted least squares estimate of positions.
  #   A X = D
  weights = np.sqrt(np.squeeze(np.asarray([w[4] for w in pos_params])))
  Aw = np.einsum('i,ij->ij', weights, A)
  Dw = np.einsum('i,ij->ij', weights, D)
  X, residuals, rank, s = np.linalg.lstsq(Aw, Dw)
  X = np.concatenate(([[0, 0, 0]], X))
  return X.T

def _compute_ls_orientation_estimate(pos_params, results, P0, lens_i):
  # Just as we did for positions, we build a matrix system and solve for the
  # least squares solution. However, now our matrix "elements" are themselves
  # matricies.
  # If we let Pi be the orientation of the ith and Rij be the measured rotation
  # matrix between the two frames, we can write
  #   Rij Pi = I Pj
  #   Rij Pi - I Pj = 0
  # We assume we know P0 so that there are nontrivial solutions. For these terms,
  #   R0j P0 = I Pj
  #   R0j = P0' Pj
  # similarly
  #   Ri0 Pi = I P0
  #   Ri0 Pi = P0
  # Therefore we solve for P to minimize ||W(RP - R0)||, where R is the matrix
  # built of all measurements Rij, R0 are those measurements starting at P0, and
  # W is a weight matrix.
  num_dists = len(pos_params)
  I = np.eye(3)
  R = np.zeros((3, 3, num_dists, np.sum(lens_i) - 1))
  R0 = np.zeros((3, 3, num_dists))
  for row, pair in enumerate(zip(pos_params, results)):
    pos_param, result = pair
    Rij = result[0]
    from_index = sum(lens_i[:pos_param[0]]) + pos_param[2] - 1
    to_index = sum(lens_i[:pos_param[1]]) + pos_param[3] - 1
    weight = pos_param[4]

    if from_index < 0:
      assert(to_index >= 0)
      # R0j = P0' Pj
      R[:, :, row, to_index] = weight * P0.T
      R0[:, :, row] = weight * Rij
    elif to_index < 0:
      # Ri0 Pi = P0
      R[:, :, row, from_index] = weight * Rij
      R0[:, :, row] = weight * P0
    else:
      # Pi gets Rij, Pj gets -I.
      # Multiply by weights here for simplicity.
      R[:, :, row, from_index] = weight * Rij
      R[:, :, row, to_index] = -weight * I
      # Corresponding entry in R0 is the zero matrix.

  # TODO(mgraczyk): The block manipulation here is nasty. Find a way to express
  #                 these manipulations more clearly.

  # Form a block matrix.
  np.set_printoptions(linewidth=200)
  R = np.reshape(np.rollaxis(R, 1, 3), (3*R.shape[2], 3*R.shape[3]), order='F')
  R0 = np.reshape(np.swapaxes(R0, 1, 2), (3*R0.shape[2], 3), order='F')

  # Determine the weighted least squares estimate of orientations.
  P, residuals, rank, s = np.linalg.lstsq(R, R0)
  P = np.concatenate((I, P))
  P = np.swapaxes(np.reshape(P, (3, -1, 3), order='F'), 1, 2)

  return P

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
  """Estimates poses using SIFT features, Flann correspondence, RANSAC essential
     matrix estimation, and least squares position estimation.

     Returns (positions, orientations)
      positions is an 3xN array of N positions in euclidian coordinate.
      orienations is a 3x3xN array of N rotation matrices.
  """
  agents = scene.agents
  len_a = len(agents)
  lens_i = np.asarray([len(a.images) for a in agents])

  # camera_points = np.zeros((3, len(agent.images)))
  # scene_points = np.zeros((3, len(agent.images)))

  # Build up a list of relative positions to compute.
  # Each member is a tuple
  #   (agent_idx_1, agent_idx_2, frame_idx_1, frame_idx_2, weight)
  pos_params = list(itertools.chain(
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
  results = pool.map(do_pose_work, pos_params)
  with open('results.pickle', 'wb') as results_f:
    pickle.dump(results, results_f)

  # with open('results.pickle', 'rb') as results_f:
    # results = pickle.load(results_f)

  positions = _compute_ls_position_estimate(pos_params, results, lens_i)

  # Zero the orientation so that it points in along the average path of travel
  # for agent 0.
  a0_avg = np.mean(positions[:, :lens_i[0]], 1)
  P0 = _rot_from_vector(a0_avg)
  orientations = _compute_ls_orientation_estimate(pos_params, results, P0, lens_i)

  return positions, orientations
