#!/usr/bin/env python
import cv2
import numpy as np
import json
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import image_data
import correspondence

def compute_locations(scene):
  """ Computes the locations and orientations of a set of moving objects from
      image sequences.
  """

  positions, orientations = correspondence.estimate_poses(scene)
  return positions, orientations

def plot_positions_orientations(scene, positions, orientations):
  # Assumes each agent has the same number of frames
  num_frames = positions.shape[1]
  len_a = len(scene.agents)
  points = np.reshape(positions, (3, len_a, -1))
  arrows = np.zeros((3, positions.shape[1]))

  unit_arrow = np.array([1, 0, 0]).T
  for i in range(num_frames):
    R = orientations[:, :, i]
    arrows[:, i] = np.dot(R, unit_arrow)

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  for a in range(len_a):
    ax.plot(points[0, a, :], points[1, a, :], points[2, a, :], linestyle='--')
    ax.scatter(points[0, a, :], points[1, a, :], points[2, a, :])
  ax.quiver(positions[0, :], positions[1, :], positions[2, :],
            arrows[0, :], arrows[1, :], arrows[2, :],
            length=0.1, arrow_length_ratio=0)

  plt.show()

def write_results(scene, positions, orientations, results_file):
  image_paths = list(image.image_path
      for agent in scene.agents for image in agent.images)
  results = {
      "frames": {
        image_paths[i]: {
          "position": positions[:, i].tolist(),
          "orientation": orientations[:, :, i].tolist()
        } for i in range(len(image_paths))
      }
  }

  json.dump(results, results_file)

def main(image_dir):
  # In order to compute the agent locations, we perform a series of steps to
  # gather information about the scene. We compute the following:
  # 1. Denoised version of the grayscale of each image.
  # 2. Nonmoving image foregrounds, so we know which pixels to ignore.
  # 3. SIFT features for each image.
  # 4. Correspondences between features in certain pairs of images using
  #    Fast Library for Approximate Nearest Neighbors (FLANN), specifically with
  #    the KD Tree indexing algorithm.
  # 4. Relative locations of each for frame for a single car at a time.
  # 5. Relative locations of each car in each.

  scene = image_data.load_scene(image_dir)
  positions, orientations = compute_locations(scene)

  # TODO(mgraczyk): Dump results to json.
  results_path = "results.json"
  with open(results_path, "w") as results_file:
    write_results(scene, positions, orientations, results_file)
    print("Wrote results to {}".format(results_path))

  plot_positions_orientations(scene, positions, orientations)

if __name__ == "__main__":
  main('./pics')
