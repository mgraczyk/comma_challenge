#!/usr/bin/env python
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import image_data
import correspondence

def compute_locations(scene):
  """ Computes the locations and orientations of a set of moving objects from
      image sequences.
  """

  # In order to compute the agent locations, we perform a series of steps to
  # gather information about the scene. We compute the following:
  # 1. Nonmoving image foregrounds, so we know which pixels to ignore.
  # 2. Orientation of the camera relative to the object's path of travel.
  # 3. Velocity of the camera.
  # 4. Relative locations of each for frame for a single car at a time.
  # 5. Relative locations of each car in each.

  X = correspondence.estimate_poses(scene)

  len_a = len(scene.agents)
  points = np.reshape(X.T, (3, len_a, -1))
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  for a in range(len_a):
    ax.plot(points[0, a, :], points[1, a, :], points[2, a, :])
    ax.scatter(points[0, a, :], points[1, a, :], points[2, a, :])
  plt.show()

def main(image_dir):
  scene = image_data.load_scene(image_dir)
  locations = compute_locations(scene)

if __name__ == "__main__":
  main('./pics')
