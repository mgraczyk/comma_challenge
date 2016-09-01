import numpy as np
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    h = ax.plot(points[0, a, :], points[1, a, :], points[2, a, :],
        linestyle='--', label=scene.agents[a].name)
    ax.scatter(points[0, a, :], points[1, a, :], points[2, a, :])
  ax.legend(loc='best')
  ax.quiver(positions[0, :], positions[1, :], positions[2, :],
            arrows[0, :], arrows[1, :], arrows[2, :],
            length=0.1, arrow_length_ratio=0)
  plt.title('Scene locations')
  plt.show()
