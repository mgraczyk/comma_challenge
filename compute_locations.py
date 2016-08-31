#!/usr/bin/env python
import cv2

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

  correspondence.estimate_poses(scene)
  # for agent in scene.agents:
    # for i in range(len(agent.images) - 1):
      # correspondence.compute_fundamental(
          # agent.images[i], agent.images[i + 1], agent.fgmask, agent.fgmask)

def main(image_dir):
  scene = image_data.load_scene(image_dir)
  locations = compute_locations(scene)

if __name__ == "__main__":
  main('./pics')
