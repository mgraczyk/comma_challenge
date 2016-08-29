#!/usr/bin/env python
import cv2

import image_data
import compute_foreground

def compute_locations(image_paths):
  """ Computes the locations and orientations of a set of moving objects from
      image sequences.

      image_paths consists of a dict with object id's as keys and lists of image
      paths as values. The images are ordered temporally.
  """

  # In order to compute the location, we perform a series of steps to gather
  # information about the scene. We compute the following:
  # 1. Nonmoving image foregrounds, so we know which pixels to ignore.
  # 2. Orientation of the camera relative to the object's path of travel.
  # 3. Velocity of the camera.
  # 4.
  fgmask = compute_foreground.compute(image_paths.values()[2])

def main(image_dir):
  image_paths = image_data.find_image_paths(image_dir)
  locations = compute_locations(image_paths)

if __name__ == "__main__":
  main('./pics')
