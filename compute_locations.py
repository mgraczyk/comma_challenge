#!/usr/bin/env python
import argparse
import cv2
import json

import image_data
import correspondence
import plotting

def compute_locations(scene):
  """ Computes the locations and orientations of a set of moving objects from
      image sequences.
  """

  positions, orientations = correspondence.estimate_poses(scene)
  return positions, orientations

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

def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Computes locations and orientations for a collection of images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    output_help = """
    Path of json file where results will be written.
    The output json structure is
    {
      "frames": {
        "./pics/0_0.jpg": {
          "orientation": [ [ 1.0, 0.0, 0.0 ],
                           [ 0.0, 1.0, 0.0 ],
                           [ 0.0, 0.0, 1.0 ] ],
          "position": [ 0.0, 0.0, 0.0 ]
        },
        ...
      }
    }
    """

    parser.add_argument('input_dir', metavar='input-dir', nargs='?', default='./pics',
                        help="Path of directory where input images are located.")
    parser.add_argument('-o', '--output-json', dest="output_json", default='./results.json',
                        help=output_help)
    parser.add_argument('--show-plot', dest='show_plot', action='store_true', default=True,
                        help="Show position and orientation plot")
    parser.add_argument('--no-show-plot', dest='show_plot', action='store_false',
                        help="Show position and orientation plot")

    return parser

def main():
  # In order to compute the agent locations, we perform a series of steps to
  # gather information about the scene. We compute the following:
  #
  # 1. Denoised version of the grayscale of each image.
  # 2. Nonmoving image foregrounds, so we know which pixels to ignore.
  # 3. SIFT features for each image.
  # 4. Correspondences between features in certain pairs of images using
  #    Fast Library for Approximate Nearest Neighbors (FLANN), specifically with
  #    the KD Tree indexing algorithm.
  # 5. Essential matrices between pairs of images using RANSAC.
  # 6. Camera pose using cheirality check with cv::recoverPose. See
  #    http://users.cecs.anu.edu.au/~hartley/Papers/cheiral/revision/cheiral.pdf
  # 7. Positions and orientations which minimize the square error in the
  #    estimated camera poses across all pairs measured.
  #
  # Areas for improvement:
  #   - Very little of the many parameters throughout the code have been tuned.
  #   - The camera focal length could be estimated from images or determined
  #     more accurately based on the model of the camera used.
  #   - The entire pose estimation process could b iterated with refined
  #     estimates of the positions of features in object space. Currently the
  #     positions of a large number of features are readily computable, but their
  #     locations are not used in order to keep the algorithm fast and simple.
  args = get_arg_parser().parse_args()

  scene = image_data.load_scene(args.input_dir)
  positions, orientations = compute_locations(scene)

  with open(args.output_json, "w") as results_file:
    write_results(scene, positions, orientations, results_file)
    print("Wrote results to {}".format(args.output_json))

  if args.show_plot:
    plotting.plot_positions_orientations(scene, positions, orientations)

if __name__ == "__main__":
  main()
