import glob
import os
import sys
import re
from collections import defaultdict
from operator import itemgetter

from datatypes import Image, Agent, Scene

def find_image_paths(image_dir):
  """ Returns a dictionary where keys are agent names and values are ordered
      lists of paths to agent frame images.
  """
  image_pattern = os.path.join(image_dir, '[0-9]*_[0-9]*.jpg')
  image_re = re.compile(".*(\d+)_(\d+)\.jpg")

  image_paths_unordered = defaultdict(dict)
  for image_path in glob.glob(image_pattern):
    match = image_re.match(image_path)
    if not match:
      sys.stderr.write("Warning: Ignoring file {}\n".format(image_path))
      continue

    agent_name = int(match.group(1), 10)
    frame_num = int(match.group(2), 10)
    image_paths_unordered[agent_name][frame_num] = image_path

  return { k: map(itemgetter(1), sorted(v.viewitems(), key=itemgetter(0)))
           for k, v in image_paths_unordered.iteritems() }

def load_agents(image_dir):
  """ Returns a list of Agents corresponding to image sequences in a directory.
  """
  all_image_paths = find_image_paths(image_dir)

  agents = [ Agent(agent_name, (
               Image(image_path) for image_path in image_paths))
             for agent_name, image_paths in all_image_paths.iteritems() ]
  return agents

def load_scene(image_dir):
  agents = load_agents(image_dir)
  return Scene(agents)
