import glob
import os
import sys
import re
from collections import defaultdict
from operator import itemgetter

def find_image_paths(image_dir):
  """ Returns a dictionary where keys are object number and values are ordered
      lists of paths to object frame images.
  """
  image_pattern = os.path.join(image_dir, '[0-9]*_[0-9]*.jpg')
  image_re = re.compile(".*(\d+)_(\d+)\.jpg")

  image_paths_unordered = defaultdict(dict)
  for image_path in glob.glob(image_pattern):
    match = image_re.match(image_path)
    if not match:
      sys.stderr.write("Warning: Ignoring file {}\n".format(image_path))
      continue

    object_id = int(match.group(1), 10)
    frame_num = int(match.group(2), 10)
    image_paths_unordered[object_id][frame_num] = image_path

  return { k: map(itemgetter(1), sorted(v.viewitems(), key=itemgetter(0)))
           for k, v in image_paths_unordered.iteritems() }
