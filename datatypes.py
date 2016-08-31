import cv2
import numpy as np
import compute_foreground
import denoise

class Image(object):
  """An image is a 2D view of a scene captured by an agent at a specific
     position and orientation.
  """
  def __init__(self, image_path):
    self._image_path = image_path
    self._frame = None
    self._gray_frame = None

    self._features = None

  @property
  def image_path(self):
    return self._image_path

  @property
  def frame(self):
    if self._frame is None:
      frame = cv2.imread(self._image_path)
      self._frame = frame
    return self._frame

  @property
  def gray_frame(self):
    if self._gray_frame is None:
      gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
      self._gray_frame = self._frame = denoise.denoise_gray(gray_frame)

    return self._gray_frame

  @property
  def features(self):
    """Returns pair (keypoints, descriptors)"""
    return self._features

  @features.setter
  def features(self, val):
    self._features = val


class Agent(object):
  """An agent is an object in a scene which captures images from different
     locations with different orientations. The camera used by the agent is
     assumed to remain constant (focal length, etc). The stationary foreground
     of the agent is also assumed to remain constant (dashboard, rear view
     mirror, etc).
  """
  def __init__(self, name, images):
    # Convert to list so we don't retain an iterator.
    images = list(images)
    if not len(images):
      raise ValueError("images cannot be empty")

    self._name = name
    self._images = images
    self._fgmask = None

    # Assumes that all images have the same height and width.
    h, w = self._images[0].frame.shape[:2]
    self._K = np.eye(3)
    self._K[0, 0] = 0.7 * w
    self._K[1, 1] = self._K[0, 0]

  @property
  def images(self):
    return self._images

  @property
  def fgmask(self):
    if self._fgmask is None:
      self._fgmask = compute_foreground.compute(self.images)

  @property
  def K(self):
    return self._K

  @K.setter
  def K(self, val):
    self._K = val

class Scene(object):
  """A scene is a collection of agents and other objects located in 3D space,
     along with a set of 2D views of the scene captured by the agents.
  """
  def __init__(self, agents):
    agents = list(agents)
    if not len(agents):
      raise ValueError("agents cannot be empty")

    self._agents = agents

  @property
  def agents(self):
    return self._agents
