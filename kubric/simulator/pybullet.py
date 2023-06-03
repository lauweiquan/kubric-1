# Copyright 2022 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=function-redefined

from urdfpy import URDF
from scipy.spatial.transform import Rotation
import time
from PIL import Image

import logging
import pathlib
import sys
import tempfile
from typing import Dict, List, Optional, Tuple, Union

import tensorflow as tf
from singledispatchmethod import singledispatchmethod

from kubric import core
from kubric.redirect_io import RedirectStream

# --- hides the "pybullet build time: May 26 2021 18:52:36" message on import
with RedirectStream(stream=sys.stderr):
  import pybullet as pb

logger = logging.getLogger(__name__)


class PyBullet(core.View):
  """Adds physics simulation on top of kb.Scene using PyBullet."""

  def __init__(self, scene: core.Scene, scratch_dir=tempfile.mkdtemp()):
    self.scratch_dir = scratch_dir
    self.physics_client = pb.connect(pb.DIRECT)  # pb.GUI

    # --- Set some parameters to fix the sticky-walls problem; see
    # https://github.com/bulletphysics/bullet3/issues/3094
    pb.setPhysicsEngineParameter(restitutionVelocityThreshold=0.,
                                 warmStartingFactor=0.,
                                 useSplitImpulse=True,
                                 contactSlop=0.,
                                 enableConeFriction=False,
                                 deterministicOverlappingPairs=True)
    # TODO: setTimeStep if scene.step_rate != 240 Hz
    super().__init__(scene, scene_observers={
        "gravity": [lambda change: pb.setGravity(*change.new)],
    })

  def __del__(self):
    try:
      pb.disconnect()
    except Exception:  # pylint: disable=broad-except
      pass  # cleanup code. ignore errors

  @singledispatchmethod
  def add_asset(self, asset: core.Asset) -> Optional[int]:
    raise NotImplementedError(f"Cannot add {asset!r}")

  def remove_asset(self, asset: core.Asset) -> None:
    if self in asset.linked_objects:
      pb.removeBody(asset.linked_objects[self])
    # TODO(klausg): unobserve

  @add_asset.register(core.Camera)
  def _add_object(self, obj: core.Camera) -> None:
    logger.debug("Ignored camera %s", obj)

  @add_asset.register(core.Material)
  def _add_object(self, obj: core.Material) -> None:
    logger.debug("Ignored material %s", obj)

  @add_asset.register(core.Light)
  def _add_object(self, obj: core.Light) -> None:
    logger.debug("Ignored light %s", obj)

  @add_asset.register(core.Cube)
  def _add_object(self, obj: core.Cube) -> Optional[int]:
    collision_idx = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=obj.scale)
    visual_idx = -1
    mass = 0 if obj.static else obj.mass
    # useMaximalCoordinates and contactProcessingThreshold are required to
    # fix the sticky walls issue;
    # see https://github.com/bulletphysics/bullet3/issues/3094
    box_idx = pb.createMultiBody(mass, collision_idx, visual_idx, obj.position,
                                 wxyz2xyzw(obj.quaternion), useMaximalCoordinates=True)
    pb.changeDynamics(box_idx, -1, contactProcessingThreshold=0)
    register_physical_object_setters(obj, box_idx)

    return box_idx

  @add_asset.register(core.Sphere)
  def _add_object(self, obj: core.Sphere) -> Optional[int]:
    radius = obj.scale[0]
    assert radius == obj.scale[1] == obj.scale[2], obj.scale  # only uniform scaling
    collision_idx = pb.createCollisionShape(pb.GEOM_SPHERE, radius=radius)
    visual_idx = -1
    mass = 0 if obj.static else obj.mass
    # useMaximalCoordinates and contactProcessingThreshold are required to
    # fix the sticky walls issue;
    # see https://github.com/bulletphysics/bullet3/issues/3094
    sphere_idx = pb.createMultiBody(mass, collision_idx, visual_idx, obj.position,
                                    wxyz2xyzw(obj.quaternion), useMaximalCoordinates=True)
    pb.changeDynamics(sphere_idx, -1, contactProcessingThreshold=0)
    register_physical_object_setters(obj, sphere_idx)

    return sphere_idx

  @add_asset.register(core.FileBasedObject)
  def _add_object(self, obj: core.FileBasedObject) -> Optional[int]:
    # TODO: support other file-formats
    if obj.simulation_filename is None:
      return None  # if there is no simulation file, then ignore this object
    path = pathlib.Path(obj.simulation_filename).resolve()
    logger.debug("Loading '%s' in the simulator", path)

    if not path.exists():
      raise IOError(f"File '{path}' does not exist.")

    scale = obj.scale[0]
    assert obj.scale[1] == obj.scale[2] == scale, "Pybullet does not support non-uniform scaling"

    # useMaximalCoordinates and contactProcessingThreshold are required to
    # fix the sticky walls issue;
    # see https://github.com/bulletphysics/bullet3/issues/3094
    if path.suffix == ".urdf":
      path_obj = str(pathlib.Path(obj.render_filename).resolve())
      path_obj = path_obj.replace("/","").replace("\\","")
      # print(path_obj)
      robot = URDF.load(str(path))
      # print(obj)
      # print("\n")
      # print(robot.name)
      
      # my code used to create a new index for links
      numbody = [pb.getBodyUniqueId(i) for i in range(pb.getNumBodies())]
      # print(numbody)
      if numbody !=[]:
        urdf_name = (str(pb.getBodyInfo(numbody[-1])[1])[2:-1])
      else:
        urdf_name = ""

      if urdf_name != robot.name:
        obj_idx = pb.loadURDF(str(path), useFixedBase=1,
                            globalScaling=scale,
                            useMaximalCoordinates=False) # <- turn off for proper physics
        # print(obj_idx)
        numbody.append(obj_idx)
        # print(numbody)
        pb.changeDynamics(numbody[-1], -1, contactProcessingThreshold=0)
        register_physical_object_setters(obj, [pb.getBodyUniqueId(i) for i in range(pb.getNumBodies())][-1])
  
      else:
        ## for reading all links including those without obj files
        link_num = -1
        for link in robot.links:
          # print(link.name)
          # print(obj.asset_id)

          link_num += 1
          if link.name == obj.asset_id:
            numbody.append(numbody[-1]+link_num)
            # print(numbody)
            pb.changeDynamics(numbody[-1], -1, contactProcessingThreshold=0)

      # print(obj, numbody[-1])
      # print([pb.getBodyUniqueId(i) for i in range(pb.getNumBodies())][-1])

      obj_idx = numbody[-1]
      # print(obj_idx)
      
    else:
      raise IOError(
          "Unsupported format '{path.suffix}' of file '{path}'")

    # if obj_idx < 0:
    #   raise IOError(f"Failed to load '{path}'") # can remove as URDF file will not be uploaded with all links

    return obj_idx # needs to be a new id while the function: register_physical_object_setters is (filenamebaseobj, urdf id of that obj(body))
                                                                                                  # only the base filenamebaseobj is needed

  def check_overlap(self, obj: core.PhysicalObject) -> bool:
    obj_idx = obj.linked_objects[self]

    body_ids = [pb.getBodyUniqueId(i) for i in range(pb.getNumBodies())]
    for body_id in body_ids:
      if body_id == obj_idx:
        continue
      overlap_points = pb.getClosestPoints(obj_idx, body_id, distance=0)
      if overlap_points:
        return True
    return False

  def get_position_and_rotation(self, obj_idx: int, baselink: bool, linkindex: int):
    if baselink == True:
      pos, quat = pb.getBasePositionAndOrientation(obj_idx)
    else:
      pos = pb.getLinkState(obj_idx, linkindex, computeForwardKinematics = True)[4]
      quat = pb.getLinkState(obj_idx, linkindex, computeForwardKinematics = True)[5]
    return pos, xyzw2wxyz(quat)  # convert quaternion format

  def get_velocities(self, obj_idx: int, baselink: bool, linkindex: int):
    if baselink == True:
      velocity, angular_velocity = pb.getBaseVelocity(obj_idx)
    else:
      velocity = pb.getLinkState(obj_idx, linkindex, computeLinkVelocity = True)[6]
      angular_velocity = pb.getLinkState(obj_idx, linkindex, computeLinkVelocity = True)[7]
    return velocity, angular_velocity

  def save_state(self, path: Union[pathlib.Path, str] = "scene.bullet"):
    """Receives a folder path as input."""
    assert self.scratch_dir is not None
    # first store in a temporary file and then copy, to support remote paths
    pb.saveBullet(str(self.scratch_dir / "scene.bullet"))
    tf.io.gfile.copy(self.scratch_dir / "scene.bullet", path, overwrite=True)

  def run(
      self,
      frame_start: int = 0,
      frame_end: Optional[int] = None
  ) -> Tuple[Dict[core.PhysicalObject, Dict[str, list]], List[dict]]:
    """
    Run the physics simulation.

    The resulting animation is saved directly as keyframes in the assets,
    and also returned (together with the collision events).

    Args:
      frame_start: The first frame from which to start the simulation (inclusive).
        Also the first frame for which keyframes are stored.
      frame_end: The last frame (inclusive) that is simulated (and for which animations
        are computed).

    Returns:
      A dict of all animations and a list of all collision events.
    """

    txt_file_path = pathlib.Path("wq/id_xyz.txt")
    txt_file_path.parent.mkdir(exist_ok=True, parents=True)
    id_xyz_txt = open(txt_file_path, "w")

    frame_end = self.scene.frame_end if frame_end is None else frame_end
    steps_per_frame = self.scene.step_rate // self.scene.frame_rate
    max_step = (frame_end - frame_start + 1) * steps_per_frame

    counter = 0
    zeros = str("0000")
    image_path = "C:/Users/Wei Quan/Desktop/test_img"

    # my own code used to create a new index for links
    obj_idxs = []
    numbody = [pb.getBodyUniqueId(i) for i in range(pb.getNumBodies())]
    for visual in numbody:
      if pb.getNumJoints(visual) == 0:
        if obj_idxs == []:
          obj_idxs.append(0)
        else:
          obj_idxs.append(obj_idxs[-1]+1)
      else:
        for j in range(pb.getNumJoints(visual)):
          if obj_idxs == []:
            obj_idxs.append(j)
          else:
            obj_idxs.append(obj_idxs[-1]+1)
          # print(obj_idxs)

    # obj_idxs = [pb.getBodyUniqueId(i) for i in range(pb.getNumBodies())]
    animation = {obj_id: {"position": [], "quaternion": [], "velocity": [], "angular_velocity": []}
                 for obj_id in obj_idxs}

    collisions = []
    for current_step in range(max_step):
      contact_points = pb.getContactPoints()
      for collision in contact_points:
        (contact_flag,
         body_a, body_b,
         link_a, link_b,
         position_a, position_b, contact_normal_b,
         contact_distance, normal_force,
         lateral_friction1, lateral_friction_dir1,
         lateral_friction2, lateral_friction_dir2) = collision
        del link_a, link_b  # < unused
        del contact_flag, contact_distance, position_a  # < unused
        del lateral_friction1, lateral_friction2  # < unused
        del lateral_friction_dir1, lateral_friction_dir2  # < unused
        if normal_force > 1e-6:
          collisions.append({
              "instances": (self._obj_idx_to_asset(body_b), self._obj_idx_to_asset(body_a)),
              "position": position_b,
              "contact_normal": contact_normal_b,
              "frame": current_step / steps_per_frame,
              "force": normal_force,
          })
      # print(obj_idxs)

      # my own code used to create a new index for links
      if current_step % steps_per_frame == 0:

        obj_idxs = []
        numbody = [pb.getBodyUniqueId(i) for i in range(pb.getNumBodies())]
        for visual in numbody:
          if pb.getNumJoints(visual) == 0:
            linkindex = -1
            baselink = True
            if obj_idxs == []:
              obj_idxs.append(0)
            else:
              obj_idxs.append(obj_idxs[-1]+1)

            position, quaternion = self.get_position_and_rotation(visual, baselink, linkindex)
            velocity, angular_velocity = self.get_velocities(visual, baselink, linkindex)

            animation[obj_idxs[-1]]["position"].append(position)
            animation[obj_idxs[-1]]["quaternion"].append(quaternion)
            animation[obj_idxs[-1]]["velocity"].append(velocity)
            animation[obj_idxs[-1]]["angular_velocity"].append(angular_velocity)

          else:
            for j in range(pb.getNumJoints(visual)):
              linkindex = -1
              linkindex += j
              # print(linkindex)
              # print(j)
              if obj_idxs ==[]:
                obj_idxs.append(j)
                baselink = True
              else:
                if linkindex == -1:
                  baselink = True
                else:
                  baselink = False
                obj_idxs.append(obj_idxs[-1]+1)
              # print(linkindex)
              # print(obj_idxs)
                
              position, quaternion = self.get_position_and_rotation(visual, baselink, linkindex)
              velocity, angular_velocity = self.get_velocities(visual, baselink, linkindex)

              animation[obj_idxs[-1]]["position"].append(position)
              animation[obj_idxs[-1]]["quaternion"].append(quaternion)
              animation[obj_idxs[-1]]["velocity"].append(velocity)
              animation[obj_idxs[-1]]["angular_velocity"].append(angular_velocity)
              # if current_step == 0:
              #   print(position,quaternion)

        # WRITE .TXT FILE FOR THE PRINT
        # id_xyz_txt.write("obj_idx: ")
        # id_xyz_txt.write(str(obj_idxs[2]))
        # id_xyz_txt.write(" Base")
        # id_xyz_txt.write("\n")
        # id_xyz_txt.write(str(pb.getBasePositionAndOrientation(obj_idxs[-1])))
        # id_xyz_txt.write("\n")
        # id_xyz_txt.write("obj_idx: 2")

        # id_xyz_txt.write(" link")
        # id_xyz_txt.write("\n")
        # id_xyz_txt.write(str(pb.getLinkStates(obj_idxs[2], [0])))
        # id_xyz_txt.write("\n\n")

      # print(current_step)
      if current_step == 24*0:
        jointindex = []
        for i in range(pb.getNumJoints(4)):    # <- number is the index number for the urdf you wanna move
          # print(i)
          jointindex.append(i)
        # print(jointindex)

        # initial position
        jointpositions = [-1.737, 0.053, 0.053, -1.842, 0.00, 1.947, 2.263, 0.00, 0.00, 0.17, 0.17, 0.00]
        # set cartesian coordinate for the robot to move
        for j in range(pb.getNumJoints(4)):
          pb.resetJointState(4, j, jointpositions[j])
        # pb.setJointMotorControlArray(2, jointindex, pb.POSITION_CONTROL,
        #                               targetPositions=jointpositions, forces = forcelist)

      if current_step == 24*25:
        # move to pickup cube
        jointpositions = [-1.579, 0.579, 0.000, -2.105, 0.00, 2.684, 2.421, 0.00, 0.00, 0.17, 0.17, 0.00]
        forcelist = [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500]
        # set cartesian coordinate for the robot to move
        pb.setJointMotorControlArray(4, jointindex, pb.POSITION_CONTROL,
                                      targetPositions=jointpositions, forces = forcelist)

      if current_step == 24*35:
        # gripper close
        jointpositions = [-1.579, 0.579, 0.000, -2.105, 0.00, 2.684, 2.421, 0.00, 0.00, 0.08, 0.08, 0.00]
        forcelist = [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500]
        # set cartesian coordinate for the robot to move
        pb.setJointMotorControlArray(4, jointindex, pb.POSITION_CONTROL,
                                      targetPositions=jointpositions, forces = forcelist)

      if current_step == 24*45:
        # move up
        jointpositions = [0, -0.105, 0.000, -2.368, 0.00, 2.368, 2.421, 0.00, 0.00, 0.08, 0.08, 0.00]
        forcelist = [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500]
        # set cartesian coordinate for the robot to move
        pb.setJointMotorControlArray(4, jointindex, pb.POSITION_CONTROL,
                                      targetPositions=jointpositions, forces = forcelist)

      if current_step == 24*55:
        # gripper open
        jointpositions=[0, -0.105, 0.000, -2.368, 0.00, 2.368, 2.421, 0.00, 0.00, 0.17, 0.17 ,0.00]
        forcelist=[500,500,500,500,500,500,500,500,500,500,500,500]
        #set cartesian coordinate for the robot to move
        pb.setJointMotorControlArray(4, jointindex, pb.POSITION_CONTROL,
                                targetPositions=jointpositions, forces=forcelist)

      pb.stepSimulation()
      
      ## for getting frame by frame to generate pybullet video in this file
      # img = pb.getCameraImage(1920, 1080, renderer=pb.ER_BULLET_HARDWARE_OPENGL)
      # rgbBuffer = img[2]
      # rgbim = Image.fromarrray(rgbBuffer)
      # rgbim.save(f"{image_path}/rgb+"+ str(zeros) + str(counter)+ '.png')
      # counter += 1
      # if counter < 9 and counter > 0:
      #   zeros = str("0000")
      # elif counter < 100 and counter > 9:
      #   zeros = str("000")
      # elif counter < 1000 and counter > 99:
      #   zeros = str("00")
      # elif counter < 10000 and counter > 999:
      #   zeros = str("0")

      # time.sleep(0.001)

    animation = {asset: animation[asset.linked_objects[self]] for asset in self.scene.assets
                 if asset.linked_objects.get(self) in obj_idxs}

    # --- Transfer simulation to renderer keyframes
    for obj in animation.keys():
      # print(obj)
      # print("\n")
      for frame_id in range(frame_end - frame_start + 1):
        obj.position = animation[obj]["position"][frame_id]
        obj.quaternion = animation[obj]["quaternion"][frame_id]
        obj.velocity = animation[obj]["velocity"][frame_id]
        obj.angular_velocity = animation[obj]["angular_velocity"][frame_id]
        obj.keyframe_insert("position", frame_id + frame_start)
        obj.keyframe_insert("quaternion", frame_id + frame_start)
        obj.keyframe_insert("velocity", frame_id + frame_start)
        obj.keyframe_insert("angular_velocity", frame_id + frame_start)
        # print(obj.position, obj.quaternion)

    # print("done")
    id_xyz_txt.close()
    return animation, collisions

  def _obj_idx_to_asset(self, idx):
    assets = [asset for asset in self.scene.assets if asset.linked_objects.get(self) == idx]
    if len(assets) == 1:
      return assets[0]
    elif len(assets) == 0:
      return None
    else:
      raise RuntimeError("Multiple assets linked to same pybullet object. That should never happen")


def xyzw2wxyz(xyzw):
  """Convert quaternions from XYZW format to WXYZ."""
  x, y, z, w = xyzw
  return w, x, y, z


def wxyz2xyzw(wxyz):
  """Convert quaternions from WXYZ format to XYZW."""
  w, x, y, z = wxyz
  return x, y, z, w


def register_physical_object_setters(obj: core.PhysicalObject, obj_idx):
  assert isinstance(obj, core.PhysicalObject), f"{obj!r} is not a PhysicalObject"

  obj.observe(setter(obj_idx, set_position), "position")
  obj.observe(setter(obj_idx, set_quaternion), "quaternion")
  # TODO Pybullet does not support rescaling. So we should warn if scale is changed
  obj.observe(setter(obj_idx, set_velocity), "velocity")
  obj.observe(setter(obj_idx, set_angular_velocity), "angular_velocity")
  obj.observe(setter(obj_idx, set_friction), "friction")
  obj.observe(setter(obj_idx, set_restitution), "restitution")
  obj.observe(setter(obj_idx, set_mass), "mass")
  obj.observe(setter(obj_idx, set_static), "static")


def setter(object_idx, func):
  def _callable(change):
    return func(object_idx, change.new, change.owner)
  return _callable


def set_position(object_idx, position, asset):  # pylint: disable=unused-argument
  # reuse existing quaternion
  _, quaternion = pb.getBasePositionAndOrientation(object_idx)
  # resetBasePositionAndOrientation zeroes out velocities, but we wish to conserve them
  velocity, angular_velocity = pb.getBaseVelocity(object_idx)
  pb.resetBasePositionAndOrientation(object_idx, position, quaternion)
  pb.resetBaseVelocity(object_idx, velocity, angular_velocity)


def set_quaternion(object_idx, quaternion, asset):  # pylint: disable=unused-argument
  quaternion = wxyz2xyzw(quaternion)  # convert quaternion format
  # reuse existing position
  position, _ = pb.getBasePositionAndOrientation(object_idx)
  # resetBasePositionAndOrientation zeroes out velocities, but we wish to conserve them
  velocity, angular_velocity = pb.getBaseVelocity(object_idx)
  pb.resetBasePositionAndOrientation(object_idx, position, quaternion)
  pb.resetBaseVelocity(object_idx, velocity, angular_velocity)


def set_velocity(object_idx, velocity, asset):  # pylint: disable=unused-argument
  _, angular_velocity = pb.getBaseVelocity(object_idx)  # reuse existing angular velocity
  pb.resetBaseVelocity(object_idx, velocity, angular_velocity)


def set_angular_velocity(object_idx, angular_velocity, asset):  # pylint: disable=unused-argument
  velocity, _ = pb.getBaseVelocity(object_idx)  # reuse existing velocity
  pb.resetBaseVelocity(object_idx, velocity, angular_velocity)


def set_mass(object_idx, mass: float, asset):
  if mass < 0:
    raise ValueError(f"mass cannot be negative ({mass})")
  if not asset.static:
    pb.changeDynamics(object_idx, -1, mass=mass)


def set_static(object_idx, is_static, asset):
  if is_static:
    pb.changeDynamics(object_idx, -1, mass=0.)
  else:
    pb.changeDynamics(object_idx, -1, mass=asset.mass)


def set_friction(object_idx, friction: float, asset):  # pylint: disable=unused-argument
  if friction < 0:
    raise ValueError("friction cannot be negative ({friction})")
  pb.changeDynamics(object_idx, -1, lateralFriction=friction)


def set_restitution(object_idx, restitution: float, asset):  # pylint: disable=unused-argument
  if restitution < 0:
    raise ValueError("restitution cannot be negative ({restitution})")
  if restitution > 1:
    raise ValueError("restitution should be below 1.0 ({restitution})")
  pb.changeDynamics(object_idx, -1, restitution=restitution)
