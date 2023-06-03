# Copyright 2022 The Kubric Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import numpy as np
import kubric as kb
from scipy.spatial.transform import Rotation
import numpy as np
from urdfpy import URDF
from pathlib import Path
from kubric.renderer.blender import Blender as KubricBlender
from kubric.simulator.pybullet import PyBullet as KubricSimulator

logging.basicConfig(filename="error.log", filemode="w", level=logging.DEBUG)  # < CRITICAL, ERROR, WARNING, INFO, DEBUG

urdf_file = Path("pybullet_data/franka_panda/panda.urdf")

def main(urdf_file: str):
  # --- create scene and attach a renderer and simulator
  scene = kb.Scene(resolution=(1920, 1080))
  scene.frame_end = 200   # < numbers of frames to render
  scene.frame_rate = 24  # < rendering framerate
  scene.step_rate = 240  # < simulation framerate
  renderer = KubricBlender(scene)
  simulator = KubricSimulator(scene)

  # --- populate the scene with objects, lights, cameras
  # scene += kb.Cube(name="floor", scale=(3, 3, 0.1), position=(0, 0, -0.1),
  #                  static=True)
  scene += kb.DirectionalLight(name="sun", position=(-1, -0.5, 3),
                              look_at=(0, 0, 0), intensity=1.5)
  scene.camera = kb.PerspectiveCamera(name="camera", position=(2, -0.5, 4),
                                      look_at=(0, 0, 0))

  # --- setting up variable for obj
  scale = 3
  position = (0,1,2.6)
  euler = (0,0,0)
  parent_file = Path(urdf_file).parent
  
  # wall = kb.FileBasedObject(
  #   asset_id="wall",
  #   render_filename="pybullet_data/wall/wall.obj",
  #   simulation_filename="pybullet_data/wall/wall.urdf",
  #   scale=7, position=(0,0,0), static = True
  # )
  # scene += wall
  
  table1 = kb.FileBasedObject(
    asset_id="table",
    render_filename="pybullet_data/wooden_table/wooden_table.obj",
    simulation_filename="pybullet_data/wooden_table/wooden_table.urdf",
    scale=3, position=(0,0,0.1), euler=(1.5708,0,1.5708), static = True
  )
  scene += table1
  
  cube = kb.FileBasedObject(
    asset_id="cube",
    render_filename="pybullet_data/fabric/cube.obj",
    simulation_filename="pybullet_data/fabric/cube.urdf",
    scale=0.2, position=(0,-0.82,2.7), static = False
  )
  scene += cube

  table2 = kb.FileBasedObject(
    asset_id="table",
    render_filename="pybullet_data/wooden_table/wooden_table.obj",
    simulation_filename="pybullet_data/wooden_table/wooden_table.urdf",
    scale=3, position=(2.5,0,0.1), euler=(1.5708,0,1.5708), static = True
  )
  scene += table2

  tray = kb.FileBasedObject(
    asset_id="tray",
    render_filename="pybullet_data/tray/tray_textured.obj",
    simulation_filename="pybullet_data/tray/tray.urdf",
    scale=2, position=(2,1,2.5), euler=(0,0,1.5708), static = True
  )
  scene += tray
  
  robot =  URDF.load(str(urdf_file))
  
  for link in robot.links:
    # print(link.name)
    for visual in link.visuals:
      filepath = parent_file.joinpath(visual.geometry.mesh.filename)

      ## not really needed cos only the base value is used
#       pos_list = []
#       euler_list = []
#       fk = robot.link_fk() # only can read initial states
#       # print(fk[link])
#       for i, add_pos in enumerate(position):
#         add_pos = fk[link][i][3]
#         pos_list.append(add_pos)
#       position = tuple(pos_list)
#       # slice to turn 4x4 transformation matrix to rotational matrix for euler conversion
#       rot_mat = fk[link][:3, :3]
#       # transform matrix to euler angles
#       r = Rotation.from_matrix(rot_mat)
#       joint_euler = r.as_euler("xyz")
#       for i, add_euler in enumerate(euler):
#         add_euler = joint_euler[i]
#         euler_list.append(add_euler)
#       euler = tuple(euler_list)
#       # print(position)
#       # print(r.as_quat())

      static = True
      
      obj = kb.FileBasedObject(
        asset_id=link.name,
        render_filename=f"{filepath}",
        simulation_filename=f"{urdf_file}",
        scale=scale, position=position, euler=euler, static=static
      )
      scene += obj

  
  # --- executes the simulation (and store keyframes)
  simulator.run()

  # --- renders the output
  renderer.save_state("wq/0simulator.blend")  
  # frames_dict = renderer.render()
  # kb.write_image_dict(frames_dict, "wq")

if __name__ == "__main__":
  main(urdf_file)
