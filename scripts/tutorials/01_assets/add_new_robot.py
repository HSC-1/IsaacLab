# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_euler_xyz
from tools.test_settings import ISAACLAB_PATH

JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Jetbot/jetbot.usd"),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)

MINIMALBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_PATH}/source/stonesteel/Minimalbot.usd"),
    actuators={
        "FL_actuator": ImplicitActuatorCfg(joint_names_expr=["front_left_joint"], damping=0.1, stiffness=None),
        "FR_actuator": ImplicitActuatorCfg(joint_names_expr=["front_right_joint"], damping=0.1, stiffness=None),
        "RL_actuator": ImplicitActuatorCfg(joint_names_expr=["reer_left_joint"], damping=0.1, stiffness=None),
        "RR_actuator": ImplicitActuatorCfg(joint_names_expr=["reer_right_joint"], damping=0.1, stiffness=None),
               
               },
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.7071,0.7071,0.0,0.0),
    ),
)
DOFBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Dofbot/dofbot.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
        },
        pos=(0.25, -0.25, 0.0),
    ),
    actuators={
        "front_joints": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "joint3_act": ImplicitActuatorCfg(
            joint_names_expr=["joint3"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "joint4_act": ImplicitActuatorCfg(
            joint_names_expr=["joint4"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
    },
)

# ISAACLAB_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    # Jetbot = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Jetbot")
    Minbot = MINIMALBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Minbot")
    Dofbot = DOFBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Dofbot")

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            # reset counters
            count = 0
            # # reset the scene entities to their initial positions offset by the environment origins
            # root_jetbot_state = scene["Jetbot"].data.default_root_state.clone()
            # root_jetbot_state[:, :3] += scene.env_origins
            # root_dofbot_state = scene["Dofbot"].data.default_root_state.clone()
            # root_dofbot_state[:, :3] += scene.env_origins
            print(scene.env_origins)
            root_min_state = scene["Minbot"].data.default_root_state.clone()
            root_min_state[:,:3] += scene.env_origins+torch.tensor([0.0,0.0,1]).to('cuda')
            print(root_min_state[:,3:7])
            # root_min_state[:,3:7] =quat_from_euler_xyz(*torch.tensor([1.5708,0,0])).to('cuda')
            print(root_min_state[:,3:7])

            print(quat_from_euler_xyz(*torch.tensor([0,1.5708,0])))
            print(quat_from_euler_xyz(*torch.tensor([0,0,1.5708])))

            scene["Minbot"].write_root_pose_to_sim(root_min_state[:,:7])
            # scene["Minbot"].write_root_velocity_to_sim(root_min_state[:, 7:])
            # # copy the default root state to the sim for the jetbot's orientation and velocity
            # scene["Jetbot"].write_root_pose_to_sim(root_jetbot_state[:, :7])
            # scene["Jetbot"].write_root_velocity_to_sim(root_jetbot_state[:, 7:])
            # scene["Dofbot"].write_root_pose_to_sim(root_dofbot_state[:, :7])
            # scene["Dofbot"].write_root_velocity_to_sim(root_dofbot_state[:, 7:])

            # # copy the default joint states to the sim
            # joint_pos, joint_vel = (
            #     scene["Jetbot"].data.default_joint_pos.clone(),
            #     scene["Jetbot"].data.default_joint_vel.clone(),
            # )
            # scene["Jetbot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # joint_pos, joint_vel = (
            #     scene["Dofbot"].data.default_joint_pos.clone(),
            #     scene["Dofbot"].data.default_joint_vel.clone(),
            # )
            # scene["Dofbot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting Jetbot and Dofbot state...")

        # drive around
        if count % 100 < 75:
            # Drive straight by setting equal wheel velocities
            action = torch.Tensor([[10.0, 10.0]])
            action1 = torch.Tensor([[10.0, 10.0,-10,-10]])
        else:
            # Turn by applying different velocities
            action = torch.Tensor([[5.0, -5.0]])
            action1 = torch.Tensor([[5.0, 5.0,-5,-5]])
        # scene["Minbot"].set_joint_velocity_target(action1)
        scene["Minbot"].set_joint_effort_target(action1)
        # scene["Jetbot"].set_joint_velocity_target(action)

        # # wave
        # wave_action = scene["Dofbot"].data.default_joint_pos
        # wave_action[:, 0:4] = 0.25 * np.sin(2 * np.pi * 0.5 * sim_time)
        # scene["Dofbot"].set_joint_position_target(wave_action)

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    print(ISAACLAB_PATH)
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
