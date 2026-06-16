from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from omni.isaac.isaac_sensor import _isaac_sensor

from . import observations as obs
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

def target_distance(env: ManagerBasedRLEnv, target_name: str) -> torch.Tensor:
    """
    i
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[target_name]
    for rigid_objs in env.scene.rigid_object_collections.values():
        default_state = rigid_objs.data.default_object_state.clone()
        state = rigid_objs.data.object_pos_w.clone() - env.scene.env_origins.clone().unsqueeze(1).expand(-1,20,-1)
        # print(f"state shape: {state.shape}")
        target_initial_state = asset.data.default_root_state[:,0:3].clone()
        target_initial_state = target_initial_state.unsqueeze(1).expand(-1,20,-1)
        score = torch.square(state - target_initial_state)/ torch.square(target_initial_state- default_state[:,:,0:3])
    # return 
    # print(torch.sum(score,(1,2)))
    
    
    
    return torch.sum(score,(1,2))