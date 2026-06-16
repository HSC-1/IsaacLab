"""
stonesteel env 
"""

import gymnasium as gym

from . import agents

gym.register(
    id="SST_base",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sst_env_cfg:StoneSteelEnvCfg",
        "ppo_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)