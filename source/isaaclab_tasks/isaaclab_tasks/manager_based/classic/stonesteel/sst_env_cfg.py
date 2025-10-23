import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg,RayCasterCfg,patterns,CameraCfg,TiledCameraCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObject, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.actuators import ImplicitActuatorCfg
from tools.test_settings import ISAACLAB_PATH
import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp

MINIMALBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_PATH}/source/stonesteel/Minimalbot.usd",scale=(1.1,1.1,1.1)),
    actuators={
        "FL_actuator": ImplicitActuatorCfg(joint_names_expr=["front_left_joint"], damping=0, stiffness=None),
        "FR_actuator": ImplicitActuatorCfg(joint_names_expr=["front_right_joint"], damping=0, stiffness=None),
        "RL_actuator": ImplicitActuatorCfg(joint_names_expr=["reer_left_joint"], damping=0, stiffness=None),
        "RR_actuator": ImplicitActuatorCfg(joint_names_expr=["reer_right_joint"], damping=0, stiffness=None),
               
               },
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0),
        rot=(0.5,0.5,0.5,0.5),
        # rot=(0.7071,0.0,0.0,0.7071),
        # rot=(1.0,0.0,0.0,0.0),
    ),
)

@configclass
class StoneSteelSceneCfg(InteractiveSceneCfg):
    """
    base scene cfg for sst
    """
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    mush = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/mush",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_PATH}/source/stonesteel/mushroom.usd",scale=(10,10,10)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1,0,0))
    )
    mush2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/mush2",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_PATH}/source/stonesteel/mushroom2.usd",scale=(0.1,0.1,0.1)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1,1,0))
    )
    frame = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/frame",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_PATH}/source/stonesteel/frame2.usd",scale=(0.1,0.1,0.1)),
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(0,0,0),rot=(0.5,0.5,0.5,0.5),)
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0,0,0),rot=(0.7071,0.7071,0.0,0.0),)
    )
    frame2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/frame2",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_PATH}/source/stonesteel/frame.usd",scale=(0.1,0.1,0.1)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.6,0,0),rot=(0.7071,0.7071,0.0,0.0),)
    )
    mushs = RigidObjectCollectionCfg(
        rigid_objects={
            f"mushroom_{i}": RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/mushroom_{i}",
                spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_PATH}/source/stonesteel/mushroom.usd",scale=(10,10,10)),
                # init_state=RigidObjectCfg.InitialStateCfg(pos=(math.cos(i*2*math.pi/20)*math.sin(i*2*math.pi/20),
                #                                                math.sin(i*2*math.pi/20)*2,2))
                init_state=RigidObjectCfg.InitialStateCfg(pos=(torch.randn(1).item()*0.2,
                                                               torch.randn(1).item()*2,1.5))
            )
            for i in range(20)
        }
    )
    # bed = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/bed",
    #     spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_PATH}/source/stonesteel/bed.usd",scale=(0.1,0.1,0.1)),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(1,1,0))
    # )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        # spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    robot: ArticulationCfg = MINIMALBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    body_cam: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Camera",
        width=100,
        height=100,
        offset=TiledCameraCfg.OffsetCfg(pos=(1,0,0),rot=(1,0,0,0),convention='world'),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
    )

@configclass
class ActionsCfg:
    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*_joint"], scale=15.0)

@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
    policy = PolicyCfg()
@configclass
class EventCfg:
    """Configuration for events."""

    randomize_mush1_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={
            "scale_range": {"x": (5, 15), "y": (5, 15), "z": (5, 15)},
            "asset_cfg": SceneEntityCfg("mush"),
        },
    )
    randomize_mush2__scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={
            "scale_range": (0.05, 0.15),
            "asset_cfg": SceneEntityCfg("mush2"),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )
    reset_mush = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        # params={
        #     "pose_range": {}, "velocity_range": {},
        #     "asset_cfg": SceneEntityCfg("mush"),
        # }
    )
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    # cart_out_of_bounds = DoneTerm(
    #     func=mdp.joint_pos_out_of_manual_limit,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    # )
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=1.0)
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
@configclass
class StoneSteelEnvCfg(ManagerBasedRLEnvCfg):
    scene: StoneSteelSceneCfg = StoneSteelSceneCfg(num_envs=3,env_spacing=4.0, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation