from isaaclab.app import AppLauncher
import argparse
import os
import time
from util import RunningMeanStd
# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to spawn.")
# parser.add_argument("--enable_cameras", type=bool, default=True, help="Number of environments to spawn.")
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Save the data from camera at index specified by ``--camera_id``.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import rl_games.algos_torch.layers
from rl_games.algos_torch import torch_ext
from skrl import config
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from rl_games.common import vecenv
import rl_games.common.divergence as divergence
from rl_games.common.extensions.distributions import CategoricalMasked
from torch.distributions import Categorical
from rl_games.algos_torch.sac_helper import SquashedNormal
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs
from rl_games.algos_torch.moving_mean_std import GeneralizedMovingStats
from rl_games.common.experience import ExperienceBuffer
from rl_games.algos_torch.models import BaseModel, BaseModelNetwork
import gymnasium as gym
import math
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from dic_model import DiC_S
from rl_games.algos_torch.torch_ext import numpy_to_torch_dtype_dict
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
#vec_env 가져오기
#vec_env에서 바로 넣나 dataset으로 만들어 넣나 차이가 있나 없어보임
#experience_buffer로 horizon == play_step()
#cnn mlp 동시 안되는 상태임
activation_dict = {
    'relu':  nn.ReLU()
    ,'tanh':  nn.Tanh()
    ,'sigmoid':  nn.Sigmoid()
    ,'elu':   nn.ELU()
    ,'selu':  nn.SELU()
    ,'swish':  nn.SiLU()
    ,'gelu': nn.GELU()
    ,'softplus':  nn.Softplus()
    ,'None':  nn.Identity()
}


       
class Network(nn.Module):
    def __init__(self, env_con, net_config:dict):
        """
            env_config에서 받아서 쓸것이 자동화를 위해 obs_shape, action_shape 더있나?
        """
        nn.Module.__init__(self)
        model_config ={}
        self.actions_num = env_con['action_space'].shape[0]
        self.input_shape = env_con['observation_space'].shape
       
        # self.actions_num = env.action_space.shape[1]
        # self.input_shape = env.observation_space['policy'].shape
       
        # model_config.update(net_config)
        # breakpoint()

        config = net_config
        
        # self.actions_num  = config.get('action_shape', 5)
        # self.input_shape = config.get('obs_shape', [1])
        self.mlp_units = config["mlp"].get("units",[32,32])
        self.mlp_act = config.get("mlp_activation","relu")
        value_size = config.get("value_size",1)
        value_activation = config.get("value_activation","None")
        mu_activation = config.get("mu_activation","None")
        sigma_activation = config.get("sigma_activation","None")
        self.discrete = config.get('discrete',False)
        self.mlp = nn.Sequential()
        self.cnn = nn.Sequential()
        # print(self.input_shape)
        self.dum_input = torch.randn((1,*self.input_shape))
        # dum_images = torch.randn(64, 3, 64, 64) # todo: 임시로 해놓은것
        # breakpoint()
        if 'cnn' in config:
            self.dum_input = self.dum_input.permute((0,3,1,2))
            # for obs shape 4
            # input expected shape (B, W, H, C)
            # convert to (B, C, W, H)
            self.has_cnn = True
            self.cnn_cfg :dict = config.get('cnn',{})
            self.permute_input = self.cnn_cfg.get('permute_input', True)
            if self.permute_input:
                self.input_shape = torch_ext.shape_whc_to_cwh(self.dum_input.shape)
                print(f"self.input_shape : {self.input_shape}")
            cnn_arg = {
                # 'ctype' : self.cnn_cfg.get('ctype',"conv2d"),
                'input_shape' : self.input_shape, 
                'convs' :self.cnn_cfg['convs'], 
                'activation' : self.cnn_cfg.get("activation","relu"), 
                'norm_func_name' : self.cnn_cfg.get('normalization',None),
                }
            self.cnn = self._build_cnn2d(**cnn_arg)
        else:
            self.has_cnn =False
        mlp_in_shape = nn.Sequential(*self.cnn)(self.dum_input).flatten(1).data.size(1)
        # mlp_in_shape = self.input_shape[0]
        # print(f"mlp_in_shape: {mlp_in_shape}")
        self.norm_input = RunningMeanStd(self.input_shape)
        self.norm_value = RunningMeanStd(value_size,)
        self.out_size = mlp_in_shape
        if 'mlp' in config:
            mlp_cfg:dict = config.get("mlp")
            mlp_args = {
                'input_size' : mlp_in_shape, 
                    'units' : self.mlp_units, 
                    'activation' : mlp_cfg.get("activation","relu"), 
                    'norm_func_name' : mlp_cfg.get("normalization",None),
                    'dense_func' : torch.nn.Linear,
                    'norm_only_first_layer' : mlp_cfg.get("norm_only_first_layer",False)
            }
            self.mlp = self._build_sequential_mlp(**mlp_args)
            if len(self.mlp_units) == 0:
                self.out_size = mlp_in_shape
            else:
                self.out_size = self.mlp_units[-1]
        self.value = nn.Linear(self.out_size, value_size)
        self.value_act = activation_dict[value_activation]
        self.mu =nn.Linear(self.out_size,self.actions_num)
        self.sigma =nn.Linear(self.out_size,self.actions_num)
        torch.nn.init.constant_(self.sigma.weight,0)
        self.mu_acti = activation_dict[mu_activation]
        self.sig_acti = activation_dict[sigma_activation]
    def norm_obs(self,obs):
        with torch.no_grad():
            return self.norm_input(obs)

    def denorm_value(self,value):
        with torch.no_grad():
            return self.norm_value(value, denorm=True)
    def find_keys(self,d, target_key):
        results = []
        if isinstance(d, dict):
            for k, v in d.items():
                if k == target_key:
                    results.append(v)
                if isinstance(v, dict):
                    results.extend(self.find_keys(v, target_key))
                elif isinstance(v, list):
                    for item in v:
                        results.extend(self.find_keys(item, target_key))
        return results
    def _build_cnn2d(self, input_shape, convs, activation, conv_func=torch.nn.Conv2d, norm_func_name=None):
            in_channels = input_shape[1]
            layers = []
            for conv in convs:
                layers.append(conv_func(in_channels=in_channels, 
                out_channels=conv['filters'], 
                kernel_size=conv['kernel_size'], 
                stride=conv['strides'], padding=conv['padding']))
                conv_func=torch.nn.Conv2d
                act = activation_dict[activation]
                layers.append(act)
                in_channels = conv['filters']
                if norm_func_name == 'layer_norm':
                    layers.append(torch_ext.LayerNorm2d(in_channels))
                elif norm_func_name == 'batch_norm':
                    layers.append(torch.nn.BatchNorm2d(in_channels))  
            return nn.Sequential(*layers)
    def _build_sequential_mlp(self, 
        input_size, 
        units, 
        activation,
        dense_func,
        norm_only_first_layer=False, 
        norm_func_name = None):
            # print('build mlp:', input_size)
            in_size = input_size
            layers = []
            need_norm = True
            for unit in units:
                layers.append(dense_func(in_size, unit))
                layers.append(activation_dict[activation])

                if not need_norm:
                    continue
                if norm_only_first_layer and norm_func_name is not None:
                   need_norm = False 
                if norm_func_name == 'layer_norm':
                    layers.append(torch.nn.LayerNorm(unit))
                elif norm_func_name == 'batch_norm':
                    layers.append(torch.nn.BatchNorm1d(unit))
                in_size = unit

            return nn.Sequential(*layers)
    
    def cal_loss(self, obs_dict):
        """
        obs_dict: dict of observation
        result_dict: dict of result from model forward
        """
        obs = obs_dict['obs']
        actions = obs_dict['actions']
        rewards = obs_dict['rewards']
        dones = obs_dict['dones']
        prev_actions = obs_dict.get('prev_actions', None)

        result_dict = self.forward(obs_dict)
        values = result_dict['value']
        # horizonLength 만큼 쌓인 Return값을 사용 
        mu = result_dict['mu']
        logstd = result_dict['sigma']
        action = result_dict['action']
        sigma = torch.exp(logstd)
        prev_neglogp = self.neglogp(action, mu, sigma, logstd)
     
        log_probs = distr.log_prob(action)

        loss = -torch.mean(log_probs * rewards)

        return loss, value, states
    def neglogp(self, x, mean, std, logstd):
        var = std**2
        D = x.size()[-1]
        term1 = 0.5 * (((x - mean)**2) / var).sum(dim=-1)
        term2 = 0.5 * torch.log(torch.tensor(2.0 * np.pi,device='cuda')) * D
        term3 = logstd.sum(dim=-1)
        return (term1 + term2 + term3)
        # return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
        #         + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
        #         + logstd.sum(dim=-1)
    
    def forward(self, obs_dict):
        # breakpoint()
        obs = obs_dict['obs']
        obs = self.norm_obs(obs)
        pre_act = obs_dict.get('action',0)
        # print(pre_act)
        # obs = obs_dict
        if self.has_cnn:
            if self.permute_input and len(obs.shape) == 4:
                    obs = obs.permute((0, 3, 1, 2))


        out = self.cnn(obs)
        out = out.flatten(1)
        out = self.mlp(out)
        # print(f"out shape: {out.shape}")
        value = self.value_act(self.value(out))
        # print(f"value shape: {value.shape}")
        # value = self.denorm_value(value)
        # print(f"value_denorm shape: {value.shape}")
        mu = self.mu_acti(self.mu(out))
        # print(f"mu shape: {mu.shape}")
        # sigma = self.sig_acti(self.sigma(out))
        logstd = self.sig_acti(self.sigma(out))
        sigma = torch.exp(logstd)
        # print(f"sigma: {sigma}")
        if (sigma <= 0).any():
            print(f"sigma < 0 {sigma}")
        distr = torch.distributions.Normal(mu, sigma, validate_args=False)
        # print(f"distr: {distr}")
        # print(f"act shape: {act.shape}")
        # neglogp = -distr.log_prob(act).sum(dim=-1).unsqueeze(1)
        if isinstance(pre_act,torch.Tensor):
            # neglogp = self.neglogp(pre_act,mu,sigma,logstd).unsqueeze(1)
            # neglogp = -distr.log_prob(pre_act)
            # print(f"neglogp shape: {neglogp.shape}")
            if len(pre_act.shape) > 1:
                neglogp = -distr.log_prob(pre_act).sum(dim=1)
            else:
                neglogp = -distr.log_prob(pre_act).sum()
                
            # print(f"neglogp shape: {neglogp.shape}")
            act = pre_act
            dict_ = {
            'mu' : mu,
            'logstd' : logstd,
            'value' : value,
            'neglogp' : neglogp,
            'entropy' : distr.entropy(),
        }
        else:
            act = distr.sample()
            neglogp = -distr.log_prob(act).sum(dim=1)
            # neglogp = -distr.log_prob(act)
            dict_ = {
            'mu' : mu,
            'logstd' : logstd,
            'value' : value, #norm_value(value),
            # 'value' : self.denorm_value(value), #norm_value(value),
            'action' : act,
            'neglogp' : neglogp,
        }
        
        # neglogp = -distr.log_prob(act).sum(dim=-1)
        # print(f" neglogp_shape {neglogp.shape}")
        # neglogp = torch.squeeze(-distr.log_prob(act).sum(dim=-1))
        # print(f"neglogp {neglogp[0]} neglogp_shape {neglogp.shape}")
        # dict_ = {
        #     'mu' : mu,
        #     'logstd' : logstd,
        #     'value' : value,
        #     'action' : act,
        #     'neglogp' : neglogp,
        #     'entropy' : distr.entropy(),
        # }
        return dict_ 
    
    def test(self):
        return {"obs":self.dum_input}        

class ExBuffer:
    def __init__(self,init_recode:list):
        self.tensor_dict = {}
        for name in init_recode:
            self.tensor_dict[name] = []
        # print(f"self.tensor_dict : {self.tensor_dict}")

    def update(self, name:str,new_data:list):
        
        if name in self.tensor_dict:
            self.tensor_dict[name].append(new_data)
        else:
            print(f"{name} not in tensor_dict")

    def swap_flatten(self):
        # print(self.tensor_dict)
        for i in self.tensor_dict.keys():
            try:
                self.tensor_dict[i] = self.tensor_dict[i].transpose(0,1).reshape(-1,*self.tensor_dict[i].shape[2:])
            except:
                print(f"swap error {i} {self.tensor_dict[i]}")

    def split_tensor_n(self, n):
        """
        minibatch size 만큼 자르기 하나의 시나리오를 n개로 나누기
        256 -> 4 : 64, 64, 64, 64
        """
        for key in self.tensor_dict.keys():
            split_size = self.tensor_dict[key].shape[0] // n
            try:
                self.tensor_dict[key] = torch.stack([self.tensor_dict[key][i*split_size:(i+1)*split_size] for i in range(n)],dim=0)
            except:
                print(f"split error {key} {self.tensor_dict[key]}")
        
        # return [self.tensor_dict[key][i*split_size:(i+1)*split_size] for i in range(n)]

class AdaptiveScheduler:
    def __init__(self, kl_threshold = 0.008):
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold
    def update(self, current_lr, entropy_coef, kl_dist, **kwargs):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr, entropy_coef
class MainModel():

    def __init__(self,env, config_params:dict):
        config = config_params['config']
        param_net= config_params['network']
        self.gamma = config.get('gamma',0.99)
        self.tau = config.get('tau',0.95)
        self.e_clip = config.get('e_clip',0.2)
        self.ending = config.get('score_to_win', 20000)
        self.env = env
        self.num_actors = config['num_actors']
        self.env_name = config.get('env_name','CartPole-v1')
        self.env_config = config.get('env_config',{})
        self.env_info = env.get_env_info()
        self.minibatch_length = config.get('minibatch_length',8)
        self.action_spce = self.env_info['action_space']
        self.horizon_length = config.get('horizon_length',256)
        self.minibatch_size = self.horizon_length // config.get('minibatch_size',64)
        self.has_central_value = config.get('central_value_config', None) is not None
        self.use_action_masks = config.get('use_action_masks', False)
        self.max_epoch = config.get('max_epochs',10000)
        self.score = 0
        self.scheduler = AdaptiveScheduler(kl_threshold=0.01)
        self.scaler = torch.GradScaler()
        algo_info = {'num_actors': self.num_actors, 
                     'horizon_length': self.horizon_length, 
                     'has_central_value': self.has_central_value,
                     'use_action_masks': self.use_action_masks
                     }
        self.Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Network(env_con=self.env_info, net_config=param_net).to(self.Device)
        print(self.net.parameters()) 
        self.obs = None
        self.dones =torch.ones((self.env.num_envs,),dtype=torch.uint8,device=self.Device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=5e-4)

        # self.experience_buffer = ExperienceBuffer(env_info=self.env_info,algo_info=algo_info,device=Device)
    def prepare_action(self, action):
        clamp_act = torch.clamp(action,-1.0,1.0)
        action_low = torch.from_numpy(self.action_spce.low.copy()).float().to(self.Device)
        action_high = torch.from_numpy(self.action_spce.high.copy()).float().to(self.Device)
        # print(f"action_low : {action_low}, action_high : {action_high}")
        rescale_act = self.rescale_actions(action_low, action_high, clamp_act)
        return rescale_act
    def rescale_actions(self,low, high, action):
        d = (high - low) / 2.0
        m = (high + low) / 2.0
        scaled_action = action * d + m
        return scaled_action
    def train_epoch(self):
        exbuf = ExBuffer(['neglogp','logstd','mu','obs','action','value','rewards','dones','returns'])
        # obs = self.env.reset()
        
        for i in range(self.horizon_length):
            with torch.inference_mode():
                output = self.net({"obs":self.obs})

                # print(f"output['action'] : {output['action'].shape}")
                
            exbuf.update("obs",self.obs) #train에서 초기화 진행
            exbuf.update("dones",self.dones.unsqueeze(1))
            self.obs,rew,self.dones,info = self.env.step(self.prepare_action(output["action"]))
            # print(f"rew shape: {rew.shape}, done shape: {dones.shape}")
            # print(f"output['value'] : {output['value'].shape}")
            if 'time_outs' in info:
                rew += self.gamma*output["value"].squeeze(1)*info['time_outs'].to(self.Device)
            for i in output.keys():
                exbuf.update(i,output[i])
            # exbuf.update("actions",output['action'])
            exbuf.update("rewards",0.6*rew.unsqueeze(1))

        for i in exbuf.tensor_dict.keys():
            try:
                exbuf.tensor_dict[i] = torch.stack(exbuf.tensor_dict[i],dim=0)
                # print(f"stacked {i} {exbuf.tensor_dict[i].shape}")
            except:
                # print(f"stack error {i} {exbuf.tensor_dict[i]}")
                pass
        with torch.inference_mode():
            output = self.net({"obs":self.obs})
            last_value = output['value']
        dons = exbuf.tensor_dict['dones'].float()
        val = exbuf.tensor_dict['value']
        rew = exbuf.tensor_dict['rewards']
        # print(f"rew {rew.shape}")
        
        adv = self.discount_values(self.dones.unsqueeze(1).float(), last_value, dons, val, rew)
        # print(f"val shape: {val.shape}, val : {val[0]}")
        # print(f"adv shape: {adv.shape}, val : {adv[0]}")
        ret = adv + val
        exbuf.tensor_dict['returns'] = ret
        exbuf.tensor_dict['advantages'] = adv
        exbuf.tensor_dict['advantages'] = (adv - adv.mean())/(adv.std()+1e-8) # advantage normalization
        
        # self.value_loss(val,)
        exbuf.swap_flatten()

        #value normalization
        # self.net.norm_value.train()
        # exbuf.tensor_dict['value'] = self.net.norm_value(exbuf.tensor_dict['value'])
        # exbuf.tensor_dict['returns'] = self.net.norm_value(exbuf.tensor_dict['returns'], denorm=False)
        # self.net.norm_value.eval()

        exbuf.split_tensor_n(self.minibatch_length) 
        # print(torch.sum(exbuf.tensor_dict['advantages'],axis=1).shape)
        # print(exbuf.tensor_dict['advantages'].shape)
        # exbuf.tensor_dict['advantages'] = (exbuf.tensor_dict['advantages'] - exbuf.tensor_dict['advantages'].mean())/(exbuf.tensor_dict['advantages'].std()+1e-8)
        # print(exbuf.tensor_dict['advantages'].shape)

        # print(f"exbuf adv shape : {torch.sum(exbuf.tensor_dict['advantages'],1)[0],exbuf.tensor_dict['advantages'][0]}")
        # print()
        # print(f'exbuf.tensor_dict[entropy] shape : {exbuf.tensor_dict["advantages"].shape}')
        for i in range(self.minibatch_length):
            with torch.autocast(device_type="cuda"):
            # print(exbuf.tensor_dict["obs"].shape)
                output = self.net({"obs":exbuf.tensor_dict["obs"][i],'action':exbuf.tensor_dict['action'][i]})
            
                
                # a_loss = self.actor_loss(exbuf.tensor_dict["neglogp"][i],output["neglogp"],exbuf.tensor_dict["advantages"][i],self.e_clip)
                a_loss2 = self.actor_loss2(exbuf.tensor_dict["neglogp"][i],output["neglogp"],exbuf.tensor_dict["advantages"][i],self.e_clip)
                v_loss = self.value_loss(exbuf.tensor_dict["value"][i],output["value"],self.e_clip,exbuf.tensor_dict["returns"][i])
                # b_loss = self.bound_loss(exbuf.tensor_dict["mu"][i])
                # print(f"output[entropy].mean() : {output['entropy'].mean()}")
                # print(f"output[neglogp].mean() : {output['neglogp'].mean()}")
                # print(f"a_loss : {a_loss.mean()} a_loss2 : {a_loss2.mean()} ")
                # print(f"a_loss : {a_loss} v_loss : {v_loss} b_loss : {b_loss} entropy : {output['entropy']}")
                loss = a_loss2 + 2 * v_loss - output["neglogp"].mean() * 0.05 #+ b_loss*0.0001
                # print(f"loss {loss.mean()}, a_loss : {a_loss2.mean()}, v_loss : {v_loss.mean()}, entropy : {output['entropy'].mean()}")
                # print(f"loss : {loss.mean()}")
                self.loss = loss.mean()
            # print(f"loss : {self.loss}")
            # self.loss.backward()
            # self.optimizer.step()
            self.score = exbuf.tensor_dict["returns"][i].mean()
            # # scaler = torch.GradScaler()
            # self.optimizer.zero_grad()
            for param in self.net.parameters():
                param.grad = None

            # lr 조절
            # with torch.no_grad():
            #     kl_dist = torch.mean(exbuf.tensor_dict["neglogp"][i] - output["neglogp"])
            #     # print(f"kl_dist : {kl_dist}") 
            # lr, _ = self.scheduler.update(self.optimizer.param_groups[0]['lr'], 0.01, kl_dist.item())   
            # for param_group in self.optimizer.param_groups:
            #     param_group['lr'] = lr

            #backward with scaler
            self.scaler.scale(self.loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # self.net.norm_input.eval()
        # scaler.step(torch.optim.Adam(self.net.parameters(), lr=3e-4))
       
        # after backward & before optimizer step
        # for name, param in self.net.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} grad norm: {param.grad.norm().item():.6f}")
        #     else:
        #         print(f"{name} grad is None")
        # # print parameter norm before/after update
        # param_norm = sum(p.norm().item() for p in self.net.parameters())
        # print("param_norm:", param_norm)
        


    def discount_values(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            # nextnonterminal = nextnonterminal.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_extrinsic_values[t]
            # print(f"delta shape: {mb_rewards[t].shape}, nextnonterminal shape: {nextnonterminal.shape}, nextvalues shape: {nextvalues.shape}, mb_extrinsic_values[t] shape: {mb_extrinsic_values[t].shape}")
            mb_advs[t] = lastgaelam = delta + self.gamma * self.tau * nextnonterminal * lastgaelam
        return mb_advs
    def bound_loss(self,mu):
        high = torch.clamp_min(mu - 1.1,0.0)**2
        low = torch.clamp_max(mu + 1.1 , 0.0)**2
        return (high+low).sum(dim=-1)
    def smooth_clamp(self,min,max,x):
        return 1/(1 + torch.exp((-(x-min)/(max-min)+0.5)*4)) * (max-min) + min
    def actor_loss2(self,old_neglogp, neglogp, adv, e_clip):
        
        old_neg = old_neglogp.view(-1)
        new_neg = neglogp.view(-1)
        advantage = adv.view(-1)

        # compute logprobs from neglogp
        old_logp = -old_neg
        new_logp = -new_neg
        # print(f"old_logp : {old_logp.shape}, new_logp : {new_logp.shape}, advantage : {advantage.shape}")
        ratio = torch.exp(new_logp - old_logp)  # p_new / p_old
        # print(f"ratio : {ratio}")
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - e_clip, 1.0 + e_clip) * advantage

        # PPO loss is negative of clipped surrogate (we want to maximize surrogate)
        loss = -torch.min(surr1, surr2).mean()

        # return per-sample loss (keep dims consistent)
        return loss.view(-1, 1)

    
    def value_loss(self, value_preds_batch, values, curr_e_clip, return_batch):
        value_pred_clipped = value_preds_batch + \
                (values - value_preds_batch).clamp(-curr_e_clip, curr_e_clip)
        value_losses = (values - return_batch)**2
        value_losses_clipped = (value_pred_clipped - return_batch)**2
        c_loss = torch.maximum(value_losses, value_losses_clipped)
        # c = (return_batch-values)**2
        return c_loss.mean()
    
    def save(self, path):
        torch.save(self.net.state_dict(), path)
    
    def train(self):
        epoch_num = 0
        self.obs = self.env.reset()
        while epoch_num <= self.max_epoch and self.score <= self.ending:
            last_score = self.score
            self.train_epoch()
            if last_score >= self.ending:
                print(f"Achieved the target score of {self.ending} at epoch {epoch_num}!")
                self.save(f"{self.env_name}_final.pth")
            if epoch_num % 100 == 0:
                print(f"epoch_num : {epoch_num}, loss : {self.loss}, score : {self.score}")
                # print(f"epoch_num : {epoch_num}, score : {self.score}")
                if epoch_num % 500 == 0:
                    self.save(f"{self.env_name}_epoch{epoch_num}.pth")
                    print(f"Model saved at epoch {epoch_num}!")
            epoch_num += 1
        pass
    def load_play(self):
        self.net.load_state_dict(torch.load(f"{self.env_name}_epoch1000.pth"))
        obs = self.env.reset()
        dones = torch.ones((self.env.num_envs,),dtype=torch.uint8,device=self.Device)
        step = 0
        while step < 1000:
            with torch.inference_mode():
                output = self.net({"obs":obs})
                # print(f"output['value'] : {output['value'].shape}")
                obs,rew,dones,_ = self.env.step(self.prepare_action(output["action"]))
                # print(f"obs {obs.shape},rew{rew.shape},done{dones.shape}")
            step += 1
        pass

def main():
    task = "Isaac-Cartpole-RGB-v0"
    task = "Isaac-Humanoid-v0"
    cfg = load_cfg_from_registry(task,"env_cfg_entry_point")
    cfg1:dict = load_cfg_from_registry(task,"rl_games_cfg_entry_point")
    env = gym.make(task,cfg=cfg)
    # print(cfg1['params']["config"])
    #시작 포인트 play step 에서 저장될때까지 shape는 똑가틍ㄴ데 그이후 왜 달라지는지 확인하기

    env_name = cfg1['params']["config"].get('env_name',task)
    num_actors = cfg1['params']["config"].get('num_actors',1)
    env_config = cfg1['params']["config"].get('env_config',{})
    # env.step()
    # print(env.action_space)
    # print(env.observation_space)
    env = RlGamesVecEnvWrapper(env, "cuda", clip_obs=math.inf, clip_actions=1.0)
    net = Network(env.get_env_info(),cfg1['params']['network']).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    obs = env.reset()
    model = MainModel(env,cfg1['params'])
    # model.train_epoch()

    # model.train()
    model.load_play()
    # print(model.experience_buffer)
    # print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    horizon = cfg1["params"]["config"]["horizon_length"]

    # print(vecenv.create_vec_env(env_name, num_actors, **env_config).get_env_info())
    # obs,_ = env.reset()
    # print(obs.shape)
    # 사람의 긴장도 로 horizon을 정함
    # 긴장도의 기준은 위험도로 정함 horizon내 done이 되면 
    # 사람의 학습은 자면서 진행 그동한 경험으로
    # 꿈에서 보는 것들은 뇌에서 만들어 낸것 dreamer의 이미지 표현도 마찬가지 

    """
    
    value 다시 보기 common.py에 prepare에 있음 훈련도 진행하는듯
    entropy loss 확인하기
    """

    step = 0
    # breakpoint()
    start_tim = time.time()

    # while step <50:
    #     for i in range(horizon):
    #         with torch.inference_mode():
    #             output = net({'obs':obs})
    #             print(f"output['value'] : {output['value'].shape}")
    #             obs,rew,dones,_ = env.step(output["action"])
    #             print(f"obs {obs.shape},rew{rew.shape},done{dones.shape}")
    #             rewards_shaper = cfg1["params"]["config"]["reward_shaper"]
    #         # print(rewards_shaper(rew))
    #         # obs,_,_,_,_ = env.step(net({'obs':obs['policy']})["action"])
    #     # env.step(net(net.test())["action"])
    #     step += 1
    # print(f"step time: {time.time() - start_tim}")

    # print(net(net.test()))

    # print(net)
    env.close()
if __name__ == "__main__":
    main()
    simulation_app.close()