import torch
import torch.nn as nn
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
        # self.actions_num = env_con['action_space'].shape[0]
        self.input_shape = env_con['observation_space'].shape
        print(f"actions_num : {self.actions_num}, input_shape : {env_con['observation_space']}")
        # self.actions_num = env.action_space.shape[1]
        # self.input_shape = env.observation_space['policy'].shape
       
        # model_config.update(net_config)
        # breakpoint()

        config = net_config
        self.value_normalize = config.get('value_normalize', False)
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
            return self.norm_value(value, denorm=True) if self.value_normalize else value
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
            # 'value' : value, #norm_value(value),
            'value' : self.denorm_value(value), #norm_value(value),
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
