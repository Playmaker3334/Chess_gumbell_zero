import torch

class Config:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.window_size = 100000
        self.batch_size = 128
        self.num_unroll_steps = 5
        self.td_steps = 100000
        self.discount = 0.997
        
        self.training_steps = 1000000
        self.checkpoint_interval = 1000
        self.lr_init = 0.002
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = 100000
        self.momentum = 0.9
        self.weight_decay = 1e-4

        self.num_simulations = 16
        self.num_sampled_actions = 8
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25
        self.c_visit = 50
        self.c_scale = 1.0

        self.num_channels = 256
        self.num_res_blocks = 10
        self.action_space_size = 4672
        self.observation_shape = (119, 8, 8)
        self.bottleneck_channels = 128
        self.broadcast_every_n = 8