import torch


class Config:
    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        self.window_size = 50000
        self.batch_size = 2048
        
        self.training_steps = 1000000
        self.checkpoint_interval = 50
        
        self.lr_init = 0.0002  
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = 100000
        self.momentum = 0.9
        self.weight_decay = 1e-4

        self.num_simulations = 16
        self.num_sampled_actions = 8
        self.c_visit = 50
        self.c_scale = 1.0

        self.num_channels = 128
        self.num_res_blocks = 4
        self.action_space_size = 4672
        self.observation_shape = (119, 8, 8)
        self.bottleneck_channels = 64
        self.broadcast_every_n = 8

        self.num_self_play_workers = 10
        self.games_per_worker = 1
        self.max_moves_per_game = 80

        self.log_dir = "logs"
        self.log_games = True
        self.log_every_n_games = 10
