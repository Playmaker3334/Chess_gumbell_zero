import torch

class Config:
    def __init__(self):
        # Detectar GPU automáticamente
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Configuración de Memoria
        self.window_size = 10000
        self.batch_size = 64
        
        # Hiperparámetros de Entrenamiento
        self.training_steps = 1000000
        self.checkpoint_interval = 50  # Evaluar cada 50 pasos
        
        # --- CORRECCIÓN CLAVE: Learning Rate más bajo (0.0002) ---
        self.lr_init = 0.0002  
        # ---------------------------------------------------------
        
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = 100000
        self.momentum = 0.9
        self.weight_decay = 1e-4

        # Configuración de Búsqueda (Gumbel)
        self.num_simulations = 16
        self.num_sampled_actions = 8
        self.c_visit = 50
        self.c_scale = 1.0

        # Arquitectura de Red (Ligera para Colab)
        self.num_channels = 128
        self.num_res_blocks = 4
        self.action_space_size = 4672
        self.observation_shape = (119, 8, 8)
        self.bottleneck_channels = 64
        self.broadcast_every_n = 8
