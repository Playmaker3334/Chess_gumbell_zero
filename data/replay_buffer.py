import numpy as np
import pickle
import os
import multiprocessing as mp
from collections import deque

class ReplayBuffer:
    def __init__(self, config):
        self.window_size = config.window_size
        self.buffer = deque(maxlen=self.window_size)
        self.config = config
        self.lock = mp.Lock()

    def save_game(self, game_history, result):
        """
        game_history: lista de tuplas (estado, politica_mejorada, jugador_actual)
        result: resultado final del juego (1 para victoria de White, -1 Black, 0 Draw)
        """
        samples = []
        for state, policy, player_color in game_history:
            value_target = result * (1 if player_color else -1)
            state_compressed = state.numpy().astype(np.float16)
            samples.append({
                "state": state_compressed,
                "policy": policy,
                "value": value_target
            })

        with self.lock:
            for sample in samples:
                self.buffer.append(sample)

    def add_samples(self, samples):
        """Agregar samples pre-procesados (usado por workers paralelos)"""
        with self.lock:
            for sample in samples:
                self.buffer.append(sample)

    def sample_batch(self, batch_size):
        with self.lock:
            if len(self.buffer) < batch_size:
                return None
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            batch = [self.buffer[i] for i in indices]

        states = np.array([b["state"] for b in batch])
        policies = np.array([b["policy"] for b in batch])
        values = np.array([b["value"] for b in batch])

        return (
            states.astype(np.float32),
            policies.astype(np.float32),
            values.astype(np.float32)
        )

    def __len__(self):
        with self.lock:
            return len(self.buffer)

    def save_buffer(self, path):
        with self.lock:
            with open(path, 'wb') as f:
                pickle.dump(list(self.buffer), f)

    def load_buffer(self, path):
        if os.path.exists(path):
            with self.lock:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    self.buffer = deque(data, maxlen=self.window_size)
