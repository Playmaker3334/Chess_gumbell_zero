import torch
from torch.utils.data import IterableDataset

class ChessDataset(IterableDataset):
    def __init__(self, replay_buffer, batch_size):
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            if len(self.replay_buffer) < self.batch_size:
                # Si no hay suficientes datos, yield dummy o espera
                # Para evitar bloqueo en loops iniciales, retornamos nada hasta tener datos
                continue
            
            states, policies, values = self.replay_buffer.sample_batch(self.batch_size)
            
            # Squeeze dim 1 si viene de numpy (B, 1, C, H, W) -> (B, C, H, W)
            states = torch.from_numpy(states).squeeze(1)
            policies = torch.from_numpy(policies)
            values = torch.from_numpy(values).unsqueeze(1) # (B, 1)

            yield states, policies, values

    def __len__(self):
        return len(self.replay_buffer)