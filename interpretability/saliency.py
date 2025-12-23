import torch
import numpy as np
import matplotlib.pyplot as plt

class SaliencyMap:
    def __init__(self, network, device):
        self.network = network
        self.device = device

    def compute_saliency(self, state_tensor):
        self.network.eval()
        state = state_tensor.to(self.device).requires_grad_()
        
        _, value = self.network(state)
        
        value.backward()
        
        saliency = state.grad.data.abs()
        saliency, _ = torch.max(saliency, dim=1) 
        saliency = saliency.squeeze().cpu().numpy()
        
        return saliency

    def save_heatmap(self, saliency_grid, filename="saliency.png"):
        plt.figure(figsize=(6, 6))
        plt.imshow(saliency_grid, cmap='hot', interpolation='nearest')
        plt.axis('off')
        
        for i in range(8):
            for j in range(8):
                if saliency_grid[i, j] > 0.1: 
                    plt.text(j, i, f"{saliency_grid[i,j]:.2f}", 
                             ha="center", va="center", color="blue", fontsize=8)

        plt.savefig(filename, bbox_inches='tight')
        plt.close()