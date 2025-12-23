import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class GumbelTrainer:
    def __init__(self, config, network):
        self.config = config
        self.network = network.to(config.device)
        self.optimizer = optim.SGD(
            self.network.parameters(), 
            lr=config.lr_init, 
            momentum=config.momentum, 
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, 
            milestones=[config.lr_decay_steps], 
            gamma=config.lr_decay_rate
        )

    def train_step(self, batch):
        """
        batch: (states, target_policies, target_values)
        """
        states, target_policies, target_values = batch
        
        states = states.to(self.config.device)
        target_policies = target_policies.to(self.config.device)
        target_values = target_values.to(self.config.device)

        self.optimizer.zero_grad()
        
        # Forward pass
        pred_policies, pred_values = self.network(states)
        
        # 1. Value Loss (MSE)
        # Queremos que el valor predicho se acerque al resultado real del juego
        value_loss = F.mse_loss(pred_values, target_values)
        
        # 2. Policy Loss (Cross Entropy / KL Divergence)
        # Queremos que la red prediga la política "mejorada" por Gumbel Search
        # LogSoftmax para estabilidad numérica
        log_pred_policies = F.log_softmax(pred_policies, dim=1)
        
        # KLDiv espera input en log-space y target en prob-space
        # reduction='batchmean' es la forma matemáticamente correcta de KL
        policy_loss = F.kl_div(log_pred_policies, target_policies, reduction='batchmean')
        
        total_loss = value_loss + policy_loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return {
            "total_loss": total_loss.item(),
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "lr": self.scheduler.get_last_lr()[0]
        }