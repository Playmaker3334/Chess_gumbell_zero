import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = BasicBlock(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, momentum=0.01, eps=1e-3)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(out + residual)

class ChessGumbelNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_input = BasicBlock(config.observation_shape[0], config.num_channels, kernel_size=3, padding=1)
        
        self.backbone = nn.ModuleList([
            ResBlock(config.num_channels) for _ in range(config.num_res_blocks)
        ])

        self.policy_head = nn.Sequential(
            BasicBlock(config.num_channels, config.num_channels, kernel_size=3, padding=1),
            nn.Conv2d(config.num_channels, 73, kernel_size=1, bias=False), 
            nn.BatchNorm2d(73),
            nn.Flatten()
        )
        
        self.value_head = nn.Sequential(
            BasicBlock(config.num_channels, 32, kernel_size=1),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

        self.action_space = config.action_space_size

    def forward(self, x):
        x = self.conv_input(x)
        for block in self.backbone:
            x = block(x)
        
        policy_logits = self.policy_head(x)
        
        flattened_policy = policy_logits.view(x.size(0), -1)
        
        value = self.value_head(x)
        
        return flattened_policy[:, :self.action_space], value