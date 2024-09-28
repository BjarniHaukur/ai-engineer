import torch
import torch.nn.functional as F
from torch import nn

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, routing_iters=3):
        super(CapsuleLayer, self).__init__()
        self.num_route_nodes = num_route_nodes
        self.num_capsules = num_capsules
        self.routing_iters = routing_iters

        self.W = nn.Parameter(torch.randn(1, num_route_nodes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = torch.zeros(1, self.num_route_nodes, self.num_capsules, 1).to(x.device)

        for iteration in range(self.routing_iters):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < self.routing_iters - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), v_j).squeeze(4)
                b_ij = b_ij + a_ij

        return v_j.squeeze(1)

    @staticmethod
    def squash(s_j):
        s_j_norm = torch.norm(s_j, dim=-1, keepdim=True)
        return (s_j_norm**2 / (1 + s_j_norm**2)) * (s_j / s_j_norm)

class PrimaryCapsules(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCapsules, self).__init__()
        self.num_capsules = num_capsules
        self.capsules = nn.Conv2d(in_channels, num_capsules * out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.capsules(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1, x.size(1) // self.num_capsules, x.size(2), x.size(3))
        x = x.view(batch_size, -1, x.size(2) * x.size(3) * x.size(4))
        return self.squash(x)

    @staticmethod
    def squash(x):
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        return (x_norm**2 / (1 + x_norm**2)) * (x / x_norm)
