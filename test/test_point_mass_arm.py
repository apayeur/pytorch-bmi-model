from effectors import PointMassArm
import torch

workspace_dim = 3
pma = PointMassArm(workspace_dim=workspace_dim)

# Produce some fake controls
controls = torch.randn((8, 15, workspace_dim))

o = pma(controls)