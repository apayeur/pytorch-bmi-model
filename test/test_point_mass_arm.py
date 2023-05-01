from effectors import PointMassArm
import torch

pma = PointMassArm(workspace_dim=3)

# Produce some fake controls
controls = torch.randn((8, 15, 3))

o = pma(controls)