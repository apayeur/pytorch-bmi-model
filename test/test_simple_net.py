import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
sys.path.append("../src")
from datasets import GaussianVelocityDataset
from networks import SimpleNet

# Reproducibility
torch.manual_seed(1)

# Define network
network_size = (5, 100, 100, 2)
net = SimpleNet(network_size=network_size)
print(net)


# Train
dataset = GaussianVelocityDataset()
dataloader = DataLoader(dataset, batch_size=8)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=2e-4)

encoder_weight_before = net.input_encoder.weight
recurrent_weight_before = net.motor_cortex.weight_hh_l0.data
readout_weight_before = net.readout_layer.weight

print("Recurrent weights before", net.motor_cortex.weight_hh_l0.data[:5, :5])
print("Readout weights before", readout_weight_before[:, 10])

epochs = 1000
for t in range(epochs):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        h0 = 0.1*torch.rand((1, 8, network_size[2]))
        pred = net(X, h0)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 100 == 0:
            print(f"Epoch {t}: loss = {loss.item():>10f}")

print("Recurrent weights after", net.motor_cortex.weight_hh_l0[:5, :5])
print("Readout weights after", net.readout_layer.weight[:, 10])



print(torch.max(net.input_encoder.weight.data - encoder_weight_before.data))
print(torch.max(net.motor_cortex.weight_hh_l0.data - recurrent_weight_before))
print(torch.max(net.readout_layer.weight - readout_weight_before))

# Plotting
pred = net(dataset[0][0], 0.1*torch.rand((1, network_size[2])))
print(pred.size())
plt.plot(pred[:,0].detach().numpy())
plt.show()