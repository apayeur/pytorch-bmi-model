import sys
sys.path.append("../src")
from datasets import GaussianVelocityDataset
import matplotlib.pyplot as plt


data = GaussianVelocityDataset()


plt.plot(data[0][1][:,0])
plt.show()
