import gpytorch
import numpy as np
import torch
import math
import torch
import gpytorch
from matplotlib import pyplot as plt

from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.likelihoods.gaussian_likelihood import (
    _GaussianLikelihoodBase,
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)

torch.set_default_tensor_type(torch.cuda.FloatTensor)

train_x = torch.rand(1000, 2)
train_x, _ = torch.sort(train_x, dim=1)

train_y = torch.stack((torch.sin(train_x[:, 0] * (2 * math.pi)), torch.cos(train_x[:, 1]* (2 * math.pi))), -1)

test_x = torch.rand(100, 2)
test_y = torch.stack((torch.sin(test_x[:, 0] * (2 * math.pi)), torch.cos(test_x[:, 1]* (2 * math.pi))), -1)

from gpytorch.constraints import GreaterThan
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize

class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([2]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([2])),
            batch_shape=torch.Size([2])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood)

training_iterations = 500


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
    optimizer.step()

model.eval()
likelihood.eval()

# Initialize plots
f, (y1_ax, y2_ax) = plt.subplots(1, 2)

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions = likelihood(model(test_x))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()

# This contains predictions for both tasks, flattened out
# The first half of the predictions is for the first task
# The second half is for the second task

train_y = train_y.cpu()
train_x = train_x.cpu()
test_y = test_y.cpu()
test_x = test_x.cpu()
mean = mean.cpu()
lower = lower.cpu()
upper = upper.cpu()

# Plot training data as black stars
y1_ax.plot(train_x[:, 0].detach().numpy(), train_y[:, 0].detach().numpy(), 'k*')
# Predictive mean as blue line
y1_ax.plot(test_x[:, 0].detach().numpy(), mean[:, 0].detach().numpy(), 'bo')
y1_ax.plot(test_x[:, 0].detach().numpy(), lower[:, 0].detach().numpy(), 'go')
y1_ax.plot(test_x[:, 0].detach().numpy(), upper[:, 0].detach().numpy(), 'ro')

y1_ax.legend(['Observed Data', 'Mean', 'lowerconfidence', 'upperconfidence'])
y1_ax.set_title('Observed Values (Likelihood)')

# Plot training data as black stars
y2_ax.plot(train_x[:, 1].detach().numpy(), train_y[:, 1].detach().numpy(), 'k*')
# Predictive mean as blue line
y2_ax.plot(test_x[:, 1].detach().numpy(), mean[:, 1].detach().numpy(), 'bo')
y2_ax.plot(test_x[:, 1].detach().numpy(), lower[:, 1].detach().numpy(), 'go')
y2_ax.plot(test_x[:, 1].detach().numpy(), upper[:, 1].detach().numpy(), 'ro')

y2_ax.legend(['Observed Data', 'Mean', 'lowerconfidence', 'upperconfidence'])
y2_ax.set_title('Observed Values (Likelihood)')


plt.show()