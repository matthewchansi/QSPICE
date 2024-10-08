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

def gen_y(x_tensor):
    return torch.stack((torch.sin(x_tensor[:, 0] * (2 * math.pi)), torch.mul(torch.cos(torch.pow(x_tensor[:, 1] *  math.pi, 2)), x_tensor[:, 1] * 100)), -1)

train_x = torch.rand(1000, 3)
train_y = gen_y(train_x)

test_x = torch.rand(100, 3)
test_y = gen_y(test_x)


from gpytorch.constraints import GreaterThan
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize

data_dim = train_x.size(-1)

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(1000, 500))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(500, 50))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(50, 2))

#feature_extractor = LargeFeatureExtractor()

# DKL https://arxiv.org/pdf/1511.02222
# https://docs.gpytorch.ai/en/stable/examples/06_PyTorch_NN_Integration_DKL/KISSGP_Deep_Kernel_Regression_CUDA.html
# https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/Batch_Independent_Multioutput_GP.html

class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        batchsize = torch.Size([train_y.shape[-1]])

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batchsize)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=batchsize),
            batch_shape=batchsize
        )

        self.feature_extractor = LargeFeatureExtractor()
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)

        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2).cuda()
model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood).cuda()

training_iterations = 500

# Find optimal model hyperparameters
model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.feature_extractor.parameters()},
    {'params': model.covar_module.parameters()},
    {'params': model.mean_module.parameters()},
    {'params': model.likelihood.parameters()},
], lr=0.01)

# Use the adam optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()

    if (i % 100 == 100-1):
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
    
    #print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
    optimizer.step()

model.eval()
likelihood.eval()

# Initialize plots
f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))

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
y1_ax.plot(test_x[:, 0].detach().numpy(), lower[:, 0].detach().numpy(), 'g.')
y1_ax.plot(test_x[:, 0].detach().numpy(), upper[:, 0].detach().numpy(), 'r.')

y1_ax.legend(['Observed Data', 'Mean', 'lowerconfidence', 'upperconfidence'])
y1_ax.set_title('Observed Values (Likelihood)')

# Plot training data as black stars
y2_ax.plot(train_x[:, 1].detach().numpy(), train_y[:, 1].detach().numpy(), 'k*')
# Predictive mean as blue line
y2_ax.plot(test_x[:, 1].detach().numpy(), mean[:, 1].detach().numpy(), 'bo')
y2_ax.plot(test_x[:, 1].detach().numpy(), lower[:, 1].detach().numpy(), 'g.')
y2_ax.plot(test_x[:, 1].detach().numpy(), upper[:, 1].detach().numpy(), 'r.')

y2_ax.legend(['Observed Data', 'Mean', 'lowerconfidence', 'upperconfidence'])
y2_ax.set_title('Observed Values (Likelihood)')
plt.savefig("multioutputgp.jpg")
plt.show()