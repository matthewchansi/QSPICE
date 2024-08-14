import gpytorch
import numpy as np
import torch

from botorch.models import SingleTaskGP

train_x = torch.rand(2000, 3)
train_y = torch.stack((torch.sin(train_x[:, 0]), torch.cos(train_x[:, 1])))

test_x = torch.rand(100, 3)
test_y = torch.stack((torch.sin(test_x[:, 0]), torch.cos(test_x[:, 1])))

'''
class MyGP(SingleTaskGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood=likelihood)
#        self.mean_module = gpytorch.means.ZeroMean()
#        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean = super().mean_module(x)
        covar = super().covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

# train_x = ...; train_y = ...
#likelihood = gpytorch.likelihoods.GaussianLikelihood()
#model = MyGP(train_x, train_y, likelihood)
'''
from gpytorch.constraints import GreaterThan
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize

train_X_dim = train_x.shape[-1]
train_Y_dim = train_y.shape[-1]
model = SingleTaskGP(train_X = train_x, 
                     train_Y= train_y, 
                     input_transform=Normalize(d=train_X_dim),
                     outcome_transform=Standardize(m=train_Y_dim))
#model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
#print(train_x.shape)
#print(np.array(model.train_inputs).shape)

from gpytorch.mlls import ExactMarginalLogLikelihood

mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
# set mll and all submodules to the specified dtype and device
mll = mll.to(train_x)

from torch.optim import SGD

optimizer = SGD([{"params": model.parameters()}], lr=0.1)

NUM_EPOCHS = 150

model.train()

for epoch in range(NUM_EPOCHS):
    # clear gradients
    optimizer.zero_grad()
    # forward pass through the model to obtain the output MultivariateNormal
    with gpytorch.settings.debug(state=False): 
        output = model(train_x)
    #print(train_y.shape)
    #print(output.shape)
    print(train_x.shape)
    print(model.mean_module(train_x).shape)
    print(model.covar_module(train_x).shape)

    # Compute negative marginal log likelihood
    loss = -mll(output, train_y)
    # back prop gradients
    loss.backward()
    # print every 10 iterations
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1:>3}/{NUM_EPOCHS} - Loss: {loss.item():>4.3f} "
            f"lengthscale: {model.covar_module.base_kernel.lengthscale.item():>4.3f} "
            f"noise: {model.likelihood.noise.item():>4.3f}"
        )
    optimizer.step()

model.eval()
# test_x = ...;
model(test_x)  # Returns the GP latent function at test_x
likelihood(model(test_x))  # Returns the (approximate) predictive posterior distribution at test_x