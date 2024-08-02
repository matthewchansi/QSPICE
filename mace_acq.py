import time
from typing import Dict, Union, Tuple, Optional
from pymoo.visualization.scatter import Scatter
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
import numpy as np
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils import standardize
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
import torch
from torch import Tensor
from botorch.acquisition.analytic import (UpperConfidenceBound, ExpectedImprovement, 
    ProbabilityOfImprovement, AnalyticAcquisitionFunction, _compute_log_prob_feas, 
    _preprocess_constraint_bounds, convert_to_target_pre_hook)
import plotly.io as pio
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform
from botorch.acquisition.objective import ScalarizedPosteriorTransform, PosteriorTransform
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior_list import PosteriorList  # pragma: no cover
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from botorch.sampling.normal import SobolQMCNormalSampler

class ProbabilityOfFeasibility(AnalyticAcquisitionFunction):
    r"""Probability of Feasibility.

    """
    def __init__(
        self,
        model: Model,
        objective_index: int,
        constraints: Dict[int, Tuple[Optional[float], Optional[float]]],
    ) -> None:
        r"""Analytic Constrained Expected Improvement.

        Args:
            model: A fitted multi-output model.
            objective_index: The index of the objective.
            constraints: A dictionary of the form `{i: [lower, upper]}`, where
                `i` is the output index, and `lower` and `upper` are lower and upper
                bounds on that output (resp. interpreted as -Inf / Inf if None)
        """
        self.posterior_transform = None
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.objective_index = objective_index
        self.constraints = constraints
        _preprocess_constraint_bounds(self, constraints=constraints)
        self.register_forward_pre_hook(convert_to_target_pre_hook)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Constrained Expected Improvement on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Probability of Feasibility values at the given
            design points `X`.
        """
        means, sigmas = self._mean_and_sigma(X)  # (b) x 1 + (m = num constraints)
        log_prob_feas = _compute_log_prob_feas(self, means=means, sigmas=sigmas)
        return log_prob_feas.exp()
    

class SelectPosteriorTransform(PosteriorTransform):
    r"""A posterior transform.
        It seems that using scalarize with 1 to select the correct outcome 
        is faster...
    """

    def __init__(self, outcome_index: float=0) -> None:
        r"""
        Args:
        """
        super().__init__()
        self.outcome_index = outcome_index

    def evaluate(self, Y: Tensor) -> Tensor:
        pass

    def forward(
        self, posterior: Union[GPyTorchPosterior, PosteriorList]
    ) -> GPyTorchPosterior:
        
        mean = posterior.mean
        mvn = posterior.distribution
        cov = mvn.lazy_covariance_matrix if mvn.islazy else mvn.covariance_matrix

        new_mean = mean[:, :, self.outcome_index]
        new_cov = cov[:, self.outcome_index, self.outcome_index].unsqueeze(1).unsqueeze(1)
        return MultivariateNormal(new_mean, new_cov)
    

class MyProblem(Problem):
    def __init__(self, n_var,
                 objs, const_fn, **kwargs):
        self.objs = objs
        self.const_fn = const_fn
        super().__init__(n_var=n_var, n_obj=len(objs), **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xt = torch.from_numpy(np.float32(np.expand_dims(x, axis=1)))
        const_mul = self.const_fn(xt)
        out["F"] = np.array([torch.mul(obj(xt), const_mul).detach().numpy() for obj in self.objs])

def selectRandSamples(batch, batchsz):
    B = torch.randperm(len(batch))
    return batch[B[:batchsz]]

X_SZ = 3
# generate synthetic data
X = torch.rand(20, X_SZ)
# print()
ub = torch.max(X, 0).values.detach().numpy()
lb = torch.min(X, 0).values.detach().numpy()
# torch.stack([torch.sin(X[:, 0]), torch.cos(X[:, 1])], -1)
#Y = torch.sin(X).sum(dim=1, keepdim=True)
#Y = standardize(Y)  # standardize to zero mean unit variance
Y = standardize(torch.stack([torch.sin(X[:, 0]), torch.sin(X[:, 1])], -1))

best_f = torch.max(Y, dim=0).values[0]
gp = SingleTaskGP(X, Y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

pt = ScalarizedPosteriorTransform(Tensor([1, 0]))

const = {1: [0.5, 1]}

UCB = UpperConfidenceBound(gp, beta=0.1, posterior_transform=pt)
PI = ProbabilityOfImprovement(gp, best_f=max(Y[0]), posterior_transform=pt)
EI = ExpectedImprovement(gp, best_f=max(Y[0]), posterior_transform=pt)
PF = ProbabilityOfFeasibility(gp, objective_index=0, constraints=const)

obs = [lambda x: -1 * UCB(x), lambda x: -1 * PI(x), lambda x: -1 * EI(x)]

problem = MyProblem(X_SZ, obs, PF, xl=lb, xu=ub)

algorithm = NSGA2(pop_size=100)

st = time.time()

res = minimize(problem, algorithm, ('n_gen', 100), verbose=False, seed=1)

print(selectRandSamples(res.X, 4))

st1 = time.time()

print(f"time nsga2: {st1-st}\n goal: 2.3s")
#p = Scatter()
#p.add(res.F, facecolor="red", edgecolor="none")
#p.show()
