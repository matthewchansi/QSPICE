#from hartmann_tutoriak import ProbabilityOfFeasibility
from botorch.acquisition.analytic import (UpperConfidenceBound, ExpectedImprovement, 
    ProbabilityOfImprovement, AnalyticAcquisitionFunction, _compute_log_prob_feas, 
    _preprocess_constraint_bounds, convert_to_target_pre_hook, LogConstrainedExpectedImprovement, ConstrainedExpectedImprovement)

from typing import Dict, Union, Tuple, Optional
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from pymoo.visualization.scatter import Scatter
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.problems.functional import FunctionalProblem
from pareto_front_testing import MyProblem
from botorch.models.model import Model

from botorch.acquisition.objective import ScalarizedPosteriorTransform, PosteriorTransform

from botorch.utils import t_batch_mode_transform
import numpy as np
import torch
import time

from torch import Tensor
X_SZ = 8
Y_SZ = 3
X = torch.rand(40, X_SZ)
Y = torch.rand(40, Y_SZ)


pt = ScalarizedPosteriorTransform(torch.cat([torch.tensor([1],
    dtype=torch.float32), torch.zeros(Y_SZ-1, dtype=torch.float32)]))

gp = SingleTaskGP(X, Y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)


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
        print(means.shape)
        print(sigmas.shape)
        print(sigmas[...,0])
        print(sigmas[:,0])
        exit()
        log_prob_feas = _compute_log_prob_feas(self, means=means, sigmas=sigmas)
        return log_prob_feas.exp()

const = {1: [0, 0.1], 2:[0.2, 0.9]}

EI = ExpectedImprovement(gp, best_f=Y[:, 0].max(), posterior_transform=pt)
CEI = ConstrainedExpectedImprovement(gp, best_f=Y[:, 0].max(), constraints=const, objective_index=0)
PF = ProbabilityOfFeasibility(gp, objective_index=0, constraints=const)

t0_sum = 0
t1_sum = 0

for i in range(0, 100):
    my_in = torch.rand(2, 1, X_SZ)
    ts = time.time()
    mypf = PF(my_in)
    t0_sum = t0_sum + time.time() - ts
    ts = time.time()
    realpf = torch.div(CEI(my_in), EI(my_in))
    t1_sum = t1_sum + time.time() - ts
    #print(mypf, realpf, torch.equal(mypf, realpf), torch.sub(mypf,realpf))
# generate synthetic data
print(t0_sum, t1_sum)
if (t1_sum > t0_sum):
    print(f"t1 is slower than t0 by {t1_sum / t0_sum * 100}%")
else:
    print(f"t0 is slower than t1 by {t0_sum / t1_sum * 100}%")