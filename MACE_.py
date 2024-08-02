from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound, AnalyticAcquisitionFunction
from botorch.acquisition.analytic import _compute_log_prob_feas, _preprocess_constraint_bounds
from botorch.utils.transforms import convert_to_target_pre_hook, t_batch_mode_transform
from botorch.models.model import Model
from typing import Dict, Optional, Tuple, Union
from torch import Tensor
import torch

class MACE(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        # best_f: Union[float, Tensor],
        objective_index: int,
        constraints: Dict[int, Tuple[Optional[float], Optional[float]]],
        maximize: bool = True,
    ) -> None:
        r"""Analytic Log Constrained Expected Improvement.

        Args:
            model: A fitted multi-output model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best feasible function value observed so far (assumed noiseless).
            objective_index: The index of the objective.
            constraints: A dictionary of the form `{i: [lower, upper]}`, where
                `i` is the output index, and `lower` and `upper` are lower and upper
                bounds on that output (resp. interpreted as -Inf / Inf if None)
            maximize: If True, consider the problem a maximization problem.
        """
        # Use AcquisitionFunction constructor to avoid check for posterior transform.
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.posterior_transform = None
        self.maximize = maximize
        self.objective_index = objective_index
        self.constraints = constraints
        # self.register_buffer("best_f", torch.as_tensor(best_f))
        _preprocess_constraint_bounds(self, constraints=constraints)
        self.register_forward_pre_hook(convert_to_target_pre_hook)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Constrained Log Expected Improvement on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Log Expected Improvement values at the given
            design points `X`.
        """
        means, sigmas = self._mean_and_sigma(X)  # (b) x 1 + (m = num constraints)
        ind = self.objective_index
        mean_obj, sigma_obj = means[..., ind], sigmas[..., ind]
        # u = _scaled_improvement(mean_obj, sigma_obj, self.best_f, self.maximize)
        # log_ei = _log_ei_helper(u) + sigma_obj.log()
        # log_prob_feas = _compute_log_prob_feas(self, means=means, sigmas=sigmas)
        print(mean_obj)
        return mean_obj # log_ei + log_prob_feas


# fit GP model
# create ExpectedImprovement class           single outcome for primary outcome
# create UpperConfidenceBound class          single outcome for primary outcome
# create ProbabilityOfImprovement class      single outcome for primary ountcome
# use _compute_log_prob_feas                    

# fit the models with the moo