import time
from pymoo.visualization.scatter import Scatter
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.problems.functional import FunctionalProblem
import numpy as np
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils import standardize
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
import torch
from botorch.acquisition.analytic import UpperConfidenceBound, ExpectedImprovement, ProbabilityOfImprovement
import plotly.io as pio

# Ax uses Plotly to produce interactive plots. These are great for viewing and analysis,
# though they also lead to large file sizes, which is not ideal for files living in GH.
# Changing the default to `png` strips the interactive components to get around this.
pio.renderers.default = "png"

# from botorch.acquisition.monte_carlo import MCAcquisitionFunction

'''
class qScalarizedUpperConfidenceBound(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        beta: Tensor,
        weights: Tensor,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super(MCAcquisitionFunction, self).__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
        self.sampler = sampler
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("weights", torch.as_tensor(weights))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate scalarized qUCB on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            Tensor: A `(b)`-dim Tensor of Upper Confidence Bound values at the
                given design points `X`.
        """
        print(X)
        posterior = self.model.posterior(X)
        samples = self.get_posterior_samples(posterior)  # n x b x q x o
        scalarized_samples = samples.matmul(self.weights)  # n x b x q
        mean = posterior.mean  # b x q x o
        scalarized_mean = mean.matmul(self.weights)  # b x q
        ucb_samples = (
            scalarized_mean
            + math.sqrt(self.beta * math.pi / 2)
            * (scalarized_samples - scalarized_mean).abs()
        )
        print(ucb_samples.max(dim=-1)[0].mean(dim=0))
        return ucb_samples.max(dim=-1)[0].mean(dim=0)
    '''


X_SZ = 8
# generate synthetic data
X = torch.rand(40, X_SZ)
# print()
ub = torch.max(X, 0).values.detach().numpy()
lb = torch.min(X, 0).values.detach().numpy()
# torch.stack([torch.sin(X[:, 0]), torch.cos(X[:, 1])], -1)
Y = torch.sin(X).sum(dim=1, keepdim=True)
Y = standardize(Y)  # standardize to zero mean unit variance

gp = SingleTaskGP(X, Y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

UCB = UpperConfidenceBound(gp, beta=0.1)
PI = ProbabilityOfImprovement(gp, best_f=max(Y))
EI = ExpectedImprovement(gp, best_f=max(Y))

obs = [lambda x: -1 * UCB(x), lambda x: -1 * EI(x), lambda x: -1 * PI(x)]


class MyFunctionalProblem(FunctionalProblem):

    # minimize LCB(x), −PI(x), −EI(x).

    def _evaluate(self, x, out, *args, **kwargs):
        # print(type(x))
        # print(x)
        # x = x
        # print(type(x))
        xt = torch.from_numpy(np.float32(np.expand_dims(x, axis=0)))
        # print(self.constr_eq, self.constr_ieq, self.xl)
        out["F"] = np.array([obj(xt).detach().numpy() for obj in self.objs])
        out["G"] = np.array([constr(x) for constr in self.constr_ieq])
        out["H"] = np.array([constr(x) for constr in self.constr_eq])


class MyProblem(Problem):
    def __init__(self, n_var,
                 objs, **kwargs):
        self.objs = objs
        super().__init__(n_var=n_var, n_obj=len(objs), **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        # print(x)
        # print(type(x))
        xt = torch.from_numpy(np.float32(np.expand_dims(x, axis=1)))
        # print(xt)
        out["F"] = np.array([obj(xt).detach().numpy() for obj in self.objs])


problem = MyProblem(X_SZ, obs, xl=lb, xu=ub)

algorithm = NSGA2(pop_size=100)

st = time.time()

res = minimize(problem,
               algorithm,
               ('n_gen', 400),
               seed=1,
               verbose=False)

st2 = time.time()


# print(res)
print(f"time nsga2: {st2-st}")

# fig, axs = plt.subplots()

print(f"pareto set size: {len(res.F)}")
print(res.X)
# print(res.pf)
# print(res.pop)

plot = Scatter()
plot.add(res.F, facecolor="green", edgecolor="red")
plot.show()
