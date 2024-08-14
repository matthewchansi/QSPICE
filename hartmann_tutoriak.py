import subprocess
import math
import time
from typing import Dict, Union, Tuple, Optional
from botorch.optim.initializers import normalize
from pymoo.visualization.scatter import Scatter
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
import numpy as np
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils import standardize
from botorch.models import SingleTaskGP
from botorch.fit import SumMarginalLogLikelihood, fit_gpytorch_mll
import torch
from torch import Tensor
from botorch.acquisition.analytic import (UpperConfidenceBound, ExpectedImprovement, 
    ProbabilityOfImprovement, AnalyticAcquisitionFunction, _compute_log_prob_feas, 
    _preprocess_constraint_bounds, convert_to_target_pre_hook, LogConstrainedExpectedImprovement, ConstrainedExpectedImprovement, 
    _scaled_improvement, _ei_helper)
# import plotly.io as pio
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform
from botorch.acquisition.objective import ScalarizedPosteriorTransform, PosteriorTransform
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior_list import PosteriorList  # pragma: no cover
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.probability.utils import (
    ndtr as Phi,
)
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize

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
    
class EnsembleAcquisitionFunction(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        beta: Union[float, Tensor],
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
    ):
        r"""Single-outcome Expected Improvement (analytic).

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
            All sourced from:
            https://github.com/pytorch/botorch/blob/3f5fcd690d26b5930fe3b93f21144d7edb0318b2/botorch/acquisition/analytic.py#L35
        """
        #legacy_ei_numerics_warning(legacy_name=type(self).__name__)
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("beta", torch.as_tensor(beta))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Expected Improvement on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Expected Improvement values at the
            given design points `X`.
        """
        #print(X.shape)
        mean, sigma = self._mean_and_sigma(X)
        u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
        #print(Phi(u).shape)
        o = torch.stack((sigma * _ei_helper(u), Phi(u), (mean if self.maximize else -mean) + self.beta.sqrt() * sigma))
        return o

class MyProblemEnsemble(Problem):
    def __init__(self, n_var, n_obj,
                 my_fn, const_fn, **kwargs):
        self.my_fn = my_fn
        self.const_fn = const_fn
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        xt = torch.from_numpy(np.float32(np.expand_dims(x, axis=1)))
        const_mul = self.const_fn(xt)
        o = self.my_fn(xt) 

        o = o * const_mul * -1
        z = o.detach().numpy().T
        #print(z.shape)
        out["F"] = z

def doMaceEnsemble(gp, max_y, lb, ub, const, randseed = 1):
    
    pt = ScalarizedPosteriorTransform(torch.cat([torch.tensor([1],
        dtype=torch.float32), torch.zeros(gp.num_outputs -1, dtype=torch.float32)]))
    PF = ProbabilityOfFeasibility(gp, objective_index=0, constraints=const) # index, lower bound, upper bound.
    ensemble = EnsembleAcquisitionFunction(gp, best_f=max_y, beta = getBeta(dims = len(ub)), posterior_transform=pt)
    problem = MyProblemEnsemble(len(lb), 3, ensemble, PF, xl=lb, xu=ub)

    algorithm = NSGA2(pop_size=100)

    res = minimize(problem, algorithm, ('n_gen', 200), verbose=True, seed=randseed)
    return res

def selectRandSamples(batch, batchsz):
    B = torch.randperm(len(batch))
    return batch[B[:batchsz]]

def listToParams(val_list, param_list):
    o = {}
    for i in range(len(param_list)):
        o[param_list[i]["name"]] = val_list[i]
    return o

def convertType(val_list, param_list):
    val_list = torch.tensor(val_list)
    if val_list.dim() == 1:
        for i in range(len(param_list)):
            if param_list[i]["value_type"] in ["int", "integer"]:
                val_list[i] = torch.round(val_list[i])
        return val_list
    else:
        for i in range(len(param_list)):
            if param_list[i]["value_type"] in ["int", "integer"]:
                val_list[:, i] = torch.round(val_list[:, i])
        return val_list


class SobolGenerator():
    def __init__(self, param_list, randseed = 1):
        self.param_list = param_list
        self.setBounds()
        self.soboleng = torch.quasirandom.SobolEngine(dimension=len(param_list), seed=randseed)

    def setBounds(self):
        pl = self.param_list
        m = []
        lb = []
        for i in pl:
            m.append(i["bounds"][1]-i["bounds"][0])
            lb.append(i["bounds"][0])
        self.lb = torch.tensor(lb) 
        self.m = torch.tensor(m)

    def gen(self):
        s = self.soboleng.draw(1)
        #print(s)
        s = s.view(2)
        torch.addcmul(self.lb, self.m, s, out=s)
        out_d = {}
        for i in range(len(self.param_list)):
            cu = self.param_list[i]
            out_d[cu["name"]] = s[i]
        return out_d
    
    def genValues(self):
        s = self.soboleng.draw(1, dtype=torch.float64)#.view(2)
        #print(s)
        s = s.squeeze(0)
        #print(s)
        torch.addcmul(self.lb, self.m, s, out=s)
        return s

def update_model(train_x, train_y, old_model=None):
    train_X_dim = train_x.shape[-1]
    train_Y_dim = train_y.shape[-1]
    model = SingleTaskGP(train_X=train_x, 
                         train_Y=train_y,#standardize(train_y),
                         input_transform=Normalize(d=train_X_dim),
                         outcome_transform=Standardize(m=train_Y_dim)
                         )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    if old_model is not None:
        print("loading state dict")
        model.load_state_dict(old_model.state_dict())

    fit_gpytorch_mll(mll)
    
    return mll, model

def update_data(train_x, train_y, new_x=None, new_y=None):
    if train_x == None or train_y == None or train_x.nelement() + train_y.nelement() == 0 :
        if new_x.dim() == 1:
            new_x = new_x.unsqueeze(-2)
        if new_y.dim() == 1:
            new_y = new_y.unsqueeze(-2)
        train_x = new_x
        train_y = new_y
    elif new_x is not None and new_y is not None:
        #new_x = torch.tensor(new_x)
        #new_y = torch.tensor(new_y)
        if new_x.dim() == 1:
            new_x = new_x.unsqueeze(-2)
        if new_y.dim() == 1:
            new_y = new_y.unsqueeze(-2)
        #print(train_x.get_device(), new_x.get_device())
        train_x = torch.cat([train_x, new_x.to(train_x)])
        train_y = torch.cat([train_y, new_y.to(train_y)])
    return train_x, train_y

def getBeta(iters = 4, dims = 7):
    t = iters
    d = dims 
    v = 0.5
    delta = 0.05
    beta = math.sqrt(2 * v * math.log(pow(t, d/2 + 2) * pow(math.pi, 2) / (3 * delta)))
    return beta

def fetchValue(out_st, my_v):
    pfx = ".meas "
    l = out_st.strip().split("\n")
    bk = ""
    for i in range(len(l)):
        if (pfx + my_v) in l[i]:
            if l[i].startswith("Warning"):
                return None
            else:
                bk = l[i+1]
                break
    if bk == "":
        return None
    bk = [j.strip() for j in bk.replace("(", " ").replace(")", " ").split(",")]
    return float(bk[0])

def setAnalysisMode(net_st, modestr):
    modestr = modestr.lower()
    modes = ["op", "ac", "tran", "dc"]
    pfx = "."
    net_st_split = net_st.split("\n")
    for i_n in range(len(net_st_split)):
        i = net_st_split[i_n]
        for j in modes:
            if i.startswith(pfx+j) and j != modestr:
                # print(j, modestr, "\n\nAAAGHGUAGUAH")
                net_st_split[i_n] = "//" + i
    return "\n".join(net_st_split)
    
def readFile(fp):
    z = ""
    with open(fp, "r", encoding="cp1252") as f:
        z = f.read()
    return z

def writeFile(fp, content):
    with open(fp, "w", encoding="cp1252") as f:
        f.write(content)

def appendToResults(fname, t):
    with open (fname, "a") as f:
        f.write(t + " ")

    
def setParams(net_st, params):
    paramstr = " ".join([i+"="+str(params[i]) for i in params])
    net_st_split = net_st.split("\n")
    for i_n in range(len(net_st_split)):
        i = net_st_split[i_n]
        if i.startswith(".param"):
                # print(j, modestr, "\n\nAAAGHGUAGUAH")
            net_st_split[i_n] = ".param " + paramstr
    return "\n".join(net_st_split)


FNAME = "ota5_nmos_22n"
STR_NETLIST = "QUX -Netlist {}.qsch".format(FNAME)
STR_SIMULATE = "QSPICE64 -binary {}.cir".format(FNAME)
STR_MEASURE = "QPOST {}.cir -o {}.out".format(FNAME, FNAME)
STR_OUTFILE = "{}.out".format(FNAME)
STR_NETFILE = "{}.cir".format(FNAME)

subprocess.run(STR_NETLIST.split(" "), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
og_netlist = readFile(STR_NETFILE)

# ty = 2d array. 
def getBestTrial(_tx, _ty, _const):
    #print(_tx)
    old_shape_x = _tx.shape
    old_shape_y = _ty.shape
    _filter = torch.ones(len(_tx), dtype=torch.bool, device=_tx.get_device())
    for i in _const:
        # apply upper, lower bounds to each const.
        # if the bound is none, don't apply the bound.
        _lower, _upper = _const[i]

        if _upper is not None:
            _filter = _filter * torch.lt(_ty[:, i], _upper)
        if _lower is not None:
            _filter = _filter * torch.gt(_ty[:, i], _lower)
    _ty = _ty.masked_select(_filter.unsqueeze(-1))
    if len(_ty) == 0:
        return None
    #print(_ty)
    _ty = _ty.view((_ty.shape[0]//old_shape_y[1], old_shape_y[1])) 
    _tx = _tx.masked_select(_filter.unsqueeze(-1))
    _tx = _tx.view((_tx.shape[0]//old_shape_x[1], old_shape_x[1])) 
    #print(_ty)
    _mx = _ty.max(dim=0)
    best = (_tx[_mx.indices[0]], _ty[_mx.indices[0]])
    #print(best)
    return best

def doTrials(params, mul=1E9):
    #print(params)
    for i in params:
        if i != "M5_WM":
            params[i] = params[i] / mul
        params[i] = params[i].item()
    
    
    new_netlist = setAnalysisMode(og_netlist, "ac")
    new_netlist = setParams(new_netlist, params)
    writeFile(STR_NETFILE, new_netlist)
    subprocess.run(STR_SIMULATE.split(" "), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(STR_MEASURE.split(" "), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    res = readFile(STR_OUTFILE)
    gain = fetchValue(res, "gain")
    gx = fetchValue(res, "gx")
    phs = fetchValue(res, "phs")
    #print(phs)
    #print(res)

    if gain == None or math.isnan(gain): gain = 0
    if gx == None or math.isnan(gx): gx = 0
    if phs == None or math.isnan(phs): phs = -180
    return torch.tensor([gain, phs, gx])

torch.set_default_dtype(torch.float64)
