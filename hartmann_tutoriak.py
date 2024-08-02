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
    _preprocess_constraint_bounds, convert_to_target_pre_hook)
import plotly.io as pio
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform
from botorch.acquisition.objective import ScalarizedPosteriorTransform, PosteriorTransform
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior_list import PosteriorList  # pragma: no cover
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from botorch.sampling.normal import SobolQMCNormalSampler

from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
MUL = 1E9

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

#p = Scatter()
#p.add(res.F, facecolor="red", edgecolor="none")
#p.show()

paramlist = [
    {
        "name": "M1_L",
        "type": "range",
        "value_type": "float",
        "bounds": [2.2E-8, 1E-6],
    },
    {
        "name": "M3_L",
        "type": "range",
        "value_type": "float",
        "bounds": [2.2E-8, 1E-6],
    },
    {
        "name": "M5_L",
        "type": "range",
        "value_type": "float",
        "bounds": [2.2E-8, 1E-6],
    },
    {
        "name": "M1_W",
        "type": "range",
        "value_type": "float",
        "bounds": [4.4E-8, 50E-6],
    },
    {
        "name": "M3_W",
        "type": "range",
        "value_type": "float",
        "bounds": [4.4E-8, 50E-6],
    },
    {
        "name": "M6_W",
        "type": "range",
        "value_type": "float",
        "bounds": [4.4E-8, 50E-6],
    },
    {
        "name": "M5_WM",
        "type": "range",
        "value_type": "int",
        "bounds": [1, 20],
    }
]

for i in paramlist:
    if "value_type" not in i or i["value_type"] == "float":
        t = i["bounds"]
        i["bounds"] = [j * MUL for j in t]

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
    def __init__(self, param_list):
        self.param_list = param_list
        self.setBounds()
        self.soboleng = torch.quasirandom.SobolEngine(dimension=len(param_list), seed=1)

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
    train_y_dim = train_y.shape[-1]
    model = SingleTaskGP(train_X=train_x, 
                         train_Y=train_y,#standardize(train_y),
                         input_transform=Normalize(d=train_X_dim),
                         outcome_transform=Standardize(m=train_y_dim)
                         )
    
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    if old_model is not None and old_model.state_dict() is not None:
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
        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])
    return train_x, train_y

def doMace(gp, pt, max_y, lb, ub, const):
    # max_y = max(ty[0])
    UCB = UpperConfidenceBound(gp, beta=0.1, posterior_transform=pt)
    PI = ProbabilityOfImprovement(gp, best_f=max_y, posterior_transform=pt)
    EI = ExpectedImprovement(gp, best_f=max_y, posterior_transform=pt)
    PF = ProbabilityOfFeasibility(gp, objective_index=0, constraints=const) # index, lower bound, upper bound.

    obs = [lambda x: -1 * UCB(x), lambda x: -1 * PI(x), lambda x: -1 * EI(x)]

    problem = MyProblem(len(pl), obs, PF, xl=lb, xu=ub)

    algorithm = NSGA2(pop_size=100)

    res = minimize(problem, algorithm, ('n_gen', 200), verbose=False, seed=1)
    return res

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
#print(STR_MEASURE)

subprocess.run(STR_NETLIST.split(" "), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
og_netlist = readFile(STR_NETFILE)

# ty = 2d array. 
def getBestTrial(_tx, _ty, _const):
    #print(_tx)
    old_shape_x = _tx.shape
    old_shape_y = _ty.shape
    _filter = torch.ones(len(_tx), dtype=torch.bool)
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

def doTrials(params):
    #print(params)
    for i in params:
        if i != "M5_WM":
            params[i] = params[i] / MUL
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
    if phs == None or math.isnan(phs): phs = 180
    return torch.tensor([gain, phs, gx])
    
torch.set_default_dtype(torch.float64)

N_SOBOL = 1024
N_MACE = 1024
trial = 0

SG = SobolGenerator(paramlist)   

tx = torch.tensor([[]])
ty = torch.tensor([[]])

const = {1: [-95, -0.01], 2:[45000000, None]}
bt = None
while trial < N_SOBOL or (trial > 512 and bt is not None):
    nx = SG.genValues()
    nx = convertType(nx, paramlist)
    #print(nx)
    nx_dict = listToParams(nx, paramlist)
    #nx = nx.unsqueeze(-2)
    ny = doTrials(nx_dict)
    #ny = nx
    #print(ny)
    #print(nx, ny)
    #print(tx, ty, nx, ny)
    #print(tx, ty, nx, ny)

    tx, ty = update_data(tx, ty, new_x = nx, new_y = ny)
    # mll, gp  = update_model(tx, ty, old_model=cur_model)
    #print(tx, ty)
    bt = getBestTrial(tx, ty, const)
    print("trial: ", trial, bt)
    print(nx, ny)
    #print(nx, ny)
    trial += 1
    
    if bt is not None:
        appendToResults("res1.txt", " ".join([str(i.item()) for i in bt[1]]) + "\n")
    else:
        appendToResults("res1.txt", "0" + "\n")

#print(tx)
#print(ty)

pl = paramlist
ub = []
lb = []

for i in pl:
    ub.append(i["bounds"][1])
    lb.append(i["bounds"][0])
# ["phs <= -0.01",  "phs >= -95", "gx >= 45000000"]
gp = None

NUM_OUTPUTS = 3
pt = ScalarizedPosteriorTransform(torch.cat([torch.tensor([1],
    dtype=torch.float32), torch.zeros(NUM_OUTPUTS-1, dtype=torch.float32)]))

print("MACE TIME.")
while trial < N_SOBOL + N_MACE:
    st0 = time.time()
    mll, gp  = update_model(tx, ty, old_model=gp)

    res = doMace(gp, pt, torch.max(ty[:, 0]), lb, ub, const)

    st1 = time.time()

    print(f"time setup: {st1-st0}")
    #print(res.X)
    
    nx = selectRandSamples(res.X, 10)
    nx = convertType(nx, paramlist)
    #print(nx)
    for i in nx:
        dct = listToParams(i, paramlist)
        #print(i)
        #print(dct)
        #doTrials(dct)
        #ny = torch.stack([torch.sin(Tensor(nx[:, 0])), torch.cos(Tensor(nx[:, 1]))], -1)
        ny = doTrials(dct)
        #print(i, ny)
        tx, ty = update_data(tx, ty, new_x = i, new_y = ny)
        bt = getBestTrial(tx, ty, const)
        print("trial: ", trial, bt)
        print(i, ny)
        
        appendToResults("res1.txt", " ".join([str(i.item()) for i in bt[0]] + [str(i.item()) for i in bt[1]]) + "\n")
        trial += 1

print(getBestTrial(tx, ty, const))