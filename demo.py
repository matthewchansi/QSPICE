import time
from hartmann_tutoriak import SobolGenerator, appendToResults, convertType, doTrials, getBestTrial, listToParams, selectRandSamples, update_data, update_model, doMaceEnsemble
import torch
import json
MUL = 1E9
torch.set_default_dtype(torch.float64)

N_SOBOL = 512
N_MACE = 256

rs = 1234
#randseed = rs
trial = 0

paramlist = []
with open("paramlist.json", "r") as f:
    paramlist = json.loads(f.read())

for i in paramlist:
    if "value_type" not in i or i["value_type"] == "float":
        t = i["bounds"]
        i["bounds"] = [j * MUL for j in t]

SG = SobolGenerator(paramlist, randseed=rs)   

tx = torch.tensor([[]])
ty = torch.tensor([[]])

const = {1: [-95, -0.01], 2:[45000000, None]}
bt = None

while trial < N_SOBOL:
    nx = SG.genValues()
    nx = convertType(nx, paramlist)
    nx_dict = listToParams(nx, paramlist)
    ny = doTrials(nx_dict)

    tx, ty = update_data(tx, ty, new_x = nx, new_y = ny)
    bt = getBestTrial(tx, ty, const)
    print("SOBOL trial: ", trial, bt)
    print(nx, ny)
    trial += 1
    
    if bt is not None:        
        appendToResults(f"res_{rs}_512.txt", " ".join([str(i.item()) for i in bt[0]] + [str(i.item()) for i in bt[1]]) + "\n")
    else:
        appendToResults(f"res_{rs}_512.txt", "0" + "\n")


torch.save(tx, f"tx_{rs}_{N_SOBOL}_mod.pt")
torch.save(ty, f"ty_{rs}_{N_SOBOL}_mod.pt")

trial = N_SOBOL
fin = N_SOBOL

tx = tx.cuda()
ty = ty.cuda()

bt = getBestTrial(tx, ty, const)

pl = paramlist
ub = []
lb = []

for i in pl:
    ub.append(i["bounds"][1])
    lb.append(i["bounds"][0])

gp = None

NUM_OUTPUTS = 3

print("MACE TIME.")
while trial - fin < N_MACE:
    st0 = time.time()
    mll, gp  = update_model(tx, ty, old_model=gp)

    res = doMaceEnsemble(gp, torch.max(ty[:, 0]), lb, ub, const, randseed=rs)
    st1 = time.time()

    print(f"time setup: {st1-st0}")
    
    nx = selectRandSamples(res.X, 64)
    nx = convertType(nx, paramlist)
    for i in nx:
        dct = listToParams(i, paramlist)
        ny = doTrials(dct)
        tx, ty = update_data(tx, ty, new_x = i, new_y = ny)
        bt = getBestTrial(tx, ty, const)
        print("MACE trial:\t", trial)
        print("curr params:\t", i, ny)
        print("best params:\t", bt)
        
        appendToResults(f"res_{rs}_DKL_512.txt", " ".join([str(i.item()) for i in bt[0]] + [str(i.item()) for i in bt[1]]) + "\n")
        trial += 1

print(getBestTrial(tx, ty, const))
torch.cuda.empty_cache()
