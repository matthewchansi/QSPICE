import time
from hartmann_tutoriak import SobolGenerator, appendToResults, convertType, doMace, doTrials, getBestTrial, listToParams, selectRandSamples, update_data, update_model, doMaceEnsemble
import torch
import json
MUL = 1E9

N_SOBOL = 1024
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
'''
while trial < N_SOBOL:
    nx = SG.genValues()
    nx = convertType(nx, paramlist)
    #print(nx)
    nx_dict = listToParams(nx, paramlist)
    #nx = nx.unsqueeze(-2)
    ny = doTrials(nx_dict)

    tx, ty = update_data(tx, ty, new_x = nx, new_y = ny)
    # mll, gp  = update_model(tx, ty, old_model=cur_model)
    #print(tx, ty)
    bt = getBestTrial(tx, ty, const)
    print("SOBOL trial: ", trial, bt)
    print(nx, ny)
    #print(nx, ny)
    trial += 1
    
    if bt is not None:        
        appendToResults(f"res_{rs}_1024.txt", " ".join([str(i.item()) for i in bt[0]] + [str(i.item()) for i in bt[1]]) + "\n")
    else:
        appendToResults(f"res_{rs}_1024.txt", "0" + "\n")
'''
trial = N_SOBOL
fin = N_SOBOL

#torch.save(tx, f"tx_{rs}_{N_SOBOL}_mod.pt")
#torch.save(ty, f"ty_{rs}_{N_SOBOL}_mod.pt")
#exit()

tx = torch.load(f"tx_1234_{N_SOBOL}.pt")
ty = torch.load(f"ty_1234_{N_SOBOL}.pt")
#torch.cuda.empty_cache()
tx = tx.cuda()
ty = ty.cuda()

bt = getBestTrial(tx, ty, const)
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

print("MACE TIME.")
while trial - fin < N_MACE:
    st0 = time.time()
    mll, gp  = update_model(tx, ty, old_model=gp)

    #res = doMace(gp, torch.max(ty[:, 0]), lb, ub, const, randseed=rs)
    res = doMaceEnsemble(gp, torch.max(ty[:, 0]), lb, ub, const, randseed=rs)
    st1 = time.time()

    print(f"time setup: {st1-st0}")
    #print(res.X)
    
    nx = selectRandSamples(res.X, 64)
    #nx = res.X
    nx = convertType(nx, paramlist)
    #print(nx)
    for i in nx:
        dct = listToParams(i, paramlist)
        ny = doTrials(dct)
        #print(i, ny)
        tx, ty = update_data(tx, ty, new_x = i, new_y = ny)
        bt = getBestTrial(tx, ty, const)
        print("MACE trial:\t", trial)
        print("curr params:\t", i, ny)
        print("best params:\t", bt)
        #print("MACE trial: ", trial, bt)
        
        appendToResults(f"res_{rs}_nn_1024.txt", " ".join([str(i.item()) for i in bt[0]] + [str(i.item()) for i in bt[1]]) + "\n")
        trial += 1
    #print(tx.is_cuda)

print(getBestTrial(tx, ty, const))
torch.cuda.empty_cache()