# from numpy import cos, deg2rad
import math
import subprocess

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.modelbridge_utils import get_pending_observation_features
from ax.modelbridge.registry import ModelRegistryBase, Models

#FNAME_LOG = "5ota.log"
#FNAME_NET = "5ota.net"
#FNAME_ASC = "5ota.asc"
#PRINT_LOG = False
MUL = 1E9               # units in NANO

def db2dec(my_v):
    my_v = float(my_v)
    return 10**(my_v/20)

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

gs = GenerationStrategy(
    steps=[
        GenerationStep(
            model=Models.SOBOL,
            num_trials=1000,  # How many trials should be produced from this generation step
            max_parallelism=1,  # Max parallelism for this step
        ),
        GenerationStep(
            model=Models.BOTORCH_MODULAR,
            num_trials=-1,  # No limitation on how many trials should be produced from this step
            max_parallelism=1,  # Parallelism limit for this step, often lower than for Sobol
        ),
    ]
)

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

ax_client = AxClient(generation_strategy=gs)
print("client loaded!")

ax_client.create_experiment(
    name="transistor_opt",
    parameters=paramlist,
    objectives={"gain": ObjectiveProperties(minimize=False)},
    outcome_constraints=["phs <= -0.01", 
                         "phs >= -95", "gx >= 45000000"],  
)

def doTrials(params):
    for i in params:
        if i != "M5_WM":
            params[i] = params[i] / MUL
    
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
    
    return {"gain": (gain, 0.0), "gx": (gx, 0.0),
             "phs": (phs, 0.0)}
    '''
    new_netlist = setAnalysisMode(og_netlist, "op")
    writeFile(STR_NETFILE, new_netlist)
    subprocess.run(STR_SIMULATE.split(" "), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(STR_MEASURE.split(" "), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(fetchValue(readFile(STR_OUTFILE), "pow"))
    '''
    
for trial in range(1500):
    parameterization, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index=trial_index, raw_data=doTrials(parameterization))

    try:
        best_parameters, values = ax_client.get_best_parameters(use_model_predictions=False)
        means, covariances = values
        # log the values
        appendToResults("res.txt", " ".join([str(i) for i in means.values()]) + "\n")
        print(means)
        print(best_parameters)
        
    except:
        pass

ax_client.save_to_json_file()

best_parameters, values = ax_client.get_best_parameters()
means, covariances = values
# log the values
appendToResults("res.txt", " ".join([str(i) for i in means.values()]) + "\n")
print(means)
print(best_parameters)