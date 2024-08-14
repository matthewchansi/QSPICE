
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
        "bounds": [2.2E-8, 5E-6], 
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

import json

jstr = json.dumps(paramlist)

with open("paramlist.json", "w") as f:
    json.dump(paramlist, f, indent=4)