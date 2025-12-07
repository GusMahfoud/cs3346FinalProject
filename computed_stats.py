import json
import os

PATH = os.path.join(os.path.dirname(__file__), "computed_stats.json")

with open(PATH) as f:
    COMPUTED_STATS = json.load(f)

ZERO = {"hp":0,"atk":0,"def":0,"spa":0,"spd":0,"spe":0}

def get_real_stats(species):
    if species is None:
        return ZERO
    sid = species.lower().replace(" ", "").replace("-", "")
    return COMPUTED_STATS.get(sid, ZERO)
