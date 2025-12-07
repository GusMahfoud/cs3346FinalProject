# team_pool_loader.py
import json
import os

POOL_PATH = "teams/team_pool.json"

def load_pool(path=POOL_PATH):
    with open(path) as f:
        return json.load(f)

def normalize(name: str):
    return name.lower().replace(" ", "").replace("-", "")
