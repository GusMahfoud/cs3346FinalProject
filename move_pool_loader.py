from poke_env.data import GenData
from poke_env.battle.move import Move
import json, os

GEN9 = GenData.from_gen(9)

JSON_PATH = os.path.join(os.path.dirname(__file__), "teams/team_pool.json")

with open(JSON_PATH, "r") as f:
    RAW = json.load(f)

# Normalize species ("Iron Valiant" → "ironvaliant")
def normalize_species(name: str):
    return name.lower().replace(" ", "").replace("-", "")

# Normalize move name ("Shadow Ball" → "shadowball")
def normalize_move_name(name: str):
    return name.lower().replace(" ", "").replace("-", "")

# Build TEAM_POOL { "dragapult": ["Draco Meteor", ...], ... }
TEAM_POOL = {
    normalize_species(entry["species"]): entry["moves"]
    for entry in RAW
}

def get_species_moves(species: str):
    sid = normalize_species(species)
    move_names = TEAM_POOL.get(sid, [])

    real_moves = []
    for name in move_names:
        mid = normalize_move_name(name)

        # only accept moves that exist in Gen9
        if mid not in GEN9.moves:
            print(f"[WARN] Move not found in Gen9: {name} → {mid}")
            continue

        # 💥 Create real Move object
        real_moves.append(Move(mid, gen=9))

    return real_moves