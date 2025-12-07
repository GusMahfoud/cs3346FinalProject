import json
from stat_calc import compute_full_stats
from poke_env.data import GenData

GEN9_DATA = GenData.from_gen(9)

def normalize(name):
    return name.lower().replace(" ", "").replace("-", "")


# ------------------------------
# Load your pool
# ------------------------------
with open("teams/team_pool.json") as f:
    pool = json.load(f)

output = {}

# ------------------------------
# If pool is a LIST, convert to dict
# ------------------------------
if isinstance(pool, list):
    pool_entries = pool
elif isinstance(pool, dict):
    # convert dict-of-dicts into list
    pool_entries = [{
        "species": name,
        **data
    } for name, data in pool.items()]
else:
    raise TypeError("team_pool.json is neither a dict nor a list")


# ------------------------------
# Process each Pokémon entry
# ------------------------------
for entry in pool_entries:
    species_name = entry.get("species")
    if not species_name:
        print("WARNING: entry missing 'species':", entry)
        continue

    sid = normalize(species_name)

    # --- Pull base stats from official Gen 9 data ---
    if sid not in GEN9_DATA.pokedex:
        raise ValueError(f"Species ID {sid} not found in Gen 9 Pokedex")

    base = GEN9_DATA.pokedex[sid]["baseStats"]
    # base = {"hp":..., "atk":..., ...}

    evs = entry.get("evs", {"hp":0,"atk":0,"def":0,"spa":0,"spd":0,"spe":0})
    ivs = entry.get("ivs", {"hp":31,"atk":31,"def":31,"spa":31,"spd":31,"spe":31})
    nature = entry.get("nature", "Hardy")

    # --- Compute level-100 stats ---
    stats = compute_full_stats(base, evs, ivs, nature)

    output[sid] = stats


# ------------------------------
# Save computed stats
# ------------------------------
with open("computed_stats.json", "w") as f:
    json.dump(output, f, indent=2)

print("✓ computed_stats.json generated successfully!")
