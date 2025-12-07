# team_pool_loader.py
import json
from poke_env.data import GenData

from stat_calc import compute_full_stats
GEN9_DATA = GenData.from_gen(9)
POKEDEX = GEN9_DATA.pokedex
POOL_PATH = "teams/team_pool.json"


def _normalize(name: str) -> str:
    """Normalize species name to POKEDEX key format."""
    return name.lower().replace(" ", "").replace("-", "")


def build_species_stats(path: str = POOL_PATH):
    """
    Returns a dict:
        species_id -> {"atk","def","spa","spd","spe","hp"}
    using your team_pool.json + POKEDEX + stat_calc.
    """
    with open(path, "r") as f:
        pool = json.load(f)

    species_stats = {}

    for mon in pool:
        species_id = _normalize(mon["species"])
        dex_entry = POKEDEX[species_id]  # raises if typo, which is good

        base_stats = dex_entry["baseStats"]

        full_stats = compute_full_stats(
            base_stats=base_stats,
            evs=mon["evs"],
            ivs=mon["ivs"],
            nature=mon["nature"],
        )

        species_stats[species_id] = full_stats

    return species_stats

# Build and export full species stat table at import time
SPECIES_STATS = build_species_stats()
