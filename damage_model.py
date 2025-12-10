# damage_model.py

from poke_env.data import GenData
from computed_stats import get_real_stats
import numpy as np

# poke-env Move class
from poke_env.battle.move import Move


# ============================================================
# Helper: consistent normalization of species names
# ============================================================
def normalize_species(name: str):
    name = name.replace(", M", "").replace(", F", "")
    name = name.replace(",", "").strip().lower()
    name = name.replace(" ", "")
    return name


GEN9_DATA = GenData.from_gen(9)
TYPE_CHART = GEN9_DATA.type_chart


def stable_sigmoid(x):
    return 1 / (1 + np.exp(-x)) if x >= 0 else np.exp(x) / (1 + np.exp(x))


# ============================================================
# Convert a Showdown JSON dict → FakeMove (poke-env style)
# ============================================================


# ============================================================
# >>> FINAL FUNCTION: estimate_damage <<<
# Works with *real Move objects* and JSON move dictionaries
# ============================================================
def estimate_damage(move, user, opp):
    if move is None or user is None or opp is None:
        return (0, 0, 0, 0)

    
    # Must now be a poke-env Move or FakeMove
    try:
        mv_name = move.id.lower()
    except:
        print(f"[DMG ERROR] Could not get move.id for object: {move}")
        return (0, 0, 0, 0)

    # TYPE MULTIPLIER
    eff = 1.0
    try:
        eff = move.type.damage_multiplier(
            opp.type_1,
            opp.type_2,
            type_chart=TYPE_CHART
        )
    except Exception as e:
        print(f"[DMG ERROR] damage_multiplier failed for {mv_name}: {e}")

    if eff == 0:
        return (0, 0, 0, 0)

    # REAL STATS (from your computed stats table)
    try:
        user_key = normalize_species(user.species)
        opp_key = normalize_species(opp.species)

        ustats = get_real_stats(user_key)
        ostats = get_real_stats(opp_key)
    except KeyError:
        print(f"[DMG ERROR] Missing stats for species: {user.species}, {opp.species}")
        return (0, 0, 0, 0)

    # CATEGORY (PHYSICAL / SPECIAL)
    cat = move.category.name.upper()

    if cat == "PHYSICAL":
        atk = ustats["atk"] * (2 + user.boosts.get("atk", 0)) / 2
        defense = ostats["def"] * (2 + opp.boosts.get("def", 0)) / 2

    elif cat == "SPECIAL":
        atk = ustats["spa"] * (2 + user.boosts.get("spa", 0)) / 2
        defense = ostats["spd"] * (2 + opp.boosts.get("spd", 0)) / 2

    else:  # Status move → deal no damage
        return (0, 0, 0, 0)

    # BASE POWER
    bp = move.base_power or 0
    if bp <= 0:
        return (0, 0, 0, 0)

    # RAW DAMAGE ESTIMATE (smoothed)
    raw = ((((((2 * 100) / 5 + 2) * bp * atk / max(1, defense)) / 50) + 2) * eff)
    raw *= 0.96

    opp_hp = opp.current_hp or opp.max_hp or 1
    frac = raw / opp_hp

    x1 = 12 * (raw - opp_hp) / opp_hp
    x2 = 8 * ((2 * raw) - opp_hp) / opp_hp

    return min(frac, 1.0), raw, stable_sigmoid(x1), stable_sigmoid(x2)