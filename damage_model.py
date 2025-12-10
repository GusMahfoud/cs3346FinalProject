# damage_model.py

from poke_env.data import GenData
from computed_stats import get_real_stats
import numpy as np

GEN9_DATA = GenData.from_gen(9)
TYPE_CHART = GEN9_DATA.type_chart
def stable_sigmoid(x):
    return 1 / (1 + np.exp(-x)) if x >= 0 else np.exp(x) / (1 + np.exp(x))

def estimate_damage(move, user, opp):
    if move is None or user is None or opp is None:
        return (0, 0, 0, 0)

    # Guard against weird missing type
    try:
        mv_name = move.id.lower()
    except:
        return (0, 0, 0, 0)

    # TYPE MULTIPLIER
    try:
        eff = move.type.damage_multiplier(
            opp.type_1,
            opp.type_2,
            type_chart=TYPE_CHART
        )
    except Exception as e:
        print(f"[DMG ERROR] damage_multiplier failed for {mv_name}: {e}")
        eff = 1.0

    if eff == 0:
        return (0, 0, 0, 0)
    #print(f"[DMG] Effectiveness = {eff} for move {mv_name} against {opp.species}")
    # STATS
    try:
        ustats = get_real_stats(user.species)
        ostats = get_real_stats(opp.species)
    except KeyError:
        return (0, 0, 0, 0)

    if move.category.name == "PHYSICAL":
        atk = ustats["atk"] * (2 + user.boosts.get("atk", 0)) / 2
        defense = ostats["def"] * (2 + opp.boosts.get("def", 0)) / 2
    elif move.category.name == "SPECIAL":
        atk = ustats["spa"] * (2 + user.boosts.get("spa", 0)) / 2
        defense = ostats["spd"] * (2 + opp.boosts.get("spd", 0)) / 2
    else:
        return (0, 0, 0, 0)

    # DAMAGE FORMULA
    bp = move.base_power or 0
    #print(f"[DMG] Move: {mv_name}, BP: {bp}, Atk: {atk}, Def: {defense}, Eff: {eff}")
    if bp <= 0:
        return (0, 0, 0, 0)

    raw = ((((((2 * 100) / 5 + 2) * bp * atk / max(1, defense)) / 50) + 2) * eff)
    raw *= 0.96  # smoothing

    opp_hp = opp.current_hp or opp.max_hp or 1
    frac = raw / opp_hp
    #print(f"[DMG] Estimated raw damage: {raw} / {opp_hp} HP ({frac*100:.1f}%) move {mv_name} against {opp.species}"  )
    x1 = 12 * (raw - opp_hp) / opp_hp
    x2 = 8 * ((2 * raw) - opp_hp) / opp_hp

    return min(frac, 1.0), raw, stable_sigmoid(x1), stable_sigmoid(x2)