# ============================================================
# encoder.py — FIXED VERSION (Correct Type Chart + Debugging)
# ============================================================

import numpy as np
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.move import Move
from poke_env.data.gen_data import GenData

from computed_stats import get_real_stats
from advanced_switcher import SwitchHeuristics

heuristic = SwitchHeuristics()

# ============================================================
# CONSTANTS & TYPE CHART
# ============================================================

GEN9_DATA = GenData.from_gen(9)
TYPE_CHART = GEN9_DATA.type_chart  # Mostly unused now, but kept for inspection

MAX_FEATURES = 450

POKEMON_ORDER = [
    "dragapult", "gholdengo", "kingambit",
    "ironvaliant", "weavile", "skeledirge",
]
SPECIES_INDEX = {name: i for i, name in enumerate(POKEMON_ORDER)}

TYPE_LIST = [
    "normal", "fire", "water", "electric", "grass", "ice",
    "fighting", "poison", "ground", "flying", "psychic",
    "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy"
]
TYPE_INDEX = {t: i for i, t in enumerate(TYPE_LIST)}

SETUP_MOVES = {"swordsdance", "nastyplot", "calmmind", "torchsong", "quiverdance", "bellydrum"}
RECOVERY_MOVES = {"recover", "slackoff", "roost", "moonlight", "morningsun"}
PIVOT_MOVES = {"uturn", "voltswitch", "flipturn", "partingshot"}
PROTECT_MOVES = {"protect"}


# ============================================================
# BASIC UTILITIES
# ============================================================

def _norm(species: str):
    return species.lower().replace("-", "").replace(" ", "")


def hp_fraction(p: Pokemon):
    if p is None or not p.max_hp:
        return 0.0
    return (p.current_hp or 0) / p.max_hp


def type_one_hot(p: Pokemon):
    out = [0] * len(TYPE_LIST)
    if p:
        for t in (p.type_1, p.type_2):
            if t:
                nm = t.name.lower()
                if nm in TYPE_INDEX:
                    out[TYPE_INDEX[nm]] = 1
                else:
                    print(f"[TYPE WARN] Unknown type {nm} in type_one_hot.")
    return out


# ============================================================
# DAMAGE ENGINE — Corrected with GEN9_DATA.damage_multiplier
# ============================================================

def stable_sigmoid(x):
    return 1 / (1 + np.exp(-x)) if x >= 0 else np.exp(x) / (1 + np.exp(x))


def estimate_damage(move: Move, user: Pokemon, opp: Pokemon):
    if move is None or user is None or opp is None:
        return (0, 0, 0, 0)

    # Guard against weird missing type
    try:
        mv_name = move.id.lower()

    except Exception as e:
        print(f"[DMG ERROR] Move has no type: {move} | {e}")
        return (0, 0, 0, 0)


    # ----------------------------------------------------------
    # TYPE MULTIPLIER — USE GEN9_DATA
    # ----------------------------------------------------------
    try:
        eff = move.type.damage_multiplier(opp.type_1, opp.type_2, type_chart=TYPE_CHART)
    except Exception as e:
        print(f"[DMG ERROR] damage_multiplier failed for {mv_name}: {e}")
        eff = 1.0


    if eff == 0:
    
        return (0, 0, 0, 0)

    # ----------------------------------------------------------
    # STATS + BOOSTS
    # ----------------------------------------------------------
    try:
        ustats = get_real_stats(user.species)
        ostats = get_real_stats(opp.species)
    except KeyError:
        print(f"[DMG ERROR] Missing stats for {user.species} or {opp.species}")
        return (0, 0, 0, 0)


    # Use .get on boosts so we never KeyError
    if move.category.name == "PHYSICAL":
        atk = ustats["atk"] * (2 + user.boosts.get("atk", 0)) / 2
        defense = ostats["def"] * (2 + opp.boosts.get("def", 0)) / 2
    elif move.category.name == "SPECIAL":
        atk = ustats["spa"] * (2 + user.boosts.get("spa", 0)) / 2
        defense = ostats["spd"] * (2 + opp.boosts.get("spd", 0)) / 2

    else:

        return (0, 0, 0, 0)

    # ----------------------------------------------------------
    # BASE POWER
    # ----------------------------------------------------------
    bp = move.base_power or 0
 

    if bp <= 0:
        return (0, 0, 0, 0)

    # ----------------------------------------------------------
    # RAW DAMAGE CALC
    # ----------------------------------------------------------
    raw = ((((((2 * 100) / 5 + 2) * bp * atk / max(1, defense)) / 50) + 2) * eff)
    raw *= 0.96

  

    # ----------------------------------------------------------
    # FRACTIONAL DAMAGE
    # ----------------------------------------------------------
    opp_hp = opp.current_hp or opp.max_hp or 1
    frac = raw / opp_hp


    x1 = 12 * (raw - opp_hp) / opp_hp
    x2 = 8 * ((2 * raw) - opp_hp) / opp_hp

    return min(frac, 1.0), raw, stable_sigmoid(x1), stable_sigmoid(x2)


# ============================================================
# MOVE ENCODING (Stable + Debugged)
# ============================================================

def encode_move(move: Move, user: Pokemon, opp: Pokemon):
    if move is None:
        return [0] * 31

    vec = []

    # ---------------- TYPE ONE-HOT ----------------
    type_vec = [0] * 18
    try:
        tname = move.type.name.lower()
        if tname in TYPE_INDEX:
            type_vec[TYPE_INDEX[tname]] = 1
        else:
            print(f"[MOVE WARN] Unknown move type {tname} for move {move.id}")
    except Exception as e:
        print(f"[MOVE ERROR] No move.type for {move.id}: {e}")
    vec.extend(type_vec)

    # ---------------- CATEGORY ONE-HOT ----------------
    cat = move.category.name.upper()
    vec.extend([
        1 if cat == "PHYSICAL" else 0,
        1 if cat == "SPECIAL" else 0,
        1 if cat == "STATUS" else 0,
    ])

    # ---------------- BASE METADATA ----------------
    vec.append(min((move.base_power or 0) / 150, 1.0))
    acc = move.accuracy if move.accuracy is not None else 100
    vec.append(acc / 100)
    vec.append(move.priority)

    # ---------------- FLAGS ----------------
    name = move.id.lower()
    vec.append(1 if getattr(move, "secondary", None) else 0)
    vec.append(1 if name in SETUP_MOVES else 0)
    vec.append(1 if name in PIVOT_MOVES else 0)
    vec.append(1 if name in RECOVERY_MOVES else 0)
    vec.append(1 if name in PROTECT_MOVES else 0)

    vec.extend([0] * 2)  # reserved

    # ---------------- EXPECTED DAMAGE ----------------
    dmg_frac, _, _, _ = estimate_damage(move, user, opp)
    vec.append(dmg_frac)

    return vec


# ============================================================
# MOVESET ENCODING — Mirroring Opponent Moves Correctly
# ============================================================

def encode_moveset_from_real_pokeenv_moves(species: str, my_team: dict, user: Pokemon, opp: Pokemon):
    norm_species = _norm(species)

    mirror = None
    for p in my_team.values():
        if p and _norm(p.species) == norm_species:
            mirror = p
            break

    source = user if mirror is None else mirror
    moves = list(source.moves.values())

    vec = []
    for i in range(4):
        mv = moves[i] if i < len(moves) else None
        vec.extend(encode_move(mv, user, opp))
    return vec


# ============================================================
# BENCH MATCHUP SCORE
# ============================================================

#def bench_matchup_score(mon: Pokemon, opp: Pokemon):
 #   if mon is None or opp is None:
  #      return 0.0
   # try:
   #     return heuristic.estimate_matchup(mon, opp)
   # except Exception as e:
    #    print(f"[BENCH WARN] estimate_matchup failed for {mon} vs {opp}: {e}")
     #   return 0.0


# ============================================================
# SINGLE MON BLOCK
# ============================================================

def encode_single_mon_block(mon: Pokemon, opp: Pokemon, is_active: bool, my_team: dict):
    # ---------------- ACTIVE MON BLOCK (≈120 dims) ----------------
    if is_active and mon is not None:
        try:
            stats = get_real_stats(mon.species)
        except KeyError:
            print(f"[ENCODE WARN] Missing stats for {mon.species}, using zeros.")
            stats = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "hp": 1}

        vec = []

        vec.extend(type_one_hot(mon))
        vec.append(1)                              # active flag
        vec.append(1 if not mon.fainted else 0)    # alive flag

        frac = hp_fraction(mon)
        vec.append(frac)
        vec.append(int(frac * 10) / 10)

        vec.append(stats["spe"] / 500 if stats["spe"] else 0.0)

        for s in ["atk", "def", "spa", "spd", "spe"]:
            boost_val = mon.boosts.get(s, 0)
            vec.append((boost_val + 6) / 12)

        vec.extend(encode_moveset_from_real_pokeenv_moves(mon.species, my_team, mon, opp))

        return vec[:120] + [0] * (120 - len(vec)) if len(vec) < 120 else vec[:120]

    # ---------------- BENCH BLOCK (very compact) ----------------
    if mon is None:
        return [0, 0, 0, 0]

    alive = 1 if not mon.fainted else 0
    hp = hp_fraction(mon)
    #matchup = bench_matchup_score(mon, opp)

    try:
        mon_stats = get_real_stats(mon.species)
    except KeyError:
        print(f"[ENCODE WARN] Missing stats for bench mon {mon.species}, using spe=0.")
        mon_stats = {"spe": 0}

    try:
        opp_stats = get_real_stats(opp.species) if opp else {"spe": 0}
    except KeyError:
        print(f"[ENCODE WARN] Missing stats for bench opp {opp.species}, using spe=0.")
        opp_stats = {"spe": 0}

    is_faster = 1 if mon_stats.get("spe", 0) > opp_stats.get("spe", 0) else 0

    return [alive, hp, is_faster]


# ============================================================
# TEAM ENCODING
# ============================================================

def encode_team(team_dict, active: Pokemon, opp: Pokemon, my_team):
    out = []
    for species in POKEMON_ORDER:
        mon = None
        for p in team_dict.values():
            if _norm(p.species) == species:
                mon = p
                break
        out.extend(encode_single_mon_block(mon, opp, mon is active, my_team))
    return out


# ============================================================
# MAIN ENCODER ENTRYPOINT
# ============================================================

def encode_state(battle):
    my = battle.active_pokemon
    opp = battle.opponent_active_pokemon
    my_team = battle.team

    vec = []

    vec.extend(encode_team(battle.team, my, opp, my_team))
    vec.extend(encode_team(battle.opponent_team, opp, my, my_team))

    # Scaled turn information
    vec.append(min(battle.turn / 30, 1.0))

    if len(vec) < MAX_FEATURES:
        vec.extend([0] * (MAX_FEATURES - len(vec)))

    return np.array(vec[:MAX_FEATURES], dtype=np.float32)
