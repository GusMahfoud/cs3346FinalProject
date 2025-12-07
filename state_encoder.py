# state_encoder.py
import numpy as np
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.move import Move

from team_pool_loader import SPECIES_STATS    # precomputed stats (hp, atk, def, spa, spd, spe)


# ============================================================
# FIXED POKEMON ORDER (must match your pool)
# ============================================================
POKEMON_ORDER = [
    "dragapult",
    "gholdengo",
    "kingambit",
    "ironvaliant",
    "weavile",
    "skeledirge",
]
SPECIES_INDEX = {name: i for i, name in enumerate(POKEMON_ORDER)}


# ------------------------------------------------------------
# Utility Normalizer
# ------------------------------------------------------------
def _norm(name: str) -> str:
    return name.lower().replace(" ", "").replace("-", "")


# ============================================================
# TRUE STATS LOOKUP (SAFE)
# ============================================================
def get_true_stats(pokemon):
    """
    Returns full stats from SPECIES_STATS (hp, atk, def, spa, spd, spe).
    Works for both ally + opponent Pokémon.
    Never depends on EVs/IVs/nature sent by Showdown.
    """

    if pokemon is None or pokemon.species is None:
        return {"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}

    sid = _norm(pokemon.species)

    return SPECIES_STATS.get(
        sid,
        {"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
    )


# ============================================================
# HP + SPEED BUCKETS
# ============================================================
def hp_bucket(p: Pokemon) -> float:
    if p is None or not p.max_hp:
        return 0.0

    frac = (p.current_hp or 0) / p.max_hp

    if frac <= 0.25: b = 0
    elif frac <= 0.50: b = 1
    elif frac <= 0.75: b = 2
    else: b = 3

    return b / 3.0


def speed_bucket(raw_speed: int) -> float:
    if raw_speed < 60: b = 0
    elif raw_speed < 100: b = 1
    elif raw_speed < 130: b = 2
    else: b = 3
    return b / 3.0


# ============================================================
# MOVESET CATEGORIES
# ============================================================
SETUP_MOVES = {
    "swordsdance", "nastyplot", "quiverdance", "calmmind",
    "bellydrum", "torchsong"
}

PRIORITY_MOVES = {
    "iceshard", "suckerpunch", "shadowsneak"
}


# ============================================================
# MOVESET SUMMARY
# ============================================================
def summarize_moves(mon: Pokemon, opp: Pokemon):
    """
    Returns:
      has_priority, has_setup, best_damage_ratio, has_SE
    """

    if mon is None or mon.moves is None:
        return [0, 0, 0.0, 0]

    has_prio = 0
    has_setup = 0
    has_se = 0
    best = 0.0

    opp_types = (opp.type_1, opp.type_2) if opp else (None, None)
    opp_hp = opp.max_hp if opp else 1

    for mv in mon.moves.values():
        if mv is None:
            continue

        mid = mv.id

        if mid in PRIORITY_MOVES:
            has_prio = 1

        if mid in SETUP_MOVES:
            has_setup = 1

        # damage approx if base_power exists
        if mv.base_power:
            mult = 1.0
            try:
                mult = mv.type.damage_multiplier(*opp_types)
            except:
                pass

            dmg = mv.base_power * mult

            # STAB
            try:
                if mv.type in {mon.type_1, mon.type_2}:
                    dmg *= 1.5
            except:
                pass

            best = max(best, min(dmg / max(opp_hp, 1), 1.0))

            if mult > 1.0:
                has_se = 1

    return [has_prio, has_setup, best, has_se]


# ============================================================
# ACTIVE PAIR ENCODING
# ============================================================
def encode_active_pair(my: Pokemon, opp: Pokemon):
    """
    Encodes:
      my_hp_bucket
      opp_hp_bucket
      my_speed_bucket
      opp_speed_bucket
      faster_flag
      my_moves_summary (4)
      opp_moves_summary (4)
      my_atk_boost
      my_spa_boost
      opp_atk_boost
      opp_spa_boost
    """

    my_stats = get_true_stats(my)
    opp_stats = get_true_stats(opp)

    my_speed = my_stats["spe"]
    opp_speed = opp_stats["spe"]

    faster = 1 if my_speed > opp_speed else 0

    vec = [
        hp_bucket(my),
        hp_bucket(opp),
        speed_bucket(my_speed),
        speed_bucket(opp_speed),
        faster,
    ]

    # moveset summaries
    vec.extend(summarize_moves(my, opp))
    vec.extend(summarize_moves(opp, my))

    # boosts
    def boost(mon, k):
        if mon is None:
            return 0.0
        stage = mon.boosts.get(k, 0)
        return (max(-6, min(6, stage)) + 6) / 12.0

    vec.append(boost(my, "atk"))
    vec.append(boost(my, "spa"))
    vec.append(boost(opp, "atk"))
    vec.append(boost(opp, "spa"))

    return vec


# ============================================================
# BENCH ENCODING
# ============================================================
def sorted_bench(team, active):
    mons = [p for p in team.values() if p is not active]

    def key_fn(p):
        if p is None:
            return 999
        return SPECIES_INDEX.get(_norm(p.species), 999)

    return sorted(mons, key=key_fn)


def encode_bench(team, active, opp):
    """
    For each bench mon:
      hp_bucket
      speed_bucket
      faster_than_opp
      moves_summary (4)
    Total = 7 features per mon, padded to 5 mons => 35 dims.
    """

    vec = []
    opp_stats = get_true_stats(opp)
    opp_speed = opp_stats["spe"]

    for p in sorted_bench(team, active):
        ps = get_true_stats(p)
        pspeed = ps["spe"]

        vec.extend([
            hp_bucket(p),
            speed_bucket(pspeed),
            1 if pspeed > opp_speed else 0,
        ])

        vec.extend(summarize_moves(p, opp))

    while len(vec) < 35:
        vec.extend([0] * 7)

    return vec


# ============================================================
# ALIVE ENCODING
# ============================================================
def encode_alive(battle):
    my_flags = [0] * len(POKEMON_ORDER)
    opp_flags = [0] * len(POKEMON_ORDER)

    for p in battle.team.values():
        idx = SPECIES_INDEX.get(_norm(p.species))
        if idx is not None:
            my_flags[idx] = 0 if p.fainted else 1

    for p in battle.opponent_team.values():
        idx = SPECIES_INDEX.get(_norm(p.species))
        if idx is not None:
            opp_flags[idx] = 0 if p.fainted else 1

    return my_flags + opp_flags


# ============================================================
# TURN BUCKET
# ============================================================
def turn_bucket(turn: int):
    if turn < 5: t = 0
    elif turn < 10: t = 1
    elif turn < 20: t = 2
    else: t = 3
    return t / 3.0


# ============================================================
# MAIN ENCODER
# ============================================================
def encode_state(battle):
    my = battle.active_pokemon
    opp = battle.opponent_active_pokemon

    vec = []
    vec.extend(encode_active_pair(my, opp))
    vec.extend(encode_bench(battle.team, my, opp))
    vec.extend(encode_bench(battle.opponent_team, opp, my))
    vec.extend(encode_alive(battle))
    vec.append(turn_bucket(battle.turn))

    return np.array(vec, dtype=np.float32)
