# state_encoder.py
import numpy as np
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.move import Move
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.weather import Weather
from poke_env.battle.field import Field

from team_pool_loader import build_species_stats

# Precomputed stats from team_pool.json (open team sheets)
SPECIES_STATS = build_species_stats()


def _normalize_species(name: str) -> str:
    return name.lower().replace(" ", "").replace("-", "")


# ==============================================================================
#                               BUCKET HELPERS
# ==============================================================================

def bucket(val, thresholds):
    for i, t in enumerate(thresholds):
        if val < t:
            return float(i) / len(thresholds)
    return 1.0


def hp_bucket(p: Pokemon):
    if not p or not p.max_hp:
        return 0.0
    return bucket((p.current_hp or 0) / p.max_hp, [0.25, 0.5, 0.75])


def stat_bucket(value):
    return bucket(value, [120, 200, 300, 400])


def power_bucket(bp):
    return bucket(bp, [40, 70, 100, 150])


# ==============================================================================
#                     ROLE & MOVE CATEGORY IDENTIFICATION
# ==============================================================================

SETUP = {"swordsdance", "nastyplot", "quiverdance", "calmmind", "bellydrum"}
PIVOT = {"uturn", "voltswitch", "flipturn"}
HAZARD = {"stealthrock", "spikes"}
HAZARD_REMOVE = {"rapidspin", "defog", "courtchange"}
RECOVERY = {"recover", "slackoff", "painsplit", "roost"}
PHASING = {"whirlwind", "dragontail"}


def move_flags(m: Move):
    if not m:
        return [0, 0, 0, 0, 0, 0]
    mid = m.id
    return [
        int(mid in SETUP),
        int(mid in HAZARD),
        int(mid in HAZARD_REMOVE),
        int(mid in PIVOT),
        int(mid in RECOVERY),
        int(mid in PHASING),
    ]


def role_flags(stats):
    if not stats:
        return [0, 0, 0, 0, 0]
    return [
        int(stats["atk"] > stats["spa"]),   # physical lean
        int(stats["spa"] > stats["atk"]),   # special lean
        int(stats["def"] > 150),            # physical wall
        int(stats["spd"] > 150),            # special wall
        int(stats["spe"] > 120),            # fast
    ]


# ==============================================================================
#                DAMAGE ESTIMATION (FOR MATCHUP / BENCH VALUE)
# ==============================================================================

def simple_threat(attacker: Pokemon, defender: Pokemon):
    """Rough one-shot danger estimate using runtime move + type info."""
    if not attacker or not defender:
        return 0.0

    best = 0.0
    for m in attacker.moves.values():
        if not m or not m.base_power:
            continue

        # Type effectiveness via Showdown's type system
        try:
            mult = m.type.damage_multiplier(defender.type_1, defender.type_2)
        except Exception:
            mult = 1.0

        dmg = m.base_power * mult

        # STAB
        try:
            if m.type in {attacker.type_1, attacker.type_2}:
                dmg *= 1.5
        except Exception:
            pass

        if defender.max_hp:
            best = max(best, dmg / defender.max_hp)

    # bucket threat level
    return bucket(best, [0.25, 0.5, 0.75, 1.0])


# ==============================================================================
#                  ENCODE INDIVIDUAL POKEMON (ACTIVE OR BENCH)
# ==============================================================================

def encode_one_pokemon(p: Pokemon, opp_active: Pokemon):
    """
    Encode a Pokémon using precomputed species stats when available,
    falling back to Showdown's own stats. This avoids None-type issues.
    """
    if p is None:
        return [0.0] * 25  # fixed length fallback

    species = _normalize_species(p.species)

    # 1. Get stable stats
    if species in SPECIES_STATS:
        full_stats = SPECIES_STATS[species]
    else:
        # fall back to battle.stats if available, else zeros
        base_stats = p.stats or {}
        full_stats = {
            "atk": base_stats.get("atk", 0) or 0,
            "def": base_stats.get("def", 0) or 0,
            "spa": base_stats.get("spa", 0) or 0,
            "spd": base_stats.get("spd", 0) or 0,
            "spe": base_stats.get("spe", 0) or 0,
            "hp": (p.max_hp or 1),
        }

    # HP bucket from current battle state
    vec = [hp_bucket(p)]

    # 2. Stat buckets (using full_stats)
    vec.extend([
        stat_bucket(full_stats["atk"]),
        stat_bucket(full_stats["spa"]),
        stat_bucket(full_stats["def"]),
        stat_bucket(full_stats["spd"]),
        stat_bucket(full_stats["spe"]),
    ])

    # 3. Role flags
    vec.extend(role_flags(full_stats))

    # 4. Speed comparison
    try:
        opp_species = _normalize_species(opp_active.species)
        if opp_species in SPECIES_STATS:
            opp_stats = SPECIES_STATS[opp_species]
        else:
            opp_stats = opp_active.stats or {}
        faster = int(full_stats["spe"] > opp_stats.get("spe", 0))
    except Exception:
        faster = 0
    vec.append(faster)

    # 5. Matchup flags based on move types and typings
    has_SE = False
    opp_SE = False

    try:
        opp_t1, opp_t2 = opp_active.type_1, opp_active.type_2
    except Exception:
        opp_t1, opp_t2 = None, None

    # p → opp
    for m in p.moves.values():
        try:
            if m and m.type.damage_multiplier(opp_t1, opp_t2) > 1:
                has_SE = True
                break
        except Exception:
            pass

    # opp → p
    try:
        my_t1, my_t2 = p.type_1, p.type_2
    except Exception:
        my_t1, my_t2 = None, None

    if opp_active is not None:
        for m in opp_active.moves.values():
            try:
                if m and m.type.damage_multiplier(my_t1, my_t2) > 1:
                    opp_SE = True
                    break
            except Exception:
                pass

    vec.extend([int(has_SE), int(opp_SE)])

    # 6. One-shot threat (both directions) using runtime info
    vec.append(simple_threat(p, opp_active))
    vec.append(simple_threat(opp_active, p))

    # 7. Aggregate move-category flags from this mon's moves
    flags = [0] * 6
    for m in p.moves.values():
        mf = move_flags(m)
        flags = [max(a, b) for a, b in zip(flags, mf)]

    vec.extend(flags)

    # This vector length is 22; for bench we pad to 25.
    return vec


# ==============================================================================
#                  ENCODE THE ENTIRE BENCH (5 MONS)
# ==============================================================================

def encode_bench(battle):
    my_active = battle.active_pokemon
    bench_vec = []

    # non-active mons on our side
    bench = [p for p in battle.team.values() if p is not my_active]

    # deterministic order
    bench = sorted(bench, key=lambda p: p.species if p else "")

    for p in bench:
        bench_vec.extend(encode_one_pokemon(p, battle.opponent_active_pokemon))

    BENCH_SIZE = 5
    ENTRY_LEN = 25  # we pad each slot to 25 for fixed length
    while len(bench_vec) < BENCH_SIZE * ENTRY_LEN:
        bench_vec.extend([0.0] * ENTRY_LEN)

    return bench_vec


# ==============================================================================
#                           WEATHER / TERRAIN
# ==============================================================================

def encode_weather_and_terrain(battle):
    wvec = [0, 0, 0, 0]
    tvec = [0, 0, 0, 0]

    # Weather
    if battle.weather:
        try:
            w = battle.weather[0] if isinstance(battle.weather, tuple) else battle.weather
            if w == Weather.SUNNYDAY:
                wvec[0] = 1
            elif w == Weather.RAINDANCE:
                wvec[1] = 1
            elif w == Weather.SANDSTORM:
                wvec[2] = 1
            elif w == Weather.HAIL:
                wvec[3] = 1
        except Exception:
            pass

    # Terrain
    for f in battle.fields:
        if f == Field.GRASSY_TERRAIN:
            tvec[0] = 1
        elif f == Field.ELECTRIC_TERRAIN:
            tvec[1] = 1
        elif f == Field.MISTY_TERRAIN:
            tvec[2] = 1
        elif f == Field.PSYCHIC_TERRAIN:
            tvec[3] = 1

    return wvec + tvec


# ==============================================================================
#                            HAZARDS + TEAM HP
# ==============================================================================

def avg_team_hp(team):
    total = 0
    maxhp = 0
    for p in team.values():
        if p and p.max_hp:
            maxhp += p.max_hp
            total += p.current_hp or 0
    return (total / maxhp) if maxhp else 0.0


# ==============================================================================
#                            FULL STATE ENCODER
# ==============================================================================

def encode_state(battle):
    vec = []

    my = battle.active_pokemon
    opp = battle.opponent_active_pokemon

    # ACTIVE MONS
    vec.extend(encode_one_pokemon(my, opp))
    vec.extend(encode_one_pokemon(opp, my))

    # BENCH
    vec.extend(encode_bench(battle))

    # GLOBALS
    vec.extend(encode_weather_and_terrain(battle))

    # Hazards (SR + Spikes layers)
    vec.extend([
        int(SideCondition.STEALTH_ROCK in battle.side_conditions),
        battle.side_conditions.get(SideCondition.SPIKES, 0) / 3.0,
        int(SideCondition.STEALTH_ROCK in battle.opponent_side_conditions),
        battle.opponent_side_conditions.get(SideCondition.SPIKES, 0) / 3.0,
    ])

    # Team-level resources
    vec.append(avg_team_hp(battle.team))
    vec.append(avg_team_hp(battle.opponent_team))

    # Pokémon remaining alive
    my_alive = sum(1 for p in battle.team.values() if not p.fainted)
    opp_alive = sum(1 for p in battle.opponent_team.values() if not p.fainted)
    vec.extend([my_alive / 6.0, opp_alive / 6.0])

    # Turn bucket
    vec.append(bucket(battle.turn, [5, 10, 20, 30]))

    return np.array(vec, dtype=np.float32)
