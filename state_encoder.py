# encoder.py
import numpy as np
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.move import Move

from computed_stats import get_real_stats


# ============================================================
# FIXED POKEMON ORDER
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

def _norm(name: str) -> str:
    return name.lower().replace(" ", "").replace("-", "")


# ============================================================
# TYPE ONE-HOT (18 types)
# ============================================================
TYPE_LIST = [
    "normal","fire","water","electric","grass","ice",
    "fighting","poison","ground","flying","psychic",
    "bug","rock","ghost","dragon","dark","steel","fairy"
]
TYPE_INDEX = {t: i for i, t in enumerate(TYPE_LIST)}

def type_one_hot(p: Pokemon):
    vec = [0] * len(TYPE_LIST)
    if p is None:
        return vec

    for t in (p.type_1, p.type_2):
        if t and t.name.lower() in TYPE_INDEX:
            vec[TYPE_INDEX[t.name.lower()]] = 1
    return vec


# ============================================================
# SPECIES ONE-HOT
# ============================================================
def species_one_hot(p: Pokemon):
    vec = [0] * len(POKEMON_ORDER)
    if p is None:
        return vec

    sid = SPECIES_INDEX.get(_norm(p.species))
    if sid is not None:
        vec[sid] = 1
    return vec


# ============================================================
# HP ENCODING
# ============================================================
def hp_fraction(p: Pokemon):
    if p is None or not p.max_hp:
        return 0.0
    return (p.current_hp or 0) / p.max_hp

def hp_bucket_10(p: Pokemon):
    frac = hp_fraction(p)
    return min(int(frac * 10) / 9.0, 1.0)


# ============================================================
# SPEED BUCKET (based on computed stats)
# ============================================================
def speed_bucket(raw: int):
    if raw < 100: b = 0
    elif raw < 200: b = 1
    elif raw < 300: b = 2
    else: b = 3
    return b / 3.0


# ============================================================
# MOVE FLAGS
# ============================================================
SETUP_MOVES = {
    "swordsdance","nastyplot","quiverdance",
    "calmmind","bellydrum","torchsong"
}
RECOVERY_MOVES = {"recover","slackoff","roost","moonlight","morningsun"}
PIVOT_MOVES = {"uturn","voltswitch","flipturn","partingshot"}
PROTECT_MOVES = {"protect"}


# ============================================================
# PER-MOVE ENCODING (14 dims)
# ============================================================
def encode_move(m: Move, user: Pokemon, opp: Pokemon):
    if m is None or user is None or opp is None:
        return [0] * 14

    # Base Power
    bp_norm = min((m.base_power or 0) / 150, 1.0)

    # STAB
    stab = 1 if m.type in {user.type_1, user.type_2} else 0

    # Effectiveness
    try:
        eff_raw = m.type.damage_multiplier(opp.type_1, opp.type_2)
    except:
        eff_raw = 1.0

    eff_norm = min(eff_raw / 4.0, 1.0)
    is_immune = 1 if eff_raw == 0 else 0

    # Accuracy
    accuracy = (m.accuracy or 100) / 100

    # Flags
    priority = 1 if m.priority > 0 else 0
    is_setup = 1 if m.id in SETUP_MOVES else 0
    is_status = 1 if m.category.name == "STATUS" else 0
    has_secondary = 1 if getattr(m, "secondary", None) else 0
    recovery = 1 if m.id in RECOVERY_MOVES else 0
    pivot = 1 if m.id in PIVOT_MOVES else 0
    protect = 1 if m.id in PROTECT_MOVES else 0

    # Failure prediction
    will_fail = 0

    # Dragon → Fairy immune
    if m.type.name == "Dragon":
        if (opp.type_1 and opp.type_1.name == "Fairy") or \
           (opp.type_2 and opp.type_2.name == "Fairy"):
            will_fail = 1

    # Wisp fails vs Fire or already statused
    if m.id == "willowisp":
        if (opp.type_1 and opp.type_1.name == "Fire") or (opp.type_2 and opp.type_2.name == "Fire"):
            will_fail = 1
        if opp.status is not None:
            will_fail = 1

    if is_immune:
        will_fail = 1

    return [
        bp_norm, stab, eff_norm, eff_raw/4.0, is_immune,
        accuracy, priority, is_setup, is_status, has_secondary,
        recovery, pivot, protect, will_fail
    ]


def encode_moveset(user: Pokemon, opp: Pokemon):
    mv_list = list(user.moves.values()) if user and user.moves else []
    vec = []
    for i in range(4):
        mv = mv_list[i] if i < len(mv_list) else None
        vec.extend(encode_move(mv, user, opp))
    return vec


# ============================================================
# BOOSTS
# ============================================================
def boost_norm(p: Pokemon, stat):
    if p is None:
        return 0.0
    return (p.boosts.get(stat, 0) + 6) / 12.0


# ============================================================
# BENCH ENCODING
# ============================================================
def bench_list(team, active):
    bench = [p for p in team.values() if p is not active]
    return sorted(bench, key=lambda p: SPECIES_INDEX.get(_norm(p.species), 999))

def encode_bench(team, active, opp):
    vec = []
    mons = bench_list(team, active)

    for p in mons:
        stats = get_real_stats(p.species)

        vec.extend(species_one_hot(p))
        vec.extend(type_one_hot(p))
        vec.append(hp_fraction(p))
        vec.append(hp_bucket_10(p))
        vec.append(speed_bucket(stats["spe"]))
        vec.extend(encode_moveset(p, opp))

    # pad to 5 * 87 dims
    while len(vec) < 5 * 87:
        vec.extend([0] * 87)

    return vec


# ============================================================
# ALIVE FLAGS
# ============================================================
def encode_alive_flags(battle):
    my = [0] * len(POKEMON_ORDER)
    opp = [0] * len(POKEMON_ORDER)

    for p in battle.team.values():
        sid = SPECIES_INDEX.get(_norm(p.species))
        if sid is not None:
            my[sid] = 1 if not p.fainted else 0

    for p in battle.opponent_team.values():
        sid = SPECIES_INDEX.get(_norm(p.species))
        if sid is not None:
            opp[sid] = 1 if not p.fainted else 0

    return my + opp


# ============================================================
# TURN BUCKET
# ============================================================
def turn_bucket(turn: int):
    if turn < 5: b = 0
    elif turn < 10: b = 1
    elif turn < 20: b = 2
    else: b = 3
    return b / 3.0


# ============================================================
# MAIN ENCODER
# ============================================================
def encode_state(battle):
    my = battle.active_pokemon
    opp = battle.opponent_active_pokemon

    my_stats = get_real_stats(my.species if my else None)
    opp_stats = get_real_stats(opp.species if opp else None)

    vec = []

    # Identity
    vec.extend(species_one_hot(my))
    vec.extend(type_one_hot(my))
    vec.extend(species_one_hot(opp))
    vec.extend(type_one_hot(opp))

    # HP & speed
    vec.append(hp_fraction(my))
    vec.append(hp_bucket_10(my))
    vec.append(hp_fraction(opp))
    vec.append(hp_bucket_10(opp))

    vec.append(speed_bucket(my_stats["spe"]))
    vec.append(speed_bucket(opp_stats["spe"]))
    vec.append(1 if my_stats["spe"] > opp_stats["spe"] else 0)

    # Movesets
    vec.extend(encode_moveset(my, opp))
    vec.extend(encode_moveset(opp, my))

    # Boosts
    for s in ["atk","def","spa","spd","spe"]:
        vec.append(boost_norm(my, s))
    for s in ["atk","def","spa","spd","spe"]:
        vec.append(boost_norm(opp, s))

    # Bench
    vec.extend(encode_bench(battle.team, my, opp))
    vec.extend(encode_bench(battle.opponent_team, opp, my))

    # Alive flags
    vec.extend(encode_alive_flags(battle))

    # Turn
    vec.append(turn_bucket(battle.turn))

    return np.array(vec, dtype=np.float32)
