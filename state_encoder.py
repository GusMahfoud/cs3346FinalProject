# state_encoder.py
import numpy as np
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.move import Move

from team_pool_loader import SPECIES_STATS    # hp/atk/def/spa/spd/spe true stats


# ============================================================
# FIXED POKEMON ORDER (MUST MATCH YOUR POOL)
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
# TRUE STATS (no EV/IV/nature dependency)
# ============================================================
def get_true_stats(p: Pokemon):
    if p is None or p.species is None:
        return {"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
    return SPECIES_STATS.get(
        _norm(p.species),
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
# MOVE CATEGORIES + NON-DAMAGE EFFECTS
# ============================================================
SETUP_MOVES = {
    "swordsdance", "nastyplot", "quiverdance",
    "calmmind", "bellydrum", "torchsong"
}

RECOVERY_MOVES = {"recover", "slackoff", "roost", "moonlight", "morningsun"}
PIVOT_MOVES = {"uturn", "voltswitch", "flipturn", "partingshot"}
PROTECT_MOVES = {"protect"}


# ============================================================
# PER-MOVE ENCODING (10 dims per move)
# ============================================================
def encode_move(m: Move, user: Pokemon, opp: Pokemon):
    """
    10-dim move vector:
    [
      base_power_norm,
      STAB_flag,
      effectiveness_norm,
      priority_flag,
      setup_flag,
      status_flag,
      secondary_flag,
      recovery_flag,
      pivot_flag,
      protect_flag
    ]
    """
    if m is None:
        return [0]*10

    # Base power normalization
    bp_norm = min((m.base_power or 0) / 150, 1.0)

    # STAB
    try:
        stab = 1 if m.type in {user.type_1, user.type_2} else 0
    except:
        stab = 0

    # Type effectiveness
    opp_types = (opp.type_1, opp.type_2) if opp else (None, None)
    try:
        eff = m.type.damage_multiplier(*opp_types)
    except:
        eff = 1.0
    eff_norm = min(eff / 4.0, 1.0)

    # Flags
    priority = 1 if m.priority > 0 else 0
    is_setup = 1 if m.id in SETUP_MOVES else 0
    is_status = 1 if m.category.name == "STATUS" else 0
    has_secondary = 1 if getattr(m, "secondary", None) else 0
    recovery = 1 if m.id in RECOVERY_MOVES else 0
    pivot = 1 if m.id in PIVOT_MOVES else 0
    protect = 1 if m.id in PROTECT_MOVES else 0

    return [
        bp_norm, stab, eff_norm, priority,
        is_setup, is_status, has_secondary,
        recovery, pivot, protect
    ]


# A Pokémon has up to 4 moves → 4 × 10 = 40 dims
def encode_moveset(user: Pokemon, opp: Pokemon):
    moves = list(user.moves.values()) if (user and user.moves) else []
    vec = []
    for i in range(4):
        mv = moves[i] if i < len(moves) else None
        vec.extend(encode_move(mv, user, opp))
    return vec


# ============================================================
# BOOST ENCODING (atk/def/spa/spd/spe)
# ============================================================
def boost_norm(p: Pokemon, name: str):
    if p is None:
        return 0.0
    b = p.boosts.get(name, 0)
    return (b + 6) / 12.0


# ============================================================
# BENCH ENCODING (48 dims per mon)
# ============================================================
def bench_list(team, active):
    bench = [p for p in team.values() if p is not active]
    return sorted(
        bench,
        key=lambda p: SPECIES_INDEX.get(_norm(p.species), 999)
    )


def encode_bench(team, active, opp):
    """
    Per bench Pokémon:
      species_one_hot (6)
      hp_bucket (1)
      speed_bucket (1)
      moveset (40)

    48 dims per mon × 5 mons = 240 dims
    """
    vec = []
    mons = bench_list(team, active)

    for p in mons:
        stats = get_true_stats(p)
        vec.extend(species_one_hot(p))
        vec.append(hp_bucket(p))
        vec.append(speed_bucket(stats["spe"]))
        vec.extend(encode_moveset(p, opp))

    # pad to 5 mons × 48 dims
    while len(vec) < 48 * 5:
        vec.extend([0]*48)

    return vec


# ============================================================
# ALIVE FLAGS
# ============================================================
def encode_alive_flags(battle):
    my = [0]*len(POKEMON_ORDER)
    opp = [0]*len(POKEMON_ORDER)

    for p in battle.team.values():
        sid = SPECIES_INDEX.get(_norm(p.species))
        if sid is not None:
            my[sid] = 0 if p.fainted else 1

    for p in battle.opponent_team.values():
        sid = SPECIES_INDEX.get(_norm(p.species))
        if sid is not None:
            opp[sid] = 0 if p.fainted else 1

    return my + opp


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
# MAIN STATE ENCODER
# ============================================================
def encode_state(battle):
    my = battle.active_pokemon
    opp = battle.opponent_active_pokemon

    my_stats = get_true_stats(my)
    opp_stats = get_true_stats(opp)

    vec = []

    # --- 1. Species identity ---
    vec.extend(species_one_hot(my))
    vec.extend(species_one_hot(opp))

    # --- 2. HP + Speed ---
    vec.append(hp_bucket(my))
    vec.append(hp_bucket(opp))
    vec.append(speed_bucket(my_stats["spe"]))
    vec.append(speed_bucket(opp_stats["spe"]))
    vec.append(1 if my_stats["spe"] > opp_stats["spe"] else 0)

    # --- 3. Movesets (active mon + opponent mon) ---
    vec.extend(encode_moveset(my, opp))
    vec.extend(encode_moveset(opp, my))

    # --- 4. Boosts (my + opp) ---
    for stat in ("atk", "def", "spa", "spd", "spe"):
        vec.append(boost_norm(my, stat))
    for stat in ("atk", "def", "spa", "spd", "spe"):
        vec.append(boost_norm(opp, stat))

    # --- 5. Bench (mine + opponent) ---
    vec.extend(encode_bench(battle.team, my, opp))
    vec.extend(encode_bench(battle.opponent_team, opp, my))

    # --- 6. Alive flags ---
    vec.extend(encode_alive_flags(battle))

    # --- 7. Turn bucket ---
    vec.append(turn_bucket(battle.turn))

    return np.array(vec, dtype=np.float32)
