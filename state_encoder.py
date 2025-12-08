# encoder.py — FINAL HIGH-RES SWITCHING VERSION
import numpy as np
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.move import Move
from computed_stats import get_real_stats


# ============================================================
# FIXED OUTPUT SIZE
# ============================================================
MAX_FEATURES = 1500

def finalize_vector(vec):
    """Pad/truncate to MAX_FEATURES so model always receives fixed length."""
    L = len(vec)
    if L < MAX_FEATURES:
        return np.array(vec + [0.0] * (MAX_FEATURES - L), dtype=np.float32)
    elif L > MAX_FEATURES:
        return np.array(vec[:MAX_FEATURES], dtype=np.float32)
    else:
        return np.array(vec, dtype=np.float32)


# ============================================================
# FIXED TEAM IDENTITY (STABLE ORDER)
# ============================================================
POKEMON_ORDER = [
    "dragapult","gholdengo","kingambit",
    "ironvaliant","weavile","skeledirge",
]
SPECIES_INDEX = {s: i for i, s in enumerate(POKEMON_ORDER)}

def _norm(n: str): return n.lower().replace(" ", "").replace("-", "")


# ============================================================
# TYPE ONE-HOT
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
# HP UTILS
# ============================================================
def hp_fraction(p: Pokemon):
    if p is None or not p.max_hp:
        return 0.0
    return (p.current_hp or 0) / p.max_hp

def hp_bucket_10(p: Pokemon):
    return min(int(hp_fraction(p) * 10) / 9.0, 1.0)


# ============================================================
# SPEED + STAT BUCKETS
# ============================================================
def speed_bucket(raw: int):
    if raw < 150: b = 0
    elif raw < 200: b = 1
    elif raw < 260: b = 2
    elif raw < 320: b = 3
    else: b = 4
    return b / 4.0


STAT_BUCKETS = {
    "atk":  [180, 260, 310],
    "def":  [200, 250, 300],
    "spa":  [180, 260, 330],
    "spd":  [180, 220, 260],
    "hp":   [300, 360]
}

def stat_bucket(statname, value):
    cuts = STAT_BUCKETS[statname]
    if value < cuts[0]: return 0
    elif value < cuts[1]: return 1
    elif len(cuts)==3 and value < cuts[2]: return 2
    else: return 3


# ============================================================
# DAMAGE ENGINE
# ============================================================
def stable_sigmoid(x):
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)

def estimate_damage(move: Move, user: Pokemon, opp: Pokemon):
    if move is None or user is None or opp is None:
        return 0, 0, 0, 0
    try:
        eff = move.type.damage_multiplier(opp.type_1, opp.type_2)
    except:
        eff = 1.0
    if eff == 0:
        return 0, 0, 0, 0

    ustats = get_real_stats(user.species)
    ostats = get_real_stats(opp.species)

    if move.category.name == "PHYSICAL":
        atk = ustats["atk"] * (2 + user.boosts["atk"]) / 2
        defense = ostats["def"] * (2 + opp.boosts["def"]) / 2
    elif move.category.name == "SPECIAL":
        atk = ustats["spa"] * (2 + user.boosts["spa"]) / 2
        defense = ostats["spd"] * (2 + opp.boosts["spd"]) / 2
    else:
        return 0, 0, 0, 0

    bp = move.base_power or 0
    raw = ((((((2*100)/5 + 2) * bp * atk / max(1, defense)) / 50) + 2) * eff)
    raw *= 0.96

    opp_hp = opp.current_hp or opp.max_hp or 1
    frac = raw / opp_hp

    # KO & 2HKO predictions
    x1 = 12 * (raw - opp_hp) / opp_hp
    x2 = 8 * ((2 * raw) - opp_hp) / opp_hp

    p_kill = stable_sigmoid(x1)
    p_2hko = stable_sigmoid(x2)

    return min(frac,1.0), raw, p_kill, p_2hko


# ============================================================
# MOVE ENCODING (20 dims)
# ============================================================
SETUP_MOVES = {
    "swordsdance","nastyplot","quiverdance","calmmind","bellydrum","torchsong"
}
RECOVERY_MOVES = {"recover","slackoff","roost","moonlight","morningsun"}
PIVOT_MOVES = {"uturn","voltswitch","flipturn","partingshot"}
PROTECT_MOVES = {"protect"}

def encode_move(m: Move, user: Pokemon, opp: Pokemon):
    if m is None:
        return [0]*20

    bp_norm = min((m.base_power or 0)/150, 1.0)
    stab = 1 if m.type in {user.type_1, user.type_2} else 0

    try: eff_raw = m.type.damage_multiplier(opp.type_1, opp.type_2)
    except: eff_raw = 1.0

    eff_norm = min(eff_raw/4.0, 1.0)
    is_immune = 1 if eff_raw == 0 else 0

    accuracy = (m.accuracy or 100) / 100
    priority = 1 if m.priority > 0 else 0

    is_setup = 1 if m.id in SETUP_MOVES else 0
    is_status = 1 if m.category.name == "STATUS" else 0
    has_secondary = 1 if getattr(m,"secondary",None) else 0
    recovery = 1 if m.id in RECOVERY_MOVES else 0
    pivot = 1 if m.id in PIVOT_MOVES else 0
    protect = 1 if m.id in PROTECT_MOVES else 0

    dmg_frac, raw_dmg, p_kill, p_2hko = estimate_damage(m, user, opp)

    return [
        bp_norm, stab, eff_norm, eff_raw/4.0, is_immune,
        accuracy, priority, is_setup, is_status, has_secondary,
        recovery, pivot, protect, is_immune,
        dmg_frac,
        raw_dmg/300,
        p_kill,
        p_2hko,
        1 if dmg_frac == 0 else 0
    ]

def encode_moveset(user, opp):
    moves = list(user.moves.values()) if user and user.moves else []
    out = []
    for i in range(4):
        mv = moves[i] if i < len(moves) else None
        out.extend(encode_move(mv, user, opp))
    return out


# ============================================================
# DANGER / THREAT
# ============================================================
def highest_expected_damage(attacker, defender):
    if attacker is None or defender is None:
        return 0
    best = 0
    for mv in attacker.moves.values():
        frac,_,_,_ = estimate_damage(mv, attacker, defender)
        best = max(best, frac)
    return best

def danger_score(my, opp):
    if my is None or opp is None:
        return 0,0
    incoming = highest_expected_damage(opp, my)
    dying = 1 if incoming >= hp_fraction(my) else 0
    return incoming, dying

def kill_threat_score(my, opp):
    if my is None or opp is None:
        return 0
    best = 0
    for mv in my.moves.values():
        _,_,p_kill,_ = estimate_damage(mv,my,opp)
        best = max(best, p_kill)
    return best


# ============================================================
# SWITCH-SCORING FOR BENCH MONS (8 NEW FEATURES)
# ============================================================
def bench_switch_features(p: Pokemon, opp: Pokemon, active: Pokemon):
    if p is None or opp is None:
        return [0]*8

    # 1. Best damage we can do
    best_offense = highest_expected_damage(p, opp)

    # 2. Best damage opponent can do to us
    best_defense = highest_expected_damage(opp, p)

    # 3. Simple matchup score (type offense minus defense)
    try:
        off = max(opp.damage_multiplier(t) for t in p.types if t)
    except:
        off = 1.0
    try:
        df = max(p.damage_multiplier(t) for t in opp.types if t)
    except:
        df = 1.0
    matchup = off - df

    # 4. Danger on switch-in
    danger_in, dying = danger_score(p, opp)

    # 5. Kill threat
    kt = kill_threat_score(p, opp)

    # 6. Speed advantage
    pst = get_real_stats(p.species)
    ost = get_real_stats(opp.species)
    speed_adv = 1 if pst["spe"] > ost["spe"] else 0

    # 7. Synergy with active (we use type synergy)
    if active:
        synergy = sum(
            int(active.damage_multiplier(t) < 1.0)
            for t in p.types if t
        )
    else:
        synergy = 0

    # 8. Predicted safe-in (1 if incoming dmg < 35%)
    safe = 1 if danger_in < 0.35 else 0

    return [
        best_offense,
        best_defense,
        matchup,
        danger_in,
        kt,
        speed_adv,
        synergy,
        safe
    ]


# ============================================================
# BENCH (115 features per Pokémon)
# ============================================================
def bench_list(team, active):
    return sorted(
        [p for p in team.values() if p is not active],
        key=lambda p: SPECIES_INDEX.get(_norm(p.species), 999)
    )

def encode_bench(team, active, opp):
    vec = []
    mons = bench_list(team, active)

    for p in mons:
        stats = get_real_stats(p.species)

        # Identity
        vec.extend(species_one_hot(p))
        vec.extend(type_one_hot(p))

        # HP + speed
        vec.append(hp_fraction(p))
        vec.append(hp_bucket_10(p))
        vec.append(speed_bucket(stats["spe"]))

        # Moveset (4 × 20 = 80)
        vec.extend(encode_moveset(p, opp))

        # NEW — switching intelligence (8)
        vec.extend(bench_switch_features(p, opp, active))

    # pad to always 5 × 115
    EXPECTED = 5 * 115
    if len(vec) < EXPECTED:
        vec.extend([0]*(EXPECTED-len(vec)))

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
            my[sid] = 1 if not p.fainted else 0

    for p in battle.opponent_team.values():
        sid = SPECIES_INDEX.get(_norm(p.species))
        if sid is not None:
            opp[sid] = 1 if not p.fainted else 0

    return my + opp


# ============================================================
# TURN BUCKET
# ============================================================
def turn_bucket(turn):
    if turn < 5: b=0
    elif turn <10: b=1
    elif turn <20: b=2
    else: b=3
    return b/3.0


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

    # HP
    vec.append(hp_fraction(my))
    vec.append(hp_bucket_10(my))
    vec.append(hp_fraction(opp))
    vec.append(hp_bucket_10(opp))

    # Speed
    vec.append(speed_bucket(my_stats["spe"]))
    vec.append(speed_bucket(opp_stats["spe"]))
    vec.append(1 if my_stats["spe"] > opp_stats["spe"] else 0)

    # Stat buckets
    for stat in ["atk","def","spa","spd","hp"]:
        vec.append(stat_bucket(stat, my_stats.get(stat,0))/3.0)
    for stat in ["atk","def","spa","spd","hp"]:
        vec.append(stat_bucket(stat, opp_stats.get(stat,0))/3.0)

    # Movesets (self + opponent)
    vec.extend(encode_moveset(my, opp))
    vec.extend(encode_moveset(opp, my))

    # Boosts
    for s in ["atk","def","spa","spd","spe"]:
        vec.append((my.boosts.get(s,0)+6)/12.0)
    for s in ["atk","def","spa","spd","spe"]:
        vec.append((opp.boosts.get(s,0)+6)/12.0)

    # Bench (5×115 each side)
    vec.extend(encode_bench(battle.team, my, opp))
    vec.extend(encode_bench(battle.opponent_team, opp, my))

    # Alive flags
    vec.extend(encode_alive_flags(battle))

    # Danger/kill threat
    dfrac, dying = danger_score(my, opp)
    vec.append(dfrac)
    vec.append(dying)
    vec.append(kill_threat_score(my, opp))

    # Turn
    vec.append(turn_bucket(battle.turn))

    return finalize_vector(vec)