import json
import random

def format_evs(evs):
    parts = []
    if evs["hp"] > 0: parts.append(f'{evs["hp"]} HP')
    if evs["atk"] > 0: parts.append(f'{evs["atk"]} Atk')
    if evs["def"] > 0: parts.append(f'{evs["def"]} Def')
    if evs["spa"] > 0: parts.append(f'{evs["spa"]} SpA')
    if evs["spd"] > 0: parts.append(f'{evs["spd"]} SpD')
    if evs["spe"] > 0: parts.append(f'{evs["spe"]} Spe')
    return " / ".join(parts)

def format_ivs(ivs):
    return (
        f'{ivs["hp"]} HP / {ivs["atk"]} Atk / {ivs["def"]} Def / '
        f'{ivs["spa"]} SpA / {ivs["spd"]} SpD / {ivs["spe"]} Spe'
    )

def mon_to_showdown(mon):
    lines = []
    lines.append(f'{mon["species"]}')
    lines.append(f'Ability: {mon["ability"]}')
    lines.append(f'{mon["nature"]} Nature')

    ev_line = format_evs(mon["evs"])
    if ev_line:
        lines.append(f'EVs: {ev_line}')

    lines.append(f'IVs: {format_ivs(mon["ivs"])}')

    for m in mon["moves"]:
        lines.append(f'- {m}')

    return "\n".join(lines)

def load_pool(json_path="teams/team_pool.json"):
    with open(json_path, "r") as f:
        return json.load(f)


# ============================================================
#  FUNCTION 1 — RANDOM LEAD
# ============================================================

def generate_random_lead_team(pool):
    """
    Always uses the same 6 Pokémon from the pool,
    but randomly shuffles which one appears first.
    Remaining 5 are kept in original pool order.
    """
    if len(pool) < 6:
        raise ValueError("Team pool must contain at least 6 Pokémon.")

    # fixed team list (first 6 entries exactly)
    fixed_team = list(pool[:6])

    # pick one to be the lead
    lead_index = random.randint(0, 5)

    # reorder: chosen lead first, rest in original order
    reordered = [fixed_team[lead_index]] + [
        fixed_team[i] for i in range(6) if i != lead_index
    ]

    return "\n\n".join(mon_to_showdown(mon) for mon in reordered)


# ============================================================
#  FUNCTION 2 — FIXED TEAM (NO RANDOMIZATION)
# ============================================================

def generate_fixed_team(pool):
    """
    Returns the SAME team order every time.
    No randomization, fully deterministic.
    """
    if len(pool) < 6:
        raise ValueError("Team pool must contain at least 6 Pokémon.")

    team = pool[:6]  # unchanged order

    return "\n\n".join(mon_to_showdown(mon) for mon in team)
