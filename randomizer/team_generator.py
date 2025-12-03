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
    parts = []
    parts.append(f'{ivs["hp"]} HP')
    parts.append(f'{ivs["atk"]} Atk')
    parts.append(f'{ivs["def"]} Def')
    parts.append(f'{ivs["spa"]} SpA')
    parts.append(f'{ivs["spd"]} SpD')
    parts.append(f'{ivs["spe"]} Spe')
    return " / ".join(parts)

def mon_to_showdown(mon):
    s = []
    s.append(f'{mon["species"]} @ {mon["item"]}')
    s.append(f'Ability: {mon["ability"]}')
    s.append(f'{mon["nature"]} Nature')

    # EVs
    ev_line = format_evs(mon["evs"])
    if ev_line:
        s.append(f'EVs: {ev_line}')

    # IVs always included
    iv_line = format_ivs(mon["ivs"])
    s.append(f'IVs: {iv_line}')

    # Moves
    for m in mon["moves"]:
        s.append(f'- {m}')

    return "\n".join(s)

def load_pool(json_path="teams/team_pool.json"):
    with open(json_path, "r") as f:
        return json.load(f)

def generate_team_from_pool(pool, size=6):
    mons = random.sample(pool, size)
    return "\n\n".join(mon_to_showdown(mon) for mon in mons)
