NATURE_MODIFIERS = {
    # atk, def, spa, spd, spe
    "Adamant": (1.1, 1.0, 0.9, 1.0, 1.0),
    "Modest":  (1.0, 1.0, 1.1, 1.0, 0.9),
    "Jolly":   (1.0, 1.1, 1.0, 1.0, 0.9),
    "Timid":   (1.0, 1.0, 1.1, 1.0, 0.9),
    "Bold":    (0.9, 1.1, 1.0, 1.0, 1.0),
    "Calm":    (0.9, 1.0, 1.0, 1.1, 1.0),
    # Add more if needed
}

def compute_stat(base, iv, ev, level, nature_mod):
    if base == 1: return 1  # Shedinja
    stat = ((2 * base + iv + (ev // 4)) * level) // 100 + 5
    return int(stat * nature_mod)

def compute_hp_stat(base, iv, ev, level):
    if base == 1: return 1
    return ((2 * base + iv + (ev // 4)) * level) // 100 + 110

def compute_full_stats(base_stats, evs, ivs, nature):
    level = 100  # You can change this to 50 consistently
    nature_mods = NATURE_MODIFIERS.get(nature, (1,1,1,1,1))

    atk = compute_stat(base_stats["atk"], ivs["atk"], evs["atk"], level, nature_mods[0])
    defe = compute_stat(base_stats["def"], ivs["def"], evs["def"], level, nature_mods[1])
    spa = compute_stat(base_stats["spa"], ivs["spa"], evs["spa"], level, nature_mods[2])
    spd = compute_stat(base_stats["spd"], ivs["spd"], evs["spd"], level, nature_mods[3])
    spe = compute_stat(base_stats["spe"], ivs["spe"], evs["spe"], level, nature_mods[4])
    hp = compute_hp_stat(base_stats["hp"], ivs["hp"], evs["hp"], level)

    return {
        "atk": atk,
        "def": defe,
        "spa": spa,
        "spd": spd,
        "spe": spe,
        "hp": hp,
    }
