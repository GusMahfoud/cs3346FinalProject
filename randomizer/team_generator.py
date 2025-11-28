import json
import random


def load_pool(path="team_pool.json"):
    """Load your 20-Pokémon JSON pool."""
    with open(path, "r") as f:
        return json.load(f)


def sample_team(pool, size=6, seed=None):
    """Pick size unique Pokémon from the pool."""
    if seed is not None:
        random.seed(seed)
    return random.sample(pool, size)


def format_team_for_showdown(team):
    """Convert Pokémon dicts into a valid Showdown importable team string."""
    lines = []

    for mon in team:
        # Header line
        lines.append(f"{mon['species']} @ {mon['item']}")
        lines.append(f"Ability: {mon['ability']}")

        # EVs
        evs = mon["evs"]
        ev_parts = [f"{value} {stat.upper()}" for stat, value in evs.items() if value > 0]
        if ev_parts:
            lines.append(f"EVs: {' / '.join(ev_parts)}")

        # Nature
        lines.append(f"{mon['nature']} Nature")

        # IVs (only list non-31)
        ivs = mon["ivs"]
        iv_parts = [f"{value} {stat.upper()}" for stat, value in ivs.items() if value != 31]
        if iv_parts:
            lines.append(f"IVs: {' / '.join(iv_parts)}")

        # Moves
        for move in mon["moves"]:
            lines.append(f"- {move}")

        lines.append("")  # blank line between Pokémon

    return "\n".join(lines).strip()


def generate_two_random_teams(path="team_pool.json", seed=None):
    """Load pool → generate Team A + Team B → return formatted strings."""
    pool = load_pool(path)

    teamA = sample_team(pool, 6, seed)
    teamB = sample_team(pool, 6, None if seed is None else seed + 1)

    teamA_str = format_team_for_showdown(teamA)
    teamB_str = format_team_for_showdown(teamB)

    return teamA_str, teamB_str


if __name__ == "__main__":
    teamA, teamB = generate_two_random_teams()

    print("\n===== TEAM A =====\n")
    print(teamA)

    print("\n===== TEAM B =====\n")
    print(teamB)
