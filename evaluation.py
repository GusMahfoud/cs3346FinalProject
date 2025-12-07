# evaluation.py
import asyncio
import os
import re

from poke_env.player.baselines import MaxBasePowerPlayer
from rl_agent import MyRLAgent
from showdown_server import start_showdown_server
from randomizer.team_generator import load_pool, generate_random_lead_team, generate_fixed_team
from fixed_ais import FixedOrderMaxBasePower
# ------------------------------------------------------------
# Directory for saving replays
# ------------------------------------------------------------
REPLAY_DIR = "eval_replays"
os.makedirs(REPLAY_DIR, exist_ok=True)


# ------------------------------------------------------------
# HTML TEMPLATE USING THE REAL SHOWDOWN REPLAY ENGINE
# ------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Replay {title}</title>
<script src="https://play.pokemonshowdown.com/js/replay-embed.js"></script>
</head>
<body>
<div style="width: 640px; margin: auto;">
    <div class="battle"></div>
</div>

<script>
var battle = new Battle({replay: true});
var log = `{LOG}`;
battle.instantAdd(log.split("\\n"));
document.querySelector('.battle').appendChild(battle.wrapper);
</script>
</body>
</html>
"""


# ------------------------------------------------------------
# Extract the battle log text (Showdown battle format)
# ------------------------------------------------------------
def extract_log_from_battle(battle):
    """
    poke-env stores the raw Showdown log text in:
       battle._replay or battle._battle_log
    depending on version.

    We defensively check both.
    """
    # Newer versions of poke-env
    if hasattr(battle, "replay") and battle.replay:
        return battle.replay

    # Older poke-env
    if hasattr(battle, "_replay") and battle._replay:
        return battle._replay

    # Even older fallback
    if hasattr(battle, "_battle_log") and battle._battle_log:
        return "\n".join(battle._battle_log)

    return None


# ------------------------------------------------------------
# Save HTML Replay
# ------------------------------------------------------------
def save_replay_html(log_text, battle_id):
    safe_id = re.sub(r"[^A-Za-z0-9_\\-]", "_", battle_id)
    filename = os.path.join(REPLAY_DIR, f"{safe_id}.html")

    html = HTML_TEMPLATE.replace("{LOG}", log_text).replace("{title}", safe_id)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

    return filename


# ------------------------------------------------------------
# Evaluation routine
# ------------------------------------------------------------
async def run_evaluation(n_battles=10):
    print(f"=== Running {n_battles} Evaluation Battles ===")

    pool = load_pool("teams/team_pool.json")
    team_rl = generate_fixed_team(pool)
    team_ai = generate_fixed_team(pool)

    print("\n=== FIXED TRAINING TEAMS ===")
    print(team_rl, "\n\n", team_ai)
    print("======================================\n")
    

    # Load trained agent
    rl_agent = MyRLAgent(
        battle_format="gen9ubers",
        model_folder="models/a2c_v5",
        team=team_rl,
        allow_switching=False,
    )

    # Opponent
    ai_agent = FixedOrderMaxBasePower(
        battle_format="gen9ubers",
        team=team_ai,
    )

    # Run evaluation
    await rl_agent.battle_against(ai_agent, n_battles=n_battles)

    print("\n=== Saving Local HTML Replays ===")

    saved = []
    for idx, battle in enumerate(rl_agent.battles.values(), 1):
        log = extract_log_from_battle(battle)

        if not log:
            print(f"[WARN] Battle {idx}: No replay log available.")
            continue

        path = save_replay_html(log, f"eval_battle_{idx}")
        print(f"[OK] Saved replay: {path}")
        saved.append(path)

    print("\n=== Evaluation complete ===")
    print("Open these files in your browser:")
    for s in saved:
        print("  -", s)


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    server_proc = start_showdown_server()
    try:
        asyncio.run(run_evaluation(n_battles=10))
    finally:
        print("\nShutting down Showdown server...")
        #server_proc.terminate()
