# model_video.py
# -----------------------------------------------------------
# Launches a single visible battle vs your trained RL agent,
# using Phase 2A config (expert switching ON).
# Epsilon is forced to 0 for full determinism.
# -----------------------------------------------------------

import asyncio
from rl_agent import MyRLAgent
from showdown_server import start_showdown_server
from randomizer.team_generator import load_pool, generate_fixed_team
from fixed_ais import FixedOrderMaxBasePower
from train_parallel2 import configure_agent_for_phase   # reuse your existing setup

PHASE = "phase2a"     # expert switching ON
TEAM_JSON = "teams/team_pool.json"
MODEL_FOLDER = "models/a2c_v50"


async def run_showcase_battle():

    print("\n=== LOADING TEAMS ===")
    pool = load_pool(TEAM_JSON)
    team_rl = generate_fixed_team(pool)
    team_ai = generate_fixed_team(pool)

    print("\n=== INIT RL AGENT (Phase 2A) ===")
    rl_agent = MyRLAgent(
        battle_format="gen9ubers",
        max_concurrent_battles=1,
        team=team_rl,
        model_folder=MODEL_FOLDER,
    )

    # Configure Phase 2A settings
    configure_agent_for_phase(rl_agent, PHASE)

    # 🔥 Force epsilon = 0 (absolutely no randomness)
    rl_agent.epsilon = 0.0
    rl_agent.epsilon_start = 0.0
    rl_agent.epsilon_end = 0.0
    rl_agent.epsilon_decay = 1.0  # ensures it stays at 0

    print("[DEBUG] Epsilon forced to 0 (fully deterministic behaviour)")

    # Opponent for phase2a (Max Base Power)
    opponent = FixedOrderMaxBasePower(
        battle_format="gen9ubers",
        max_concurrent_battles=1,
        team=team_ai,
    )

    print("\n=== STARTING VISIBLE BATTLE ===")
    print("Open the URL shown below in your browser.\n")

    # Run exactly *one* battle and do not close it prematurely
    await rl_agent.battle_against(opponent, n_battles=1)

    print("\n=== BATTLE COMPLETE ===")
    print("The browser tab stays open so you can inspect the full battle.")


if __name__ == "__main__":
    print("[MAIN] Starting Pokémon Showdown server...")
    server_proc = start_showdown_server()

    try:
        asyncio.run(run_showcase_battle())
    finally:
        print("[MAIN] You may close the battle UI whenever you're done.")