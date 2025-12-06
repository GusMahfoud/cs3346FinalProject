import asyncio
from poke_env.player.baselines import MaxBasePowerPlayer
from rl_agent import MyRLAgent
from randomizer.team_generator import load_pool, generate_team_from_pool
from showdown_server import start_showdown_server

# -----------------------------
# TRAINING SETTINGS
# -----------------------------
MINIBATCH = 50
MAX_PARALLEL = 16

# Folder where checkpoint.pth will be saved/loaded
MODEL_FOLDER = "models/a2c_v3"


async def train_forever():
    # -----------------------------
    # LOAD TEAM POOL AND CHOOSE ONCE
    # -----------------------------
    pool = load_pool("teams/team_pool.json")

    team_rl = generate_team_from_pool(pool)
    team_ai = generate_team_from_pool(pool)

    print("\n=== FIXED TEAMS FOR ENTIRE TRAINING ===")
    print("RL Team:", team_rl)
    print("AI Team:", team_ai)
    print("========================================\n")

    # -----------------------------
    # CREATE AGENTS (FIXED TEAMS)
    # -----------------------------
    rl_agent = MyRLAgent(
        battle_format="gen9ubers",
        model_path=MODEL_FOLDER,
        max_concurrent_battles=MAX_PARALLEL,
        team=team_rl,
    )

    ai_agent = MaxBasePowerPlayer(
        battle_format="gen9ubers",
        max_concurrent_battles=MAX_PARALLEL,
        team=team_ai,
    )

    print("\n=== STARTING INFINITE TRAINING LOOP (FIXED TEAMS) ===\n")

    cycle = 0

    while True:
        cycle += 1
        print(f"========== MINIBATCH {cycle}  ({MINIBATCH} battles) ==========")

        # -----------------------------
        # RUN BATTLES
        # -----------------------------
        await rl_agent.battle_against(
            ai_agent,
            n_battles=MINIBATCH
        )

        print(
            f"Minibatch result: {rl_agent.n_won_battles}/"
            f"{rl_agent.n_finished_battles} won (lifetime)\n"
        )

        # -----------------------------
        # PROCESS EXPERIENCE + TRAIN
        # -----------------------------
        if rl_agent.episode_buffer:
            rl_agent._process_episodes()

        while len(rl_agent.experience_buffer) >= rl_agent.batch_size:
            rl_agent._train_batch()

        # -----------------------------
        # SAVE MODEL
        # -----------------------------
        rl_agent.save_model(MODEL_FOLDER)

        print("Mini-batch complete. Continuing...\n")


# ==========================================================
#                     MAIN ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    server_proc = start_showdown_server()
    try:
        asyncio.run(train_forever())
    finally:
        print("\nShutting down Showdown server...")
        server_proc.terminate()
