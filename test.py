# ============================================================
# single_worker_test.py  --  Single-Worker RL Debug Trainer
# ============================================================

import asyncio
from rl_agent import MyRLAgent
from showdown_server import start_showdown_server
from fixed_ais import FixedOrderMaxBasePower
from randomizer.team_generator import load_pool, generate_fixed_team


# ============================================================
# SIMPLE TRAINING LOOP (1 battle at a time)
# ============================================================

async def run_single_worker_test():
    print("\n=== SINGLE-WORKER RL DEBUG TRAINER ===")

    # -----------------------------
    # Load consistent teams
    # -----------------------------
    pool = load_pool("teams/team_pool.json")
    team_rl = generate_fixed_team(pool)
    team_ai = generate_fixed_team(pool)

    # -----------------------------
    # Init RL + opponent
    # -----------------------------
    rl = MyRLAgent(
        battle_format="gen9ubers",
        max_concurrent_battles=1,
        team=team_rl,
        model_folder="debug_model",
    )

    opponent = FixedOrderMaxBasePower(
        battle_format="gen9ubers",
        max_concurrent_battles=1,
        team=team_ai,
    )

    # Force settings for test
    rl.allow_switching = False
    rl.use_expert_switching = False
    rl.rl_switch_enabled = False

    # -----------------------------
    # 100 battles, 1 at a time
    # -----------------------------
    NUM_TEST_BATTLES = 100

    prev_finished = rl.n_finished_battles

    for i in range(1, NUM_TEST_BATTLES + 1):

        # Run 1 battle only
        await rl.battle_against(opponent, n_battles=1)

        battle_finished_now = rl.n_finished_battles - prev_finished
        prev_finished = rl.n_finished_battles

        print(f"\n[BATTLE {i}] finished={battle_finished_now}")
        print(f"[DEBUG] Epsilon after battle: {rl.epsilon:.6f}")
        print(f"[DEBUG] ≈ {rl.epsilon*100:.2f}% random actions")

        # Check if model saved
        rl.save_model()

    print("\n=== SINGLE-WORKER TEST COMPLETE ===")
    print(f"Final epsilon: {rl.epsilon:.6f}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("[MAIN] Starting Pokémon Showdown server...")
    server_proc = start_showdown_server()

    try:
        asyncio.run(run_single_worker_test())
    finally:
        print("[MAIN] Terminating Pokémon Showdown server...")
        server_proc.terminate()