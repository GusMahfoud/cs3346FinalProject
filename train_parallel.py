import asyncio
from collections import deque

from poke_env.player.baselines import MaxBasePowerPlayer, SimpleHeuristicsPlayer
from rl_agent import MyRLAgent

from randomizer.team_generator import load_pool, generate_random_lead_team
from showdown_server import start_showdown_server


# ============================================================
# TRAINING CONSTANTS
# ============================================================
MINIBATCH = 50
MAX_PARALLEL = 16
MODEL_FOLDER = "models/a2c_v4"

ROLLING_WINDOW = 20   # number of minibatches for rolling average

# Win-rate thresholds (rolling)
PHASE1_THRESHOLD = 0.60
PHASE2A_THRESHOLD = 0.70
PHASE2B_THRESHOLD = 0.70     # mastery threshold (optional)

MIN_BATCHES_PHASE1 = 10   # must complete at least 10 batches before switching
MIN_BATCHES_PHASE2A = 10  # must complete at least 10 batches before switching again

# ============================================================
# HELPER
# ============================================================
def compute_rolling_avg(values: deque):
    if len(values) == 0:
        return 0.0
    return sum(values) / len(values)


# ============================================================
# MAIN TRAINING LOOP
# ============================================================
async def train_forever():

    # ------------------------------------------
    # Load pool and create fixed teams
    # ------------------------------------------
    pool = load_pool("teams/team_pool.json")
    team_rl = generate_random_lead_team(pool)
    team_ai = generate_random_lead_team(pool)

    print("\n=== FIXED TRAINING TEAMS ===")
    print(team_rl, "\n\n", team_ai)
    print("============================\n")

    # ------------------------------------------
    # Create the RL agent (starts with NO switching)
    # ------------------------------------------
    rl_agent = MyRLAgent(
        battle_format="gen9ubers",
        model_folder=MODEL_FOLDER,
        max_concurrent_battles=MAX_PARALLEL,
        team=team_rl,
        allow_switching=True,   # explicitly start in Phase 1
    )

    # Start Phase 1 opponent
    ai_agent = SimpleHeuristicsPlayer(
        battle_format="gen9ubers",
        max_concurrent_battles=MAX_PARALLEL,
        team=team_ai,
    )

    phase = "PHASE1"



    # Rolling win-rate storage
    rolling_winrates = deque(maxlen=ROLLING_WINDOW)

    cycle = 0
    print("\n===== START TRAINING (PHASE 1: Only Attacking) =====\n")

    # ============================================================
    # Infinite curriculum training loop
    # ============================================================
    while True:
        cycle += 1
        print(f"\n========== MINIBATCH {cycle} ({MINIBATCH} battles) ==========")

        # ----------------------------------------------------------
        # Run battles
        # ----------------------------------------------------------
                # ----------------------------------------------------------
        # Run battles
        # ----------------------------------------------------------
        prev_finished = rl_agent.n_finished_battles
        prev_wins = rl_agent.n_won_battles

        await rl_agent.battle_against(ai_agent, n_battles=MINIBATCH)

        # ----------------------------------------------------------
        # Compute batch-specific win rate
        # ----------------------------------------------------------
        batch_finished = rl_agent.n_finished_battles - prev_finished
        batch_wins = rl_agent.n_won_battles - prev_wins

        batch_winrate = (batch_wins / batch_finished) if batch_finished > 0 else 0.0

        # Rolling average update
        rolling_winrates.append(batch_winrate)
        rolling_avg = compute_rolling_avg(rolling_winrates)

        # ----------------------------------------------------------
        # Lifetime stats
        # ----------------------------------------------------------
        lifetime_finished = rl_agent.n_finished_battles
        lifetime_wins = rl_agent.n_won_battles
        lifetime_winrate = (
            lifetime_wins / lifetime_finished if lifetime_finished > 0 else 0.0
        )

        # ----------------------------------------------------------
        # Pretty reporting
        # ----------------------------------------------------------
        print(f"\n[{phase}] ======== BATCH {cycle} RESULTS ========")
        print(
            f"[{phase}] Batch:    {batch_wins} / {batch_finished} "
            f"({batch_winrate*100:.1f}%)"
        )
        print(
            f"[{phase}] Lifetime: {lifetime_wins} / {lifetime_finished} "
            f"({lifetime_winrate*100:.1f}%)"
        )
        print(
            f"[{phase}] Rolling ({len(rolling_winrates)}/{ROLLING_WINDOW}): "
            f"{rolling_avg:.3f}"
        )


        # ----------------------------------------------------------
        # Train the agent
        # ----------------------------------------------------------
        if rl_agent.episode_buffer:
            rl_agent._process_episodes()

        while len(rl_agent.experience_buffer) >= rl_agent.batch_size:
            rl_agent._train_batch()

        rl_agent.save_model()

        # ============================================================
        # PHASE TRANSITIONS BASED ON ROLLING AVERAGE
        # ============================================================

        # ------------------------------
        # Move from Phase 1 → Phase 2A
        # ------------------------------
                # ============================================================
        # PHASE TRANSITIONS BASED ON ROLLING AVERAGE + MIN BATCHES
        # ============================================================

        # ------------------------------
        # Phase 1 → Phase 2A
        # ------------------------------
        if (
            phase == "PHASE1"
            and cycle >= MIN_BATCHES_PHASE1
            and rolling_avg >= PHASE1_THRESHOLD
        ):
            print("\n=== ADVANCING TO PHASE 2A (Switching enabled vs MaxDamage) ===\n")

            phase = "PHASE2A"
            rl_agent.allow_switching = True
            rl_agent.epsilon = rl_agent.epsilon_start
            rolling_winrates.clear()

            # reset AI opponent
            ai_agent = MaxBasePowerPlayer(
                battle_format="gen9ubers",
                max_concurrent_battles=MAX_PARALLEL,
                team=team_ai,
            )
            continue

        # ------------------------------
        # Phase 2A → Phase 2B
        # ------------------------------
        if (
            phase == "PHASE2A"
            and cycle >= MIN_BATCHES_PHASE2A
            and rolling_avg >= PHASE2A_THRESHOLD
        ):
            print("\n=== ADVANCING TO PHASE 2B (Switching vs Heuristic AI) ===\n")

            phase = "PHASE2B"
            rl_agent.epsilon = rl_agent.epsilon_start
            rolling_winrates.clear()

            ai_agent = SimpleHeuristicsPlayer(
                battle_format="gen9ubers",
                max_concurrent_battles=MAX_PARALLEL,
                team=team_ai,
            )
            continue


        # ------------------------------
        # Optional: mastery message
        # ------------------------------
        if phase == "PHASE2B" and rolling_avg >= PHASE2B_THRESHOLD:
            print("\n=== RL AGENT HAS BEATEN THE CURRICULUM! Continuing refinement... ===\n")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    server_proc = start_showdown_server()
    try:
        asyncio.run(train_forever())
    finally:
        print("\nShutting down Showdown server...")
        server_proc.terminate()
