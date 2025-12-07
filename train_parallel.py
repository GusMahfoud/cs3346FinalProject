import asyncio
from collections import deque

from poke_env.player.baselines import MaxBasePowerPlayer, SimpleHeuristicsPlayer
from rl_agent import MyRLAgent

from randomizer.team_generator import load_pool, generate_random_lead_team, generate_fixed_team
from showdown_server import start_showdown_server


# ============================================================
# TRAINING CONSTANTS
# ============================================================
MINIBATCH = 50
MAX_PARALLEL = 16
MODEL_FOLDER = "models/a2c_v7"

ROLLING_WINDOW = 20   # number of minibatches for rolling winrate

# Win-rate thresholds (rolling)
PHASE1_THRESHOLD = 0.6   # empirically correct for moves-only phase
PHASE2A_THRESHOLD = 0.65
PHASE2B_THRESHOLD = 0.70  # mastery threshold (optional)

# Minimum batches before transitions
MIN_BATCHES_PHASE1 = 12
MIN_BATCHES_PHASE2A = 15


# ============================================================
# ROLLING AVG HELPER
# ============================================================
def compute_rolling_avg(values: deque):
    return sum(values) / len(values) if values else 0.0


# ============================================================
# MAIN TRAINING LOOP
# ============================================================
async def train_forever():

    # ------------------------------------------
    # Load pool + generate fixed teams
    # ------------------------------------------
    pool = load_pool("teams/team_pool.json")
    team_rl = generate_fixed_team(pool)
    team_ai = generate_fixed_team(pool)

    print("\n=== FIXED TRAINING TEAMS ===")
    print(team_rl, "\n\n", team_ai)
    print("======================================\n")

    # ------------------------------------------
    # Create RL agent — PHASE 1 SETTINGS
    # ------------------------------------------
    rl_agent = MyRLAgent(
        battle_format="gen9ubers",
        model_folder=MODEL_FOLDER,
        max_concurrent_battles=MAX_PARALLEL,
        team=team_rl,

        # PHASE 1 = moves only
        allow_switching=False,
        use_expert_switching=True,
    )

    # Phase 1 opponent
    ai_agent = MaxBasePowerPlayer(
        battle_format="gen9ubers",
        max_concurrent_battles=MAX_PARALLEL,
        team=team_ai,
    )

    phase = "PHASE1"
    rolling_winrates = deque(maxlen=ROLLING_WINDOW)
    cycle = 0

    print("===== START TRAINING: PHASE 1 (Moves Only) =====\n")

    # ============================================================
    # Infinite curriculum loop
    # ============================================================
    while True:
        cycle += 1
        print(f"\n========== MINIBATCH {cycle} ({MINIBATCH} battles) ==========")

        # ----------------------------------------------------------
        # Run battles
        # ----------------------------------------------------------
        prev_finished = rl_agent.n_finished_battles
        prev_wins = rl_agent.n_won_battles

        await rl_agent.battle_against(ai_agent, n_battles=MINIBATCH)

        # Batch stats
        batch_finished = rl_agent.n_finished_battles - prev_finished
        batch_wins = rl_agent.n_won_battles - prev_wins
        batch_winrate = (batch_wins / batch_finished) if batch_finished > 0 else 0.0

        # Rolling avg update
        rolling_winrates.append(batch_winrate)
        rolling_avg = compute_rolling_avg(rolling_winrates)

        # Lifetime stats
        lifetime_finished = rl_agent.n_finished_battles
        lifetime_wins = rl_agent.n_won_battles
        lifetime_winrate = (
            lifetime_wins / lifetime_finished if lifetime_finished else 0.0
        )

        # ----------------------------------------------------------
        # Pretty console reporting
        # ----------------------------------------------------------
        print(f"\n[{phase}] ======== RESULTS FOR BATCH {cycle} ========")
        print(
            f"[{phase}] Batch:    {batch_wins}/{batch_finished} "
            f"({batch_winrate*100:.1f}%)"
        )
        print(
            f"[{phase}] Lifetime: {lifetime_wins}/{lifetime_finished} "
            f"({lifetime_winrate*100:.1f}%)"
        )
        print(
            f"[{phase}] Rolling ({len(rolling_winrates)}/{ROLLING_WINDOW}): "
            f"{rolling_avg*100:.1f}%"
        )

        # ----------------------------------------------------------
        # Training
        # ----------------------------------------------------------
        if rl_agent.episode_buffer:
            rl_agent._process_episodes()

        while len(rl_agent.experience_buffer) >= rl_agent.batch_size:
            rl_agent._train_batch()

        rl_agent.save_model()

        # ============================================================
        # PHASE TRANSITIONS
        # ============================================================

        # ------------------------------
        # Phase 1 → Phase 2A
        # ------------------------------
        if (
            phase == "PHASE1"
            and cycle >= MIN_BATCHES_PHASE1
            and rolling_avg >= PHASE1_THRESHOLD
        ):
            print("\n=== ADVANCING TO PHASE 2A (Switching vs MaxDamage) ===\n")

            phase = "PHASE2A"
            rl_agent.allow_switching = True
            rl_agent.use_expert_switching = True  # enable expert guidance
            rl_agent.epsilon = rl_agent.epsilon_start
            rolling_winrates.clear()

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
            print("\n=== ADVANCING TO PHASE 2B (Switching vs SimpleHeuristics AI) ===\n")

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
        # Optional mastery message
        # ------------------------------
        if phase == "PHASE2B" and rolling_avg >= PHASE2B_THRESHOLD:
            print("\n=== RL AGENT HAS MASTERED THE CURRICULUM! ===\n")


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
