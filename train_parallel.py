# train_parallel.py

import asyncio
from collections import deque

from rl_agent import MyRLAgent
from showdown_server import start_showdown_server

from randomizer.team_generator import (
    load_pool,
    generate_fixed_team,
)

from fixed_ais import (
    FixedOrderRandomPlayer,
    FixedOrderMaxBasePower,
    FixedOrderSimpleHeuristics,
)

# ============================================================
# TRAINING CONSTANTS
# ============================================================
MINIBATCH = 50
MAX_PARALLEL = 16
MODEL_FOLDER = "models/a2c_v10"

ROLLING_WINDOW = 20

# Phase thresholds
WARMUP_THRESHOLD = 0.55
PHASE1_THRESHOLD = 0.60
PHASE2A_THRESHOLD = 0.67
PHASE2B_THRESHOLD = 0.70  # mastery

# Minimum cycles per stage
MIN_WARMUP_CYCLES = 4
MIN_PHASE1_CYCLES = 12
MIN_PHASE2A_CYCLES = 12

# Allow resuming from anywhere
START_PHASE = "warmup"   # "warmup" | "phase1" | "phase2a" | "phase2b"


# ============================================================
# Helper: Rolling average window
# ============================================================
def rolling_avg(values: deque):
    return sum(values) / len(values) if values else 0.0


# ============================================================
# MAIN TRAINING LOOP
# ============================================================
async def train_forever():

    # ----------------------------------------------------------
    # Load teams (fixed for both sides)
    # ----------------------------------------------------------
    pool = load_pool("teams/team_pool.json")
    team_rl = generate_fixed_team(pool)
    team_ai = generate_fixed_team(pool)

    print("\n=== FIXED TRAINING TEAMS ===")
    print("RL Team:\n", team_rl)
    print("AI Team:\n", team_ai)
    print("====================================\n")

    # ----------------------------------------------------------
    # Instantiate RL agent
    # allow_switching will be configured per phase
    # ----------------------------------------------------------
    rl_agent = MyRLAgent(
        battle_format="gen9ubers",
        max_concurrent_battles=MAX_PARALLEL,
        team=team_rl,
        model_folder=MODEL_FOLDER,
    )

    # ----------------------------------------------------------
    # Curriculum setup
    # ----------------------------------------------------------
    phase = START_PHASE.lower()

    if phase == "warmup":
        ai_agent = FixedOrderRandomPlayer(
            battle_format="gen9ubers",
            max_concurrent_battles=MAX_PARALLEL,
            team=team_ai,
        )
        rl_agent.allow_switching = False
        rl_agent.use_expert_switching = False

    elif phase == "phase1":
        ai_agent = FixedOrderMaxBasePower(
            battle_format="gen9ubers",
            max_concurrent_battles=MAX_PARALLEL,
            team=team_ai,
        )
        rl_agent.allow_switching = False
        rl_agent.use_expert_switching = False

    elif phase == "phase2a":
        ai_agent = FixedOrderMaxBasePower(
            battle_format="gen9ubers",
            max_concurrent_battles=MAX_PARALLEL,
            team=team_ai,
        )
        rl_agent.allow_switching = True
        rl_agent.use_expert_switching = True

    elif phase == "phase2b":
        ai_agent = FixedOrderSimpleHeuristics(
            battle_format="gen9ubers",
            max_concurrent_battles=MAX_PARALLEL,
            team=team_ai,
        )
        rl_agent.allow_switching = True
        rl_agent.use_expert_switching = False

    else:
        raise ValueError(f"Invalid START_PHASE: {START_PHASE}")

    print(f"\n===== STARTING TRAINING AT PHASE: {phase.upper()} =====\n")

    # ----------------------------------------------------------
    # Rolling performance window
    # ----------------------------------------------------------
    winrates = deque(maxlen=ROLLING_WINDOW)
    cycle = 0

    # ============================================================
    # MAIN TRAINING LOOP
    # ============================================================
    while True:
        cycle += 1
        print(f"\n========== MINIBATCH {cycle} ({MINIBATCH} battles) ==========")

        prev_finished = rl_agent.n_finished_battles
        prev_wins = rl_agent.n_won_battles

        await rl_agent.battle_against(ai_agent, n_battles=MINIBATCH)

        # Stats
        batch_finished = rl_agent.n_finished_battles - prev_finished
        batch_wins = rl_agent.n_won_battles - prev_wins
        batch_winrate = batch_wins / batch_finished if batch_finished > 0 else 0

        winrates.append(batch_winrate)
        r_avg = rolling_avg(winrates)

        print(
            f"[{phase}] Batch {cycle}: {batch_wins}/{batch_finished} "
            f"({batch_winrate*100:.1f}%)"
        )

        # --- Lifetime stats ---
        lifetime_finished = rl_agent.n_finished_battles
        lifetime_wins = rl_agent.n_won_battles

        if lifetime_finished > 0:
            lifetime_winrate = lifetime_wins / lifetime_finished
        else:
            lifetime_winrate = 0.0

        print(
            f"[{phase}] Lifetime: {lifetime_wins}/{lifetime_finished} "
            f"({lifetime_winrate*100:.1f}%)"
        )

        print(
            f"[{phase}] Rolling ({len(winrates)}): {r_avg*100:.1f}%"
        )

        # Train neural net updates
        if rl_agent.episode_buffer:
            rl_agent._process_episodes()

        while len(rl_agent.experience_buffer) >= rl_agent.batch_size:
            rl_agent._train_batch()

        rl_agent.save_model()

        # ========================================================
        # Phase Transitions
        # ========================================================

        # -----------------------
        # Warmup → Phase 1
        # -----------------------
        if (
            phase == "warmup"
            and cycle >= MIN_WARMUP_CYCLES
            and r_avg >= WARMUP_THRESHOLD
        ):
            print("\n=== ADVANCING TO PHASE 1: Moves Only vs MaxBasePower ===\n")

            phase = "phase1"
            winrates.clear()

            rl_agent.allow_switching = False
            rl_agent.use_expert_switching = False

            ai_agent = FixedOrderMaxBasePower(
                battle_format="gen9ubers",
                max_concurrent_battles=MAX_PARALLEL,
                team=team_ai,
            )
            continue

        # -----------------------
        # Phase 1 → Phase 2A
        # -----------------------
        if (
            phase == "phase1"
            and cycle >= MIN_PHASE1_CYCLES
            and r_avg >= PHASE1_THRESHOLD
        ):
            print("\n=== ADVANCING TO PHASE 2A: Switching + Expert Guidance ===\n")

            phase = "phase2a"
            winrates.clear()

            rl_agent.allow_switching = True
            rl_agent.use_expert_switching = True

            ai_agent = FixedOrderMaxBasePower(
                battle_format="gen9ubers",
                max_concurrent_battles=MAX_PARALLEL,
                team=team_ai,
            )
            continue

        # -----------------------
        # Phase 2A → Phase 2B
        # -----------------------
        if (
            phase == "phase2a"
            and cycle >= MIN_PHASE2A_CYCLES
            and r_avg >= PHASE2A_THRESHOLD
        ):
            print("\n=== ADVANCING TO PHASE 2B: vs SimpleHeuristics ===\n")

            phase = "phase2b"
            winrates.clear()

            rl_agent.allow_switching = True
            rl_agent.use_expert_switching = False

            ai_agent = FixedOrderSimpleHeuristics(
                battle_format="gen9ubers",
                max_concurrent_battles=MAX_PARALLEL,
                team=team_ai,
            )
            continue

        # -----------------------
        # Optional Mastery
        # -----------------------
        if phase == "phase2b" and r_avg >= PHASE2B_THRESHOLD:
            print("\n=== RL AGENT HAS MASTERED THE CURRICULUM! ===\n")


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    server_proc = start_showdown_server()
    try:
        asyncio.run(train_forever())
    finally:
        print("\nShutting down Pokémon Showdown server...")
        server_proc.terminate()
