# train_parallel.py

import asyncio
from collections import deque
import sys
import select

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
MODEL_FOLDER = "models/a2c_v17"

ROLLING_WINDOW = 20

MAX_PHASE_BATTLES = 10000         # hard cap per phase
MANUAL_SKIP_KEY = "6"             # press "6" to skip a phase

# --- thresholds per phase ---
WARMUP_THRESHOLD  = 0.80
PHASE1_THRESHOLD  = 0.60
PHASE2A_THRESHOLD = 0.90
PHASE2B_THRESHOLD = 0.60
PHASE3A_THRESHOLD = 0.60
PHASE3B_THRESHOLD = 0.75

# --- minimum cycles before advancement is allowed ---
MIN_WARMUP_CYCLES  = 4
MIN_PHASE1_CYCLES  = 40
MIN_PHASE2A_CYCLES = 50
MIN_PHASE2B_CYCLES = 60
MIN_PHASE3A_CYCLES = 60
MIN_PHASE3B_CYCLES = 80

TRAIN_PHASES = [
    "warmup",
    "phase1",
    "phase2a",
    "phase2b",
    "phase3a",
    "phase3b",
]

START_PHASE = "warmup"   # You may switch manually when restarting program


# ============================================================
# Helper: Rolling Window Average
# ============================================================
def rolling_avg(values: deque):
    return sum(values) / len(values) if values else 0.0


# ============================================================
# Helper: Agent Switching Configuration
# ============================================================

def configure_agent_for_phase(agent: MyRLAgent, phase: str):
    p = phase.lower()

    if p == "warmup":
        agent.allow_switching      = False
        agent.use_expert_switching = False
        agent.rl_switch_enabled    = False

    elif p == "phase1":
        agent.allow_switching      = False
        agent.use_expert_switching = False
        agent.rl_switch_enabled    = False

    elif p == "phase2a":  # Expert switching, MaxPower opponent
        agent.allow_switching      = True
        agent.use_expert_switching = True
        agent.rl_switch_enabled    = False

    elif p == "phase2b":  # RL switching, MaxPower opponent
        agent.allow_switching      = True
        agent.use_expert_switching = False
        agent.rl_switch_enabled    = True

    elif p == "phase3a":  # Expert switching, Heuristics opponent
        agent.allow_switching      = True
        agent.use_expert_switching = True
        agent.rl_switch_enabled    = False

    elif p == "phase3b":  # RL switching, Heuristics opponent
        agent.allow_switching      = True
        agent.use_expert_switching = False
        agent.rl_switch_enabled    = True

    else:
        raise ValueError(f"Unknown phase: {phase}")

    print(f"[CONFIG] Phase={p.upper()}  "
          f"allow={agent.allow_switching}  "
          f"expert={agent.use_expert_switching}  "
          f"rl_switch={agent.rl_switch_enabled}")


# ============================================================
# Helper: Select Opponent for Each Phase
# ============================================================

def opponent_for_phase(phase: str, team_ai, max_concurrent=MAX_PARALLEL):
    p = phase.lower()

    if p in ["warmup"]:
        return FixedOrderRandomPlayer(
            battle_format="gen9ubers",
            max_concurrent_battles=max_concurrent,
            team=team_ai
        )

    if p in ["phase1", "phase2a", "phase2b"]:
        return FixedOrderMaxBasePower(
            battle_format="gen9ubers",
            max_concurrent_battles=max_concurrent,
            team=team_ai
        )

    if p in ["phase3a", "phase3b"]:
        return FixedOrderSimpleHeuristics(
            battle_format="gen9ubers",
            max_concurrent_battles=max_concurrent,
            team=team_ai
        )

    raise ValueError(f"No opponent defined for phase: {phase}")


# ============================================================
# Detect keyboard press (non-blocking)
# ============================================================
def key_pressed() -> str | None:
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.readline().strip()
    return None


# ============================================================
# MAIN TRAINING LOOP
# ============================================================
async def train_forever():

    # ----------------------------------------------------------
    # Load fixed teams
    # ----------------------------------------------------------
    pool = load_pool("teams/team_pool.json")
    team_rl = generate_fixed_team(pool)
    team_ai = generate_fixed_team(pool)

    print("\n=== FIXED TRAINING TEAMS ===")
    print("RL Team:\n", team_rl)
    print("AI Team:\n", team_ai)
    print("====================================\n")

    # ----------------------------------------------------------
    # Initialize RL Agent
    # ----------------------------------------------------------
    rl_agent = MyRLAgent(
        battle_format="gen9ubers",
        max_concurrent_battles=MAX_PARALLEL,
        team=team_rl,
        model_folder=MODEL_FOLDER,
    )

    # ----------------------------------------------------------
    # Start at chosen phase
    # ----------------------------------------------------------
    phase_index = TRAIN_PHASES.index(START_PHASE.lower())
    phase = TRAIN_PHASES[phase_index]

    configure_agent_for_phase(rl_agent, phase)
    ai_agent = opponent_for_phase(phase, team_ai)

    print(f"\n===== STARTING TRAINING AT PHASE: {phase.upper()} =====\n")

    # Rolling window for winrate
    winrates = deque(maxlen=ROLLING_WINDOW)

    cycle = 0
    phase_battles = 0

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    while True:
        cycle += 1
        print(f"\n========== MINIBATCH {cycle} ({MINIBATCH} battles) ==========")

        prev_finished = rl_agent.n_finished_battles
        prev_wins = rl_agent.n_won_battles

        # Run battles
        await rl_agent.battle_against(ai_agent, n_battles=MINIBATCH)

        # Update counters
        batch_finished = rl_agent.n_finished_battles - prev_finished
        batch_wins = rl_agent.n_won_battles - prev_wins
        batch_winrate = batch_wins / batch_finished if batch_finished else 0

        phase_battles += batch_finished

        winrates.append(batch_winrate)
        r_avg = rolling_avg(winrates)

        # --- Output stats ---
        print(f"[{phase}] Batch {cycle}: {batch_wins}/{batch_finished} ({batch_winrate*100:.1f}%)")
        lifetime_finished = rl_agent.n_finished_battles
        lifetime_wins = rl_agent.n_won_battles
        lifetime_winrate = lifetime_wins / lifetime_finished if lifetime_finished else 0
        print(f"[{phase}] Lifetime: {lifetime_wins}/{lifetime_finished} ({lifetime_winrate*100:.1f}%)")
        print(f"[{phase}] Rolling {len(winrates)} → {r_avg*100:.1f}%")
        print(f"[{phase}] Phase battles so far: {phase_battles}")

        # --- internal training ---
        if rl_agent.episode_buffer:
            rl_agent._process_episodes()
            print("[TRAIN] Processed episodes into experience buffer.")

        while len(rl_agent.experience_buffer) >= rl_agent.batch_size:
            rl_agent._train_batch()
            print("[TRAIN] Trained one batch.")

        rl_agent.save_model()

        # ------------------------------------------------------
        # Check for manual skip (press "6")
        # ------------------------------------------------------
        if key_pressed() == MANUAL_SKIP_KEY:
            print("\n=== MANUAL ADVANCE REQUESTED ===")
            advance = True
        else:
            advance = False

        # ------------------------------------------------------
        # Decide thresholds for each phase
        # ------------------------------------------------------
        min_cycles = {
            "warmup":  MIN_WARMUP_CYCLES,
            "phase1":  MIN_PHASE1_CYCLES,
            "phase2a": MIN_PHASE2A_CYCLES,
            "phase2b": MIN_PHASE2B_CYCLES,
            "phase3a": MIN_PHASE3A_CYCLES,
            "phase3b": MIN_PHASE3B_CYCLES,
        }[phase]

        thresholds = {
            "warmup":  WARMUP_THRESHOLD,
            "phase1":  PHASE1_THRESHOLD,
            "phase2a": PHASE2A_THRESHOLD,
            "phase2b": PHASE2B_THRESHOLD,
            "phase3a": PHASE3A_THRESHOLD,
            "phase3b": PHASE3B_THRESHOLD,
        }[phase]

        # ------------------------------------------------------
        # Determine if we should move to next phase
        # ------------------------------------------------------
        cond_time = (cycle >= min_cycles)
        cond_perf = (r_avg >= thresholds)
        cond_cap  = (phase_battles >= MAX_PHASE_BATTLES)
        cond_any  = cond_time and cond_perf

        if advance or cond_any or cond_cap:
            print("\n=== ADVANCING TO NEXT PHASE ===\n")

            phase_index += 1
            if phase_index >= len(TRAIN_PHASES):
                print("\n=== TRAINING COMPLETE — NO MORE PHASES ===")
                break

            # Set next phase
            phase = TRAIN_PHASES[phase_index]
            winrates.clear()
            phase_battles = 0

            print(f"\n>>> ENTERING {phase.upper()} <<<\n")

            configure_agent_for_phase(rl_agent, phase)
            ai_agent = opponent_for_phase(phase, team_ai)
            continue
# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("[MAIN] Starting Pokémon Showdown server...")
    server_proc = start_showdown_server()

    try:
        print("[MAIN] Entering async training loop...")
        asyncio.run(train_forever())
    except KeyboardInterrupt:
        print("\n[MAIN] KeyboardInterrupt detected. Shutting down...")
    finally:
        print("[MAIN] Terminating Pokémon Showdown server...")
        server_proc.terminate()
        print("[MAIN] Shutdown complete.")