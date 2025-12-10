# train_parallel.py  --  Curriculum RL with per-phase epsilon resets
#                        + FULL LIFETIME STATS
#                        + PER-PHASE LR / ENTROPY SETTINGS
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
MODEL_FOLDER = "models/a2c_v50"

ROLLING_WINDOW = 20
MAX_PHASE_BATTLES = 10000
MANUAL_SKIP_KEY = "6"

# ============================================================
# Phase Difficulty Thresholds
# ============================================================

THRESHOLDS = {
    "warmup": 0.85,
    "phase1": 0.80,
    "phase2a": 0.80,
    "phase2b": 0.55,
    "phase3b": 0.60,
}

# ============================================================
# Minimum cycles before progress
# ============================================================

MIN_CYCLES = {
    "warmup": 6,
    "phase1": 65,
    "phase2a": 65,
    "phase2b": 80,
    "phase3b": 100,
}

# ============================================================
# Epsilon schedules (per phase)
# ============================================================

EPSILON_START = {
    "warmup": 1.0,
    "phase1": 0.8,
    "phase2a": 0.3,
    "phase2b": 0.2,   # low: RL switching, don't destroy policy
    "phase3b": 0.1,   # final heuristic boss
}

EPSILON_END = {
    "warmup": 0.2,
    "phase1": 0.10,
    "phase2a": 0.05,
    "phase2b": 0.05,
    "phase3b": 0.03,
}

EPSILON_DECAY = {
    "warmup": 0.9990,
    "phase1": 0.9990,
    "phase2a": 0.9991,
    "phase2b": 0.99864,
    "phase3b": 0.99864,
}

# ============================================================
# Per-phase LR & entropy (stability knobs)
# ============================================================

LR_PER_PHASE = {
    "warmup": 1e-3,
    "phase1": 1e-3,
    "phase2a": 7e-4,
    "phase2b": 3e-4,   # more conservative with RL switching
    "phase3b": 3e-4,
}

ENTROPY_PER_PHASE = {
    "warmup": 0.08,
    "phase1": 0.06,
    "phase2a": 0.05,
    "phase2b": 0.03,
    "phase3b": 0.02,
}

# ============================================================
# Supported Phases (NO PHASE 3A)
# ============================================================

TRAIN_PHASES = [
    "warmup",
    "phase1",
    "phase2a",
    "phase2b",
    "phase3b",
]

START_PHASE = "warmup"


# ============================================================
# Helpers
# ============================================================

def rolling_avg(values: deque):
    return sum(values) / len(values) if values else 0.0


# ============================================================
# Configure agent based on curriculum phase
# ============================================================

def configure_agent_for_phase(agent: MyRLAgent, phase: str):
    p = phase.lower()

    if p == "warmup":
        agent.allow_switching = False
        agent.use_expert_switching = False
        agent.rl_switch_enabled = False

    elif p == "phase1":
        agent.allow_switching = False
        agent.use_expert_switching = False
        agent.rl_switch_enabled = False

    elif p == "phase2a":
        agent.allow_switching = True
        agent.use_expert_switching = True
        agent.rl_switch_enabled = False  # expert only

    elif p == "phase2b":
        agent.allow_switching = True
        agent.use_expert_switching = False
        agent.rl_switch_enabled = True   # RL controls switching

    elif p == "phase3b":
        agent.allow_switching = True
        agent.use_expert_switching = False
        agent.rl_switch_enabled = True   # RL controls switching vs heuristic bot

    else:
        raise ValueError(f"Unknown phase: {phase}")

    # Reset epsilon schedule for this phase
    agent.epsilon = EPSILON_START[p]
    agent.epsilon_start = EPSILON_START[p]
    agent.epsilon_end = EPSILON_END[p]
    agent.epsilon_decay = EPSILON_DECAY[p]

    # Per-phase LR and entropy
    agent.set_lr(LR_PER_PHASE[p])
    agent.set_entropy_coef(ENTROPY_PER_PHASE[p])

    print(
        f"[CONFIG] Phase={p.upper()} | allow={agent.allow_switching} | "
        f"expert={agent.use_expert_switching} | rl_switch={agent.rl_switch_enabled}"
    )
    print(
        f"[CONFIG] Epsilon: start={agent.epsilon_start:.3f}, "
        f"end={agent.epsilon_end:.3f}, decay={agent.epsilon_decay:.6f}"
    )


# ============================================================
# Opponent selection
# ============================================================

def opponent_for_phase(phase: str, team_ai):
    p = phase.lower()

    if p == "warmup":
        return FixedOrderRandomPlayer(
            battle_format="gen9ubers",
            max_concurrent_battles=MAX_PARALLEL,
            team=team_ai,
        )

    if p in ["phase1", "phase2a", "phase2b"]:
        return FixedOrderMaxBasePower(
            battle_format="gen9ubers",
            max_concurrent_battles=MAX_PARALLEL,
            team=team_ai,
        )

    if p == "phase3b":
        return FixedOrderSimpleHeuristics(
            battle_format="gen9ubers",
            max_concurrent_battles=MAX_PARALLEL,
            team=team_ai,
        )


# ============================================================
# Manual skip handler
# ============================================================

def key_pressed():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.readline().strip()
    return None


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

async def train_forever():

    # Load teams
    pool = load_pool("teams/team_pool.json")
    team_rl = generate_fixed_team(pool)
    team_ai = generate_fixed_team(pool)

    print("\n=== FIXED TRAINING TEAMS LOADED ===")

    # RL agent instance
    rl_agent = MyRLAgent(
        battle_format="gen9ubers",
        max_concurrent_battles=MAX_PARALLEL,
        team=team_rl,
        model_folder=MODEL_FOLDER,
    )

    # Lifetime stats
    lifetime_finished = 0
    lifetime_wins = 0
    lifetime_batches = 0

    lifetime_by_phase = {p: {"wins": 0, "finished": 0, "batches": 0} for p in TRAIN_PHASES}

    # Phase stats
    phase_index = TRAIN_PHASES.index(START_PHASE)
    phase = TRAIN_PHASES[phase_index]

    configure_agent_for_phase(rl_agent, phase)
    ai_agent = opponent_for_phase(phase, team_ai)

    winrates = deque(maxlen=ROLLING_WINDOW)
    phase_finished = 0
    phase_wins = 0
    phase_batches = 0

    cycle = 0

    while True:
        cycle += 1
        lifetime_batches += 1
        phase_batches += 1

        print(f"\n========== MINIBATCH {cycle} ({MINIBATCH} battles) ==========")

        prev_finished = rl_agent.n_finished_battles
        prev_wins = rl_agent.n_won_battles

        # --- RUN BATTLES ---
        await rl_agent.battle_against(ai_agent, n_battles=MINIBATCH)

        # --- COMPUTE DELTAS ---
        batch_finished = rl_agent.n_finished_battles - prev_finished
        batch_wins = rl_agent.n_won_battles - prev_wins

        lifetime_finished += batch_finished
        lifetime_wins += batch_wins

        phase_finished += batch_finished
        phase_wins += batch_wins

        batch_winrate = batch_wins / batch_finished if batch_finished else 0.0
        winrates.append(batch_winrate)
        r_avg = rolling_avg(winrates)

        print(f"[{phase}] Batch {cycle}: {batch_wins}/{batch_finished} ({batch_winrate*100:.1f}%)")
        print(f"[{phase}] Epsilon now: {rl_agent.epsilon:.4f}  (~{rl_agent.epsilon*100:.1f}% random)")
        print(f"[{phase}] Rolling avg: {r_avg*100:.1f}% ({len(winrates)} samples)")
        print(
            f"[{phase}] Phase battles: {phase_finished} | "
            f"Record: {phase_wins}/{phase_finished} ({(phase_wins/phase_finished*100):.1f}%)"
        )

        print(
            f"--- Lifetime: {lifetime_wins}/{lifetime_finished} "
            f"({(lifetime_wins/lifetime_finished)*100:.1f}%) across {lifetime_batches} batches"
        )

        # --- PROCESS TRAINING BUFFERS ---
        if rl_agent.episode_buffer:
            rl_agent._process_episodes()
            print("[TRAIN] Episodes processed.")

        while len(rl_agent.experience_buffer) >= rl_agent.batch_size:
            rl_agent._train_batch()
            print("[TRAIN] Trained 1 batch.")

        rl_agent.save_model()

        # --- ADVANCEMENT CONDITIONS ---
        manual = (key_pressed() == MANUAL_SKIP_KEY)
        enough_cycles = phase_batches >= MIN_CYCLES[phase]
        good_avg = r_avg >= THRESHOLDS[phase]
        too_long = phase_finished >= MAX_PHASE_BATTLES

        if manual or (enough_cycles and good_avg) or too_long:

            lifetime_by_phase[phase]["wins"] += phase_wins
            lifetime_by_phase[phase]["finished"] += phase_finished
            lifetime_by_phase[phase]["batches"] += phase_batches

            print("\n===== PHASE SUMMARY =====")
            print(
                f"Phase {phase.upper()} | Wins={phase_wins}/{phase_finished} "
                f"({(phase_wins/phase_finished)*100:.1f}%) over {phase_batches} batches"
            )

            print("\n=== ADVANCING TO NEXT PHASE ===")

            phase_index += 1

            # --- TRAINING COMPLETE ---
            if phase_index >= len(TRAIN_PHASES):
                print("\n===== TRAINING COMPLETE =====")
                print("\n===== FINAL LIFETIME STATS =====")
                print(
                    f"Total wins: {lifetime_wins}/{lifetime_finished} "
                    f"({(lifetime_wins/lifetime_finished)*100:.2f}%)"
                )

                print("Per-phase breakdown:")
                for p, rec in lifetime_by_phase.items():
                    if rec["finished"] > 0:
                        wr = rec["wins"] / rec["finished"] * 100
                        print(
                            f"  {p.upper()}: {rec['wins']}/{rec['finished']} "
                            f"({wr:.1f}%) in {rec['batches']} batches"
                        )
                break

            # --- RESET FOR NEXT PHASE ---
            phase = TRAIN_PHASES[phase_index]
            winrates.clear()

            phase_finished = 0
            phase_wins = 0
            phase_batches = 0

            print(f"\n>>> ENTERING {phase.upper()} <<<")
            configure_agent_for_phase(rl_agent, phase)
            ai_agent = opponent_for_phase(phase, team_ai)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("[MAIN] Starting Pokémon Showdown server...")
    server_proc = start_showdown_server()

    try:
        asyncio.run(train_forever())
    finally:
        print("[MAIN] Terminating Pokémon Showdown server...")
        server_proc.terminate()