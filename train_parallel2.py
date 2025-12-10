# train_parallel.py  -- 4-PHASE CURRICULUM:
#   warmup  = no switching vs RANDOM
#   phase1  = no switching vs MaxBasePower
#   phase2  = heuristic-only switching vs MaxBasePower
#   phase3  = heuristic-only switching vs SimpleHeuristic

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
MODEL_FOLDER = "models/a2c_v48"

ROLLING_WINDOW = 20
MAX_PHASE_BATTLES = 10000
MANUAL_SKIP_KEY = "6"

# ============================================================
# PHASE ORDER (NEW)
# ============================================================

TRAIN_PHASES = [
    "warmup",   # no switching, random opponent
    "phase1",   # no switching, MaxBasePower
    "phase2",   # heuristic switching, MaxBasePower
    "phase3",   # heuristic switching, SimpleHeuristic
]

START_PHASE = "warmup"

# ============================================================
# Phase Difficulty Thresholds
# ============================================================

THRESHOLDS = {
    "warmup": 0.85,
    "phase1": 0.85,
    "phase2": 0.65,
    "phase3": 0.60,
}

# ============================================================
# Minimum cycles before progress
# ============================================================

MIN_CYCLES = {
    "warmup": 6,
    "phase1": 40,
    "phase2": 65,
    "phase3": 75,
}

# ============================================================
# Epsilon schedules
# ============================================================

EPSILON_START = {
    "warmup": 1.0,
    "phase1": 0.6,
    "phase2": 0.6,
    "phase3": 0.5,
}

EPSILON_END = {
    "warmup": 0.15,
    "phase1": 0.05,
    "phase2": 0.05,
    "phase3": 0.03,
}

EPSILON_DECAY = {
    "warmup": 0.9990,
    "phase1": 0.9990,
    "phase2": 0.9987,
    "phase3": 0.9985,
}

# ============================================================
# LR & ENTROPY per phase
# ============================================================

LR_PER_PHASE = {
    "warmup": 1e-3,
    "phase1": 7e-4,
    "phase2": 5e-4,
    "phase3": 4e-4,
}

ENTROPY_PER_PHASE = {
    "warmup": 0.01,
    "phase1": 0.01,
    "phase2": 0.01,
    "phase3": 0.01,
}

# ============================================================
# Helper
# ============================================================

def rolling_avg(values: deque):
    return sum(values) / len(values) if values else 0.0

# ============================================================
# Configure agent for each phase
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

    elif p == "phase2":
        agent.allow_switching = True
        agent.use_expert_switching = True   # heuristic chooses switches
        agent.rl_switch_enabled = False

    elif p == "phase3":
        agent.allow_switching = True
        agent.use_expert_switching = True   # heuristic chooses switches
        agent.rl_switch_enabled = False

    else:
        raise ValueError(f"Unknown phase: {phase}")

    # Epsilon reset
    agent.epsilon = EPSILON_START[p]
    agent.epsilon_start = EPSILON_START[p]
    agent.epsilon_end = EPSILON_END[p]
    agent.epsilon_decay = EPSILON_DECAY[p]

    # LR + Entropy
    agent.set_lr(LR_PER_PHASE[p])
    agent.set_entropy_coef(ENTROPY_PER_PHASE[p])

    print(
        f"[CONFIG] Phase={p.upper()} | allow={agent.allow_switching} | "
        f"expert={agent.use_expert_switching}"
    )
    print(
        f"[CONFIG] ε={agent.epsilon_start:.2f}→{agent.epsilon_end:.2f} decay={agent.epsilon_decay}"
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

    if p == "phase1":
        return FixedOrderMaxBasePower(
            battle_format="gen9ubers",
            max_concurrent_battles=MAX_PARALLEL,
            team=team_ai,
        )

    if p == "phase2":
        return FixedOrderMaxBasePower(
            battle_format="gen9ubers",
            max_concurrent_battles=MAX_PARALLEL,
            team=team_ai,
        )

    if p == "phase3":
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
# MAIN LOOP
# ============================================================

async def train_forever():

    # Load fixed teams
    pool = load_pool("teams/team_pool.json")
    team_rl = generate_fixed_team(pool)
    team_ai = generate_fixed_team(pool)

    print("\n=== FIXED TRAINING TEAMS LOADED ===")

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

    phase_index = TRAIN_PHASES.index(START_PHASE)
    phase = TRAIN_PHASES[phase_index]

    configure_agent_for_phase(rl_agent, phase)
    ai_agent = opponent_for_phase(phase, team_ai)

    winrates = deque(maxlen=ROLLING_WINDOW)
    phase_finished = 0
    phase_wins = 0
    phase_batches = 0
    cycle = 0

    # MAIN TRAINING LOOP
    while True:
        cycle += 1
        lifetime_batches += 1
        phase_batches += 1

        print(f"\n========== MINIBATCH {cycle} ({MINIBATCH} battles) ==========")

        prev_finished = rl_agent.n_finished_battles
        prev_wins = rl_agent.n_won_battles

        await rl_agent.battle_against(ai_agent, n_battles=MINIBATCH)

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
        print(f"[{phase}] ε now: {rl_agent.epsilon:.4f}  (~{rl_agent.epsilon*100:.1f}% random)")
        print(f"[{phase}] Rolling avg: {r_avg*100:.1f}%")
        print(f"[{phase}] Record: {phase_wins}/{phase_finished}")

        print(
            f"Lifetime: {lifetime_wins}/{lifetime_finished} "
            f"({(lifetime_wins/lifetime_finished)*100:.1f}%)"
        )

        # Process training buffers
        if rl_agent.episode_buffer:
            rl_agent._process_episodes()

        while len(rl_agent.experience_buffer) >= rl_agent.batch_size:
            rl_agent._train_batch()

        rl_agent.save_model()

        # ADVANCEMENT CONDITIONS
        manual = (key_pressed() == MANUAL_SKIP_KEY)
        enough_cycles = phase_batches >= MIN_CYCLES[phase]
        good_avg = r_avg >= THRESHOLDS[phase]
        too_long = phase_finished >= MAX_PHASE_BATTLES

        if manual or (enough_cycles and good_avg) or too_long:

            lifetime_by_phase[phase]["wins"] += phase_wins
            lifetime_by_phase[phase]["finished"] += phase_finished
            lifetime_by_phase[phase]["batches"] += phase_batches

            print("\n===== PHASE COMPLETE =====")
            print(
                f"{phase.upper()}: {phase_wins}/{phase_finished} "
                f"({(phase_wins/phase_finished)*100:.1f}%)"
            )

            phase_index += 1

            if phase_index >= len(TRAIN_PHASES):
                print("\n===== TRAINING COMPLETE =====")
                print("Final results:")
                for p, rec in lifetime_by_phase.items():
                    if rec["finished"] > 0:
                        wr = rec["wins"] / rec["finished"] * 100
                        print(
                            f"{p.upper()}: {rec['wins']}/{rec['finished']} "
                            f"({wr:.1f}%)"
                        )
                break

            # NEXT PHASE
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