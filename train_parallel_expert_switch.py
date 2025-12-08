# train_parallel.py  --  4-Phase Curriculum RL + Expert Switches Only
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
MODEL_FOLDER = "models/a2c_v26"

ROLLING_WINDOW = 20
MAX_PHASE_BATTLES = 12000
MANUAL_SKIP_KEY = "6"

# ============================================================
# Curriculum Phases (Simplified to 4)
# ============================================================
TRAIN_PHASES = ["warmup", "phase1", "phase2", "phase3"]
START_PHASE = "warmup"

# ============================================================
# Difficulty thresholds for advancement
# ============================================================
THRESHOLDS = {
    "warmup": 0.85,
    "phase1": 0.80,
    "phase2": 0.60,
    "phase3": 0.60,
}

# ============================================================
# Minimum minibatch cycles before eligible to advance
# ============================================================
MIN_CYCLES = {
    "warmup": 6,
    "phase1": 70,
    "phase2": 90,
    "phase3": 120,
}

# ============================================================
# Optimized epsilon resets / decays (Strategy B)
# ============================================================
EPSILON_RESET = {
    "warmup": 1.0,   # heavy exploration
    "phase1": 0.8,
    "phase2": 0.6,
    "phase3": 0.4,
}

EPSILON_DECAY = {
    "warmup": 0.9990,   # → eps ~0.20 after ~1800 battles
    "phase1": 0.9988,   # → eps ~0.10 after ~2100 battles
    "phase2": 0.9986,   # → eps ~0.05 after ~2600 battles
    "phase3": 0.9984,   # → eps ~0.02 after ~3000 battles
}

# ============================================================
# Helpers
# ============================================================
def rolling_avg(values: deque):
    return sum(values) / len(values) if values else 0.0


# ============================================================
# Phase configuration
# ============================================================
def configure_agent_for_phase(agent: MyRLAgent, phase: str):
    p = phase.lower()

    # Switching logic (RL switching is fully disabled)
    if p == "warmup":
        agent.allow_switching = False
        agent.use_expert_switching = False

    elif p == "phase1":
        agent.allow_switching = False
        agent.use_expert_switching = False

    elif p == "phase2":   # expert switching ON
        agent.allow_switching = True
        agent.use_expert_switching = True

    elif p == "phase3":   # harder opponent, expert switching ON
        agent.allow_switching = True
        agent.use_expert_switching = True

    else:
        raise ValueError(f"Unknown phase: {phase}")

    # RL switching always OFF now
    agent.rl_switch_enabled = False

    # Epsilon configuration
    agent.epsilon = EPSILON_RESET[p]
    agent.epsilon_decay = EPSILON_DECAY[p]

    print(
        f"[CONFIG] Phase={p.upper()} | allow_switch={agent.allow_switching} | "
        f"expert_switch={agent.use_expert_switching}"
    )
    print(
        f"[CONFIG] Epsilon reset → {agent.epsilon:.3f}, decay={agent.epsilon_decay:.6f}"
    )


# ============================================================
# Opponent per phase
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
# MAIN TRAINING LOOP
# ============================================================
async def train_forever():

    # Load pool and generate fixed teams
    pool = load_pool("teams/team_pool.json")
    team_rl = generate_fixed_team(pool)
    team_ai = generate_fixed_team(pool)

    print("\n=== FIXED TRAINING TEAMS LOADED ===")

    # RL agent
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

    # Phase stats
    phase_index = TRAIN_PHASES.index(START_PHASE)
    phase = TRAIN_PHASES[phase_index]
    phase_finished = 0
    phase_wins = 0
    phase_batches = 0

    configure_agent_for_phase(rl_agent, phase)
    ai_agent = opponent_for_phase(phase, team_ai)

    # Rolling winrates
    winrates = deque(maxlen=ROLLING_WINDOW)
    cycle = 0

    # --------------------------
    # TRAINING LOOP
    # --------------------------
    while True:
        cycle += 1
        lifetime_batches += 1
        phase_batches += 1

        print(f"\n========= MINIBATCH {cycle} ({MINIBATCH} battles) =========")

        prev_finished = rl_agent.n_finished_battles
        prev_wins = rl_agent.n_won_battles

        # Run battles
        await rl_agent.battle_against(ai_agent, n_battles=MINIBATCH)

        batch_finished = rl_agent.n_finished_battles - prev_finished
        batch_wins = rl_agent.n_won_battles - prev_wins

        # Update lifetime stats
        lifetime_finished += batch_finished
        lifetime_wins += batch_wins

        # Update phase stats
        phase_finished += batch_finished
        phase_wins += batch_wins

        batch_winrate = batch_wins / batch_finished if batch_finished else 0
        winrates.append(batch_winrate)
        rolling = rolling_avg(winrates)

        print(f"[{phase}] Batch Result: {batch_wins}/{batch_finished} ({batch_winrate*100:.1f}%)")
        print(f"[{phase}] Epsilon now: {rl_agent.epsilon:.4f} (≈ {rl_agent.epsilon*100:.1f}% random actions)")
        print(f"[{phase}] Rolling Avg: {rolling*100:.1f}% ({len(winrates)} samples)")
        print(
            f"[{phase}] Phase record: {phase_wins}/{phase_finished} "
            f"({(phase_wins/phase_finished*100):.1f}%) over {phase_batches} batches"
        )
        print(
            f"Lifetime: {lifetime_wins}/{lifetime_finished} "
            f"({(lifetime_wins/lifetime_finished*100):.1f}%)"
        )

        # Training
        if rl_agent.episode_buffer:
            rl_agent._process_episodes()

        while len(rl_agent.experience_buffer) >= rl_agent.batch_size:
            rl_agent._train_batch()

        rl_agent.save_model()

        # Skipping
        if key_pressed() == MANUAL_SKIP_KEY:
            print(">>> Manual skip to next phase")
            enough_cycles, good_avg = True, True
        else:
            enough_cycles = phase_batches >= MIN_CYCLES[phase]
            good_avg = rolling >= THRESHOLDS[phase]

        too_long = phase_finished >= MAX_PHASE_BATTLES

        if (enough_cycles and good_avg) or too_long:

            print(f"\n===== PHASE {phase.upper()} COMPLETE =====")
            print(
                f"Final Phase Record: {phase_wins}/{phase_finished} "
                f"({(phase_wins/phase_finished*100):.1f}%)"
            )

            # Move to next phase
            phase_index += 1
            if phase_index >= len(TRAIN_PHASES):
                print("\n===== TRAINING COMPLETE =====")
                break

            phase = TRAIN_PHASES[phase_index]
            winrates.clear()

            phase_finished = 0
            phase_wins = 0
            phase_batches = 0

            print(f"\n=== ENTERING {phase.upper()} ===")
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