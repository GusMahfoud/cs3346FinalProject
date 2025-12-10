# train_parallel.py  --  Curriculum RL: warmup → phase3a → phase3b
#                        tuned for 5-action policy + heuristic boss
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
MODEL_FOLDER = "models/a2c_v42"   # bump version

ROLLING_WINDOW = 20
MAX_PHASE_BATTLES = 12000
MANUAL_SKIP_KEY = "6"

# ============================================================
# PHASE ORDER
# ============================================================

TRAIN_PHASES = [
    "warmup",   # no switching, just learn KO / damage / basic play
    "phase3a",  # expert switching vs heuristic boss
    "phase3b",  # RL switching vs heuristic boss
]

START_PHASE = "warmup"

# ============================================================
# Phase Difficulty Thresholds
# - We keep them reasonable so it doesn't advance while still clueless.
# ============================================================

THRESHOLDS = {
    "warmup": 0.85,  # must crush MaxBasePower first
    "phase3a": 0.70, # decent vs heuristic with expert switching
    "phase3b": 0.62, # slightly above 60% vs heuristic on its own
}

# ============================================================
# Minimum cycles before progress
# ============================================================

MIN_CYCLES = {
    "warmup": 8,
    "phase3a": 80,   # enough time to copy expert switching patterns
    "phase3b": 120,  # longer grinding vs boss with RL switching
}

# ============================================================
# Epsilon schedules (5-action policy)
# ============================================================

EPSILON_START = {
    "warmup": 1.0,
    "phase3a": 0.75,   # more exploration while expert handles switches
    "phase3b": 0.75,   # lower so it doesn't destroy learned policy
}

EPSILON_END = {
    "warmup": 0.05,
    "phase3a": 0.08,
    "phase3b": 0.05,
}

EPSILON_DECAY = {
    "warmup": 0.9990,
    "phase3a": 0.9991,
    "phase3b": 0.9987,
}

# ============================================================
# LR & ENTROPY (for 5-action A2C/PPO-like agent)
# ============================================================

LR_PER_PHASE = {
    "warmup": 1e-3,     # learn basics quickly
    "phase3a": 7e-4,    # more careful vs heuristic + expert
    "phase3b": 4e-4,    # conservative while learning RL switching
}

ENTROPY_PER_PHASE = {
    "warmup": 0.01,
    "phase3a": 0.015,   # encourage move exploration while expert switches
    "phase3b": 0.012,   # slightly more exploitation vs boss
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
        # No switching: learn moves/KO fundamentals
        agent.allow_switching = False
        agent.use_expert_switching = False
        agent.rl_switch_enabled = False

    elif p == "phase3a":
        # Expert switching vs heuristic boss:
        #   RL chooses moves, expert decides WHEN/WHO to switch.
        agent.allow_switching = True
        agent.use_expert_switching = True
        agent.rl_switch_enabled = False

    elif p == "phase3b":
        # Full RL switching vs SAME heuristic boss:
        #   RL chooses both move vs switch and which switch.
        agent.allow_switching = True
        agent.use_expert_switching = False
        agent.rl_switch_enabled = True

    else:
        raise ValueError(f"Unknown phase: {phase}")

    # Epsilon reset
    agent.epsilon = EPSILON_START[p]
    agent.epsilon_start = EPSILON_START[p]
    agent.epsilon_end = EPSILON_END[p]
    agent.epsilon_decay = EPSILON_DECAY[p]

    # Phase-specific LR & entropy
    agent.set_lr(LR_PER_PHASE[p])
    agent.set_entropy_coef(ENTROPY_PER_PHASE[p])

    print(
        f"[CONFIG] Phase={p.upper()} | allow={agent.allow_switching} | "
        f"expert={agent.use_expert_switching} | rl_switch={agent.rl_switch_enabled}"
    )
    print(
        f"[CONFIG] ε={agent.epsilon_start:.2f}→{agent.epsilon_end:.2f} "
        f"decay={agent.epsilon_decay}, LR={LR_PER_PHASE[p]:.1e}, "
        f"entropy={ENTROPY_PER_PHASE[p]:.3f}"
    )

# ============================================================
# Opponent selection
# ============================================================

def opponent_for_phase(phase: str, team_ai):
    p = phase.lower()

    if p == "warmup":
        # Strong but dumb: Max BP teaches KO patterns
        return FixedOrderMaxBasePower(
            battle_format="gen9ubers",
            max_concurrent_battles=MAX_PARALLEL,
            team=team_ai,
        )

    if p in ["phase3a", "phase3b"]:
        # Same heuristic boss for both:
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

    # Start at warmup
    phase_index = TRAIN_PHASES.index(START_PHASE)
    phase = TRAIN_PHASES[phase_index]

    configure_agent_for_phase(rl_agent, phase)
    ai_agent = opponent_for_phase(phase, team_ai)

    # Rolling stats
    winrates = deque(maxlen=ROLLING_WINDOW)
    phase_finished = 0
    phase_wins = 0
    phase_batches = 0
    cycle = 0

    # Training loop
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