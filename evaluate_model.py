"""Evaluate a trained RL model against an opponent."""
import asyncio
from poke_env.player.baselines import MaxBasePowerPlayer
from rl_agent import MyRLAgent
from randomizer.team_generator import load_pool, generate_team_from_pool
from showdown_server import start_showdown_server

NUM_BATTLES = 100
MAX_PARALLEL = 16

async def main():
    # Load team pool
    pool = load_pool("teams/team_pool.json")
    
    # Generate teams
    team_rl = generate_team_from_pool(pool)
    team_ai = generate_team_from_pool(pool)
    
    # Load trained RL agent (will auto-load model if exists)
    rl_agent = MyRLAgent(
        battle_format="gen9ubers",
        team=team_rl,
        max_concurrent_battles=MAX_PARALLEL,
        epsilon_start=0.0,  # No exploration during evaluation
        epsilon_end=0.0
    )
    
    # Opponent AI
    ai_agent = MaxBasePowerPlayer(
        battle_format="gen9ubers",
        team=team_ai,
        max_concurrent_battles=MAX_PARALLEL
    )
    
    print(f"Evaluating trained model over {NUM_BATTLES} battles...")
    print("(Using epsilon=0.0 for pure exploitation)\n")
    
    await rl_agent.battle_against(ai_agent, n_battles=NUM_BATTLES)
    
    print("\n=== EVALUATION RESULTS ===")
    print(f"RL Agent Wins: {rl_agent.n_won_battles}/{rl_agent.n_finished_battles}")
    print(f"Win Rate: {rl_agent.n_won_battles / max(rl_agent.n_finished_battles, 1):.2%}")
    print(f"AI Agent Wins: {ai_agent.n_won_battles}")
    
    if hasattr(rl_agent, 'training_stats') and rl_agent.training_stats['battles_completed'] > 0:
        print(f"\nTraining Stats:")
        print(f"  Total Battles Trained: {rl_agent.training_stats['battles_completed']}")
        print(f"  Training Win Rate: {rl_agent.training_stats['wins'] / max(rl_agent.training_stats['battles_completed'], 1):.2%}")

if __name__ == "__main__":
    server_proc = start_showdown_server()
    try:
        asyncio.run(main())
    finally:
        print("\nShutting down Showdown server...")
        server_proc.terminate()

