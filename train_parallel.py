import asyncio
from poke_env.player.baselines import MaxBasePowerPlayer
from rl_agent import MyRLAgent
from randomizer.team_generator import load_pool, generate_team_from_pool
from showdown_server import start_showdown_server

TOTAL_BATTLES = 1000
MAX_PARALLEL = 16   # poke-env handles parallel battles internally

async def main():
    # Load team pool
    pool = load_pool("teams/team_pool.json")

    # Generate teams
    team_rl = generate_team_from_pool(pool)
    team_ai = generate_team_from_pool(pool)

    # Set up agents with team strings
    rl_agent = MyRLAgent(
        battle_format="gen9ubers",
        team=team_rl,
        max_concurrent_battles=MAX_PARALLEL
    )

    ai_agent = MaxBasePowerPlayer(
        battle_format="gen9ubers",
        team=team_ai,
        max_concurrent_battles=MAX_PARALLEL
    )

    print("Starting training...")
    await rl_agent.battle_against(ai_agent, n_battles=TOTAL_BATTLES)

    print("\n=== RESULTS ===")
    print(f"RL Agent: {rl_agent.n_won_battles}/{rl_agent.n_finished_battles} won")
    print(f"AI Agent: {ai_agent.n_won_battles}")
    
    # Process any remaining episodes and train on remaining experiences
    if len(rl_agent.episode_buffer) > 0:
        rl_agent._process_episodes()
    
    if len(rl_agent.experience_buffer) > 0:
        print("Training on remaining experiences...")
        while len(rl_agent.experience_buffer) >= rl_agent.batch_size:
            rl_agent._train_batch()
    
    # Save trained model
    rl_agent.save_model("mlp_model.pth")

if __name__ == "__main__":
    server_proc = start_showdown_server()
    try:
        asyncio.run(main())
    finally:
        print("\nShutting down Showdown server...")
        server_proc.terminate()
