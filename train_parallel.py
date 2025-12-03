import asyncio
from poke_env.player.baselines import MaxBasePowerPlayer
from rl_agent import MyRLAgent
from randomizer.team_generator import load_pool, generate_team_from_pool

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

if __name__ == "__main__":
    asyncio.run(main())
