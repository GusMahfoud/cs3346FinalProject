# rl_vs_showdown_ai.py
import asyncio
from showdown_server import start_showdown_server

from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder
from poke_env.player.baselines import MaxBasePowerPlayer

from rl_agent import MyRLAgent
from randomizer.team_generator import generate_two_random_teams


async def main():
    # Load teams
    team_rl_str, team_ai_str = generate_two_random_teams("teams/team_pool.json")

    print("\n=== RL Agent Team ===\n", team_rl_str)
    print("\n=== Showdown AI Team ===\n", team_ai_str)

    # Build RL agent (NO server config parameter)
    rl_agent = MyRLAgent(battle_format="gen9randombattle")


    # Build heuristic AI
    ai_agent = MaxBasePowerPlayer(battle_format="gen9randombattle")


    print("\nStarting RL Agent vs Heuristic AI...\n")
    await rl_agent.battle_against(ai_agent, n_battles=1)

    print("\n=== RESULTS ===")
    print(f"RL Agent wins: {rl_agent.n_won_battles}")
    print(f"AI Agent wins: {ai_agent.n_won_battles}")


if __name__ == "__main__":
    proc = start_showdown_server()
    try:
        asyncio.run(main())
    finally:
        print("\nShutting down Showdown server...")
        proc.terminate()
