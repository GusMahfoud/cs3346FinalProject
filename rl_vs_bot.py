# rl_vs_bot.py
import asyncio
from rl_agent import MyRLAgent
from poke_env.player.baselines import MaxBasePowerPlayer
from showdown_server import start_showdown_server


async def main():
    opponent = MaxBasePowerPlayer(battle_format="gen9randombattle")
    rl_agent = MyRLAgent(battle_format="gen9randombattle")

    await rl_agent.battle_against(opponent, n_battles=10)

    print(
        f"RLAgent: {rl_agent.n_won_battles}/{rl_agent.n_finished_battles} "
        f"won vs {opponent.username}"
    )


if __name__ == "__main__":
    server_proc = start_showdown_server()
    try:
        asyncio.run(main())
    finally:
        print("\nShutting down Showdown server...")
        server_proc.terminate()
