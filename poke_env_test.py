import asyncio
from poke_env.player import RandomPlayer
from showdown_server import start_showdown_server


async def main():
    p1 = RandomPlayer(battle_format="gen9randombattle")
    p2 = RandomPlayer(battle_format="gen9randombattle")

    await p1.battle_against(p2, n_battles=1)

    print(f"{p1.username}: {p1.n_won_battles}/{p1.n_finished_battles} won")
    print(f"{p2.username}: {p2.n_won_battles}/{p2.n_finished_battles} won")


if __name__ == "__main__":
    server_proc = start_showdown_server()
    try:
        asyncio.run(main())
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=3)
        except Exception:
            server_proc.kill()
