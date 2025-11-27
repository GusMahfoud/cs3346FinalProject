import subprocess
import os
import time

def start_showdown_server():
    if not os.path.exists("pokemon-showdown"):
        print("Cloning Pokémon Showdown...")
        subprocess.run(["git", "clone", "https://github.com/smogon/pokemon-showdown.git"])

    print("Starting Pokémon Showdown server...")
    os.chdir("pokemon-showdown")

    subprocess.Popen(
        ["node", "pokemon-showdown", "start"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    time.sleep(3)
    print("Server ready at ws://localhost:8000\n")
    os.chdir("..")


if __name__ == "__main__":
    start_showdown_server()
