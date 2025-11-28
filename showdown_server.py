# showdown_server.py
import subprocess
import os
import time
import shutil
import urllib.request

# Location of Showdown data
SHOWDOWN_DIR = os.path.join(os.getcwd(), "pokemon-showdown")
SHOWDOWN_URL = "http://localhost:8000"


def ensure_showdown_checkout():
    """Clone repo + ensure config.js exists + update port."""
    if not os.path.exists(SHOWDOWN_DIR):
        print("Cloning Pokémon Showdown...")
        subprocess.run(
            ["git", "clone", "https://github.com/smogon/pokemon-showdown.git"],
            check=True,
        )

    # Install dependencies
    node_modules = os.path.join(SHOWDOWN_DIR, "node_modules")
    if not os.path.exists(node_modules):
        print("Running npm install...")
        subprocess.run(["npm", "install"], cwd=SHOWDOWN_DIR, check=True)

    # Ensure config.js exists
    config_dir = os.path.join(SHOWDOWN_DIR, "config")
    config_js = os.path.join(config_dir, "config.js")
    example_js = os.path.join(config_dir, "config-example.js")

    if not os.path.exists(config_js):
        print("Copying config config-example.js -> config.js")
        shutil.copy(example_js, config_js)

    # Force port = 8000
    with open(config_js, "r") as f:
        lines = f.readlines()

    with open(config_js, "w") as f:
        port_set = False
        for line in lines:
            if line.strip().startswith("exports.port"):
                f.write("exports.port = 8000;\n")
                port_set = True
            else:
                f.write(line)

        # If exports.port not found, add it
        if not port_set:
            f.write("\nexports.port = 8000;\n")

    print("Verified: Showdown configured to use port 8000.")


def wait_for_server(timeout=20.0, poll_interval=0.5):
    """Wait until the Showdown server is reachable."""
    start = time.time()
    last_err = None

    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(SHOWDOWN_URL, timeout=2):
                print("Showdown responded — server ready.")
                return True
        except Exception as e:
            last_err = e
            time.sleep(poll_interval)

    print(f"Warning: Showdown did not respond within {timeout}s.")
    if last_err:
        print("Last error:", last_err)
    return False


def start_showdown_server():
    """Start local Showdown server and wait for readiness."""
    ensure_showdown_checkout()

    # Ensure logs/repl directory exists (Showdown requires it)
    os.makedirs(os.path.join(SHOWDOWN_DIR, "logs", "repl"), exist_ok=True)

    print("Starting Pokémon Showdown server...")

    # Run from the PARENT directory of pokemon-showdown
    parent_dir = os.path.dirname(SHOWDOWN_DIR)

    proc = subprocess.Popen(
        ["node", "pokemon-showdown", "start", "--no-security"],
        cwd=SHOWDOWN_DIR,
        stdout=None,
        stderr=None,
    )



    # Let node initialize
    time.sleep(1.5)

    if proc.poll() is not None:
        raise RuntimeError(
            "Showdown crashed on startup. Ensure Node.js is installed and port 8000 is free."
        )

    wait_for_server()

    print("Server running at ws://localhost:8000/")
    return proc


if __name__ == "__main__":
    p = start_showdown_server()
    try:
        print("Showdown running. Ctrl+C to stop.")
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Shutting down Showdown...")
        p.terminate()
