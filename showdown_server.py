# showdown_server.py
import subprocess
import os
import time
import shutil
import urllib.request

# This is from my base user directory because long file locations crash this dumb shit
SHOWDOWN_DIR = "pokemon-showdown"
SHOWDOWN_DIR = os.path.expanduser("~/ps-showdown")



def ensure_showdown_checkout():
    """Clone + basic setup for the Showdown repo."""
    if not os.path.exists(SHOWDOWN_DIR):
        print("Cloning Pokémon Showdown...")
        subprocess.run(
            ["git", "clone", "https://github.com/smogon/pokemon-showdown.git"],
            check=True,
        )

    # npm install if needed
    node_modules = os.path.join(SHOWDOWN_DIR, "node_modules")
    if not os.path.exists(node_modules):
        print("Running npm install...")
        subprocess.run(["npm", "install"], cwd=SHOWDOWN_DIR, check=True)

    # copy config-example.js -> config.js if needed
    config_dir = os.path.join(SHOWDOWN_DIR, "config")
    config_js = os.path.join(config_dir, "config.js")
    example_js = os.path.join(config_dir, "config-example.js")

    if not os.path.exists(config_js):
        print("Copying config/config-example.js -> config/config.js")
        shutil.copy(example_js, config_js)


def wait_for_server(timeout=20.0, poll_interval=0.5):
    """Wait until http://localhost:8000 responds, or timeout."""
    start = time.time()
    last_err = None

    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(SHOWDOWN_URL, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception as e:
            last_err = e
        time.sleep(poll_interval)

    print(f"Warning: Showdown did not respond within {timeout}s.")
    if last_err:
        print(f"Last error: {last_err}")
    return False


def start_showdown_server():
    """Start a local Showdown server suitable for poke-env."""
    ensure_showdown_checkout()

    print("Starting Pokémon Showdown server...")
    # Don't pipe stdout/stderr unless you plan to consume them
    proc = subprocess.Popen(
        ["node", "pokemon-showdown", "start", "--no-security"],
        cwd=SHOWDOWN_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Give Node a moment to crash if it's going to
    time.sleep(2.0)
    if proc.poll() is not None:
        raise RuntimeError("Showdown server process exited immediately. "
                           "Check that Node is installed and the port is free.")

    # Optional: actively wait for HTTP to come up
    wait_for_server()

    print("Server ready at ws://localhost:8000\n")
    return proc


if __name__ == "__main__":
    server_proc = start_showdown_server()
    try:
        # Keep the process alive if run directly
        print("Showdown running. Ctrl+C to exit.")
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Stopping Showdown...")
        server_proc.terminate()
