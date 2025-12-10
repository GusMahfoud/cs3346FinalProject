Reinforcement Learning for Pokémon Showdown (Mac Setup Guide)

This project trains and evaluates a reinforcement learning (RL) agent in the Pokémon Showdown battle simulator using the poke-env library.

It includes:
A local Pokémon Showdown server (Node.js)
A custom RL agent (MyRLAgent) in PyTorch
A high-dimensional state encoder and reward shaping
Scripts for training, evaluation, and a “visible” battle (model_video.py)

Important: Keep the Project Path Short

Pokémon Showdown uses Unix domain sockets for internal REPL processes, with a maximum path length.
If your project lives in a very deep directory (e.g. on your Desktop in several nested folders), you can see errors like:

CRASH: Error: listen EINVAL: invalid argument /Users/you/Desktop/Very/Deep/Path/.../pokemon-showdown/logs/repl/abusemonitor-remote-6864

The server often still works (you’ll see Worker 1 now listening on 0.0.0.0:8000), but to avoid this Clone the project into a short path, or if you already cloned somewhere deep and see listen EINVAL errors, just move the whole folder to a shorter path and rerun.

1. Prerequisites (Mac)

You’ll need:
Homebrew (recommended): https://brew.sh/
Git
Python 3.11 (3.13 can cause pip / library issues)
Node.js (LTS is fine)

From Terminal:
# Install git, Python 3.11, and Node.js
brew install git python@3.11 node
Check versions:
git --version
python3.11 --version
node --version
npm --version

2. Clone the Repository
Choose a short path (e.g. your home directory):
cd ~
git clone https://github.com/GusMahfoud/cs3346FinalProject.git cs3346FinalProject
cd cs3346FinalProject

3. Create and Activate a Virtual Environment (Python 3.11)

Create a venv:
python3.11 -m venv .venv

Activate it:
source .venv/bin/activate
# your prompt should now start with (.venv)

Verify Python version inside the venv:

python --version
# Expect: Python 3.11.x

4. Install Python Dependencies

First, upgrade pip tooling:
python -m pip install --upgrade pip setuptools wheel

Then install project requirements:
pip install -r requirements.txt

If for some reason requirements.txt is missing or incomplete, you at least need:
pip install torch poke-env tensorboard numpy

If you hit ModuleNotFoundError later for something like matplotlib or opencv-python, just install it in the same venv:

pip install matplotlib
pip install opencv-python

5. How the Showdown Server Is Managed

You do not need to manually clone or start Pokémon Showdown.
The script showdown_server.py automatically:
Clones pokemon-showdown into ./pokemon-showdown (if not already present).
Runs npm install inside pokemon-showdown.
Ensures config.js exists and sets exports.port = 8000.
Kills anything already running on port 8000.

Starts Showdown with:
node pokemon-showdown start --no-security


Polls http://localhost:8000 until the server responds, then prints:
Server running at ws://localhost:8000/
All training/eval scripts call start_showdown_server() for you.

6. Running a Visible Battle (model_video.py)

The model_video.py script:
Starts the local Showdown server
Loads a trained model (if available)
Launches a single “visible” battle
Prints a URL you can open in your browser to watch
From the project root, with the venv active:
python model_video.py


You should see output like:

[MAIN] Starting Pokémon Showdown server...
Verified: Showdown configured to use port 8000.
Starting Pokémon Showdown server...
...
Worker 1 now listening on 0.0.0.0:8000
Test your server at http://localhost:8000
Showdown responded — server ready.
Server running at ws://localhost:8000/

=== LOADING TEAMS ===

=== INIT RL AGENT (Phase 2A) ===
[RL] Model initialized with state_size=1500, lr=0.001
[RL] Loaded checkpoint (state_size=1500).
...
=== STARTING VISIBLE BATTLE ===
Open the URL shown below in your browser.
...
=== BATTLE COMPLETE ===
The browser tab stays open so you can inspect the full battle.


If a browser URL is printed, open it to see the battle UI. The script will keep the server alive until it finishes.

🔎 Note: You may still see some CRASH: Error: listen EINVAL: invalid argument ... logs/repl/... messages from Showdown due to Unix socket path length.
If they appear but are followed by Worker 1 now listening on 0.0.0.0:8000 and Showdown responded — server ready, the server is running and you can ignore those messages.
For fewer warnings, keep the project path short (e.g. ~/cs3346FinalProject).

Once you've clicked the URL, you should be brought to Pokemon Showdown. To watch the RLAgent battle click the "Ubers battle started between MYRLAgent1 and FixedOrderMaxBase1 text in the top right of the screen

7. Training the RL Agent (Optional)

To run the main training loop (curriculum RL):
python train_parallel.py

Typical behavior:
Starts the Showdown server
Loads the fixed training teams from teams/team_pool.json
Creates MyRLAgent with the configured model folder
Trains in phases (warmup, phase1, phase2a, phase2b, phase3b)
Prints batch winrates, rolling averages, epsilon, etc.

Saves checkpoints to models/<a2c_vx>/
Stop training with Ctrl+C. The script should terminate the Showdown server in its finally block.

8. Evaluating the Model (Optional)

You can then change the MODEL_FOLDER variable in the model_video.py file to point to your newly trained model

9. Common Issues & Fixes
9.1 ModuleNotFoundError: No module named 'torch' (or other libs)

Make sure the venv is active and install missing packages:

source .venv/bin/activate
pip install torch
# install others as needed

9.2 listen EINVAL: invalid argument ... logs/repl/...

This is due to long Unix socket paths inside pokemon-showdown/logs/repl/....
Mitigation:

Move the project to a shorter path, e.g.:

mv ~/Desktop/Fourth-Year/CS3346/cs3346FinalProject ~/cs3346FinalProject


Re-run your scripts from there.

As long as you see:

Worker 1 now listening on 0.0.0.0:8000
Showdown responded — server ready.
Server running at ws://localhost:8000/


the main server is functioning.

9.3 Weird pip errors (especially under Python 3.13)

If you see things like:

ModuleNotFoundError: No module named 'pip._vendor.packaging._structures'

you’re probably in a broken venv with Python 3.13.
Fix by deleting and recreating the venv with Python 3.11 (as shown above).