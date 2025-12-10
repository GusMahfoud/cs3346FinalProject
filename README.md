Reinforcement Learning for Pokémon Showdown (Setup Guide)
========================================================

This project trains and evaluates a reinforcement learning (RL) agent in the Pokémon Showdown battle simulator using the poke-env library.

It includes:

- A local Pokémon Showdown server (Node.js)
- A custom RL agent (MyRLAgent) in PyTorch
- A high-dimensional state encoder and reward shaping
- Scripts for training, evaluation, and a “visible” battle (`model_video.py`)

**Important: Keep the Project Path Short (All OSes)**

Pokémon Showdown uses Unix domain sockets for internal REPL processes, which have a maximum path length (even on Windows, because Node.js emulates them).  
If your project lives in a deeply nested folder, you may see:

CRASH: Error: listen EINVAL: invalid argument ... logs/repl/...

The server often still works, but to avoid this:

**Use a short path:**

- macOS: `~/cs3346FinalProject`
- Windows: `C:\cs3346FinalProject`

If you're already in a deep path, simply move the entire folder to a shorter root-level directory.

---

# MacOS Setup Guide

### 1. Prerequisites (Mac)

You’ll need:

- Homebrew (recommended): https://brew.sh/
- Git  
- Python 3.11 (Python 3.13 can cause pip / library issues)  
- Node.js (LTS is fine)

Install dependencies:

brew install git python@3.11 node

Check versions:

git --version

python3.11 --version

node --version

npm --version

---

### 2. Clone the Repository

cd ~

git clone https://github.com/GusMahfoud/cs3346FinalProject.git

cd cs3346FinalProject

### 3. Create and Activate a Virtual Environment (Python 3.11)

python3.11 -m venv .venv

source .venv/bin/activate

python --version

Expect: `Python 3.11.x`

---

### 4. Install Python Dependencies

python -m pip install --upgrade pip setuptools wheel

pip install -r requirements.txt

If needed manually:

pip install torch poke-env tensorboard numpy matplotlib opencv-python

---

### 5. How the Showdown Server Is Managed

`showdown_server.py` automatically:

- clones Pokémon Showdown  
- runs `npm install`  
- writes `config.js`  
- frees port 8000  
- launches the server  
- waits until ready  

You never need to manually touch the server repo.

---

### 6. Running a Visible Battle (`model_video.py`)

python model_video.py

Look for:

Showdown responded — server ready.

Server running at ws://localhost:8000/

Then open the printed URL and click the battle in the top-right.(Watch video for clarifcation)

`listen EINVAL` → move project to shallower path.

---

### 7. Training the RL Agent (Optional)

python train_parallel.py

Stop with **Ctrl + C**.

---

### 8. Evaluating the Model (Optional)

Change inside `model_video.py`:

MODEL_FOLDER = "models/a2c_vX", where X is your newly trained version

---

# Windows Setup Guide

Windows is similar to macOS, but long-path issues are more common — keep project shallow.

---

### 1. Prerequisites (Windows)

You’ll need:

- Git → https://git-scm.com/download/win  
- Python 3.11 → https://www.python.org/downloads/release/python-3110/  
  -  Check **Add Python to PATH**  
- Node.js LTS → https://nodejs.org/

Verify versions:

git --version

python --version

node --version

npm --version


---

### 2. Clone the Repository (Windows)

cd C:

git clone https://github.com/GusMahfoud/cs3346FinalProject.git

cd cs3346FinalProject

---

### 3. Create and Activate Virtual Environment (Python 3.11)

python -m venv .venv

.venv\Scripts\activate

python --version

Deactivate:

deactivate


---

### 4. Install Python Dependencies

python -m pip install --upgrade pip setuptools wheel

pip install -r requirements.txt

Or manually:

pip install torch poke-env tensorboard numpy matplotlib opencv-python

---

### 5. How the Showdown Server Works (Windows)

`showdown_server.py`:
- clones `pokemon-showdown/`
- runs `npm install`
- writes `config.js`
- frees port 8000
- launches: node pokemon-showdown start --no-security

---

### 6. Running a Visible Battle (`model_video.py`)

python model_video.py

Expect output:

Showdown responded — server ready.

Server running at ws://localhost:8000/

Long paths → `listen EINVAL`  

Fix: move to `C:\cs3346FinalProject`

---

### 7. Training the RL Agent (Optional)

python train_parallel.py

Stop with **Ctrl + C**.

---

### 8. Evaluating the Model (Optional)

Set folder in `model_video.py`:

MODEL_FOLDER = "models/a2c_vX", where X is your newly trained model

---

### 9. Common Issues (Windows)

**Torch installation problems:**

pip install torch --index-url https://download.pytorch.org/whl/cpu

**“npm not recognized”:** restart terminal.

**Path-too-long crashes:** move project to root.