# Pokémon Battle RL Agent

A reinforcement learning agent that learns to play Pokémon battles using **weight-based policy** and **REINFORCE** algorithm.

## Overview

- **Learning**: Policy gradient (REINFORCE)
- **Actions**: 7 heuristic-based strategic actions
- **Training**: Hybrid approach (bot opponents → self-play)
- **Simulator**: `poke-env` (Pokémon Showdown Python library)

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# or: source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up Local Pokémon Showdown Server

**IMPORTANT**: `poke-env` requires a Pokémon Showdown server to run battles.

See `SETUP_SERVER.md` for detailed instructions. Quick steps:

1. Install Node.js (https://nodejs.org/)
2. Clone Pokémon Showdown: `git clone https://github.com/smogon/pokemon-showdown.git`
3. Install dependencies: `cd pokemon-showdown && npm install`
4. Start server: `node pokemon-showdown start --no-security`

**Keep the server running** in a separate terminal while training!

### 3. Train Agent

```bash
python -m src.trainer
```

This will:
- Train against Random bot (300 episodes)
- Train against stronger bot (300 episodes)
- Self-play training (400 episodes)

### 4. Evaluate Agent

```bash
python -m src.evaluator
```

## Project Structure

```
cs3346FinalProject/
├── src/
│   ├── agent.py          # RL agent with weight-based policy
│   ├── actions.py        # Heuristic action definitions
│   ├── state_features.py # State feature extraction
│   ├── trainer.py        # Training loop
│   └── evaluator.py      # Evaluation
├── config.yaml           # Hyperparameters
├── requirements.txt      # Dependencies
└── README.md
```

## Configuration

Edit `config.yaml` to adjust:
- Learning rate
- Exploration (epsilon)
- Rewards (win/loss/penalties)
- Training phases

## How It Works

1. **State**: Extract features from battle (HP, types, status, etc.)
2. **Actions**: 7 strategic actions (e.g., "use best type-effective move")
3. **Policy**: Weight-based selection `P(a|s) = softmax(θ[a] · φ(s))`
4. **Learning**: REINFORCE updates weights based on episode rewards

## CS 3346 Alignment

✅ **MDP Formulation**: States, actions, rewards, policy  
✅ **Policy Gradient**: REINFORCE algorithm  
✅ **Feature Vectors**: State representation φ(s)  
✅ **Discrete Actions**: Heuristic action space  
✅ **Reward Shaping**: Win/loss + step penalties

## Troubleshooting

**Issue**: `poke-env` installation fails  
**Solution**: Make sure you have Python 3.8+ and pip is up to date

**Issue**: Training is slow  
**Solution**: Reduce `num_episodes` in `config.yaml` for testing

**Issue**: Agent not learning  
**Solution**: Check learning rate, try different reward shaping

