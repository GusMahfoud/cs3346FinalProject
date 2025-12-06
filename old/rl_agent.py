from poke_env.player.player import Player

import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from collections import deque

# Correct engine modules (found on your system)
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.move import Move
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.weather import Weather
from poke_env.battle.field import Field

# And you SHOULD also import Battle from the same directory:
from poke_env.battle.battle import Battle

from rewards import RewardCalculator

# -----------------------------
# CONSTANTS
# -----------------------------

STATUS_LIST = ["brn", "par", "slp", "tox", "frz"]

TYPES = [
    "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison",
    "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark",
    "steel", "fairy"
]
TYPE_INDEX = {t: i for i, t in enumerate(TYPES)}

BOOSTS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]

# Weather and terrain maps – only snow + grassy actually appear in our 20-mon pool,
# but we keep them generic so this stays usable if you expand later.
WEATHER_KEYS = [
    Weather.SUNNYDAY,
    Weather.RAINDANCE,
    Weather.SANDSTORM,
    Weather.HAIL,       # or Weather.SNOW in newer gens; poke-env maps appropriately
]
TERRAIN_KEYS = [
    Field.GRASSY_TERRAIN,
    Field.ELECTRIC_TERRAIN,
    Field.MISTY_TERRAIN,
    Field.PSYCHIC_TERRAIN,
]



# -----------------------------
# PER-POKÉMON ENCODING
# -----------------------------

def encode_pokemon(poke):
    """Encode one Pokémon into a fixed-length feature vector."""
    vec = []

    # Active / fainted
    vec.append(1.0 if getattr(poke, "active", False) else 0.0)
    vec.append(1.0 if poke.fainted else 0.0)

    # HP fraction
    if poke.max_hp and poke.current_hp is not None:
        vec.append(poke.current_hp / poke.max_hp)
    else:
        vec.append(0.0)

    # Status (one-hot)
    status_vec = [0] * len(STATUS_LIST)
    if poke.status:
        # poke.status is a Status enum -> use name.lower()
        name = str(poke.status).lower()
        if name in STATUS_LIST:
            status_vec[STATUS_LIST.index(name)] = 1
    vec.extend(status_vec)

    # Types (one-hot each)
    type1 = [0] * len(TYPES)
    type2 = [0] * len(TYPES)
    if poke.type_1 in TYPE_INDEX:
        type1[TYPE_INDEX[poke.type_1]] = 1
    if poke.type_2 in TYPE_INDEX:
        type2[TYPE_INDEX[poke.type_2]] = 1
    vec.extend(type1)
    vec.extend(type2)

    # Stat boosts
    for b in BOOSTS:
        boost_value = poke.boosts.get(b, 0)
        vec.append(boost_value / 6.0)

    # Moves: up to 4 slots
    moves_list = list(poke.moves.values())
    for i in range(4):
        if i < len(moves_list):
            move = moves_list[i]

            # Move exists
            vec.append(1.0)

            # PP fraction
            if move.max_pp:
                vec.append(move.current_pp / move.max_pp)
            else:
                vec.append(0.0)

            # Base power normalized (0 if purely status)
            bp = move.base_power or 0
            vec.append(bp / 250.0)

            # Move type one-hot
            type_vec = [0] * len(TYPES)
            if move.type in TYPE_INDEX:
                type_vec[TYPE_INDEX[move.type]] = 1
            vec.extend(type_vec)
        else:
            # Empty slot: no move
            vec.extend([0.0, 0.0, 0.0] + [0.0] * len(TYPES))

    return vec

# -----------------------------
# SIDE CONDITION HELPERS
# -----------------------------

def encode_side_conditions(sc_dict):
    """
    Encode side conditions for one side (ours or opponent's).
    Returns a small fixed-length vector.
    Only Stealth Rock, Spikes and Future Sight are relevant for our 20-mon pool.
    """
    vec = []

    # Stealth Rock: boolean
    rocks_active = 1.0 if SideCondition.STEALTH_ROCK in sc_dict else 0.0
    vec.append(rocks_active)

    # Spikes: layers 0-3 normalized
    spikes_layers = sc_dict.get(SideCondition.SPIKES, 0)
    vec.append(spikes_layers / 3.0)


    return vec

def encode_weather_and_terrain(battle):
    """
    Encode global weather and terrain into a one-hot + duration vector.
    For this pool:
      - Weather: Snow can be set by Chilly Reception (G-Slowking)
      - Terrain: Grassy Terrain from Rillaboom's Grassy Surge
    """
    vec = []

    # Weather one-hot + normalized duration
    weather_one_hot = [0.0] * len(WEATHER_KEYS)
    weather_duration = 0.0
    if battle.weather:
        # Handle both tuple (weather, turns) and single weather value
        if isinstance(battle.weather, tuple) and len(battle.weather) == 2:
            w, turns = battle.weather
        else:
            # Just weather value, no turns info
            w = battle.weather
            turns = 0  # Unknown duration
        if w in WEATHER_KEYS:
            weather_one_hot[WEATHER_KEYS.index(w)] = 1.0
        weather_duration = min(turns, 8) / 8.0
    vec.extend(weather_one_hot)
    vec.append(weather_duration)

    # Terrain one-hot + normalized duration
    terrain_one_hot = [0.0] * len(TERRAIN_KEYS)
    terrain_duration = 0.0
    if battle.fields:
        for f, turns in battle.fields.items():
            if f in TERRAIN_KEYS:
                terrain_one_hot[TERRAIN_KEYS.index(f)] = 1.0
                terrain_duration = min(turns, 8) / 8.0
                break
    vec.extend(terrain_one_hot)
    vec.append(terrain_duration)

    return vec

# -----------------------------
# FULL BATTLE STATE ENCODING
# -----------------------------

def encode_state(battle):
    """Encode a full battle state into a vector."""
    vec = []

    # Our 6 Pokémon
    for poke in battle.team.values():
        vec.extend(encode_pokemon(poke))

    # Opponent's 6 Pokémon (public info only – unrevealed stays mostly zeroed)
    for poke in battle.opponent_team.values():
        vec.extend(encode_pokemon(poke))

    # Our side conditions (Stealth Rock, Spikes, Future Sight)
    vec.extend(encode_side_conditions(battle.side_conditions))

    # Opponent side conditions
    vec.extend(encode_side_conditions(battle.opponent_side_conditions))

    # Global weather + terrain
    vec.extend(encode_weather_and_terrain(battle))

    # Trick Room – not present in this 20-mon sample, but cheap and generic
    # Trick Room (pseudoWeather stored in battle.fields)
    tr = battle.fields.get("trickroom", 0)
    vec.append(1.0 if tr > 0 else 0.0)
    vec.append(min(tr, 8) / 8.0)    # duration normalization


    # Turn number normalized
    vec.append(battle.turn / 100.0)

    return np.array(vec, dtype=np.float32)

# -----------------------------
# MLP MODEL
# -----------------------------

class MLPPolicy(nn.Module):
    """Simple MLP for policy learning."""
    
    def __init__(self, state_size, action_size=10, hidden_size=512):
        super(MLPPolicy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
        
    def forward(self, state):
        """Forward pass: state -> action logits."""
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

# -----------------------------
# RL AGENT
# -----------------------------

class MyRLAgent(Player):
    """
    Base RL-ready agent with:
    - 10 discrete actions (4 moves + 6 switch options)
    - Rich game state encoding tailored to the 20-Pokémon mini-meta
    - Random policy by default (plug your model into compute_policy)
    - Integrated rewards system for RL training
    """

    def __init__(
        self,
        battle_format="gen9ubers",
        learning_rate=0.0001,      # safer, more stable
        batch_size=128,            # more samples per update
        gamma=0.99,                # longer credit assignment
        epsilon_start=1.0,
        epsilon_end=0.10,
        epsilon_decay=0.9997,      # VERY slow decay
        model_path=None,
        **kwargs
    ):

        super().__init__(battle_format=battle_format, **kwargs)
        self.action_size = 10
        self.reward_calc = RewardCalculator()
        self.battle_history = []  # Store (state, action, reward) tuples for current battle
        self.episode_buffer = []  # Store complete episodes (battles) for training
        self.experience_buffer = []  # Store processed (state, action, return) tuples for batch training
        self.prev_battle_state = None
        self.current_reward_score = 0.0
        self.last_action_was_switch = False
        
        # MLP model setup
        # Calculate state size dynamically (will be set on first encode_state call)
        self.state_size = None
        self.model = None
        self.optimizer = None
        self.batch_size = batch_size
        self.gamma = gamma  # Discount factor for returns
        self.learning_rate = learning_rate
        
        # Epsilon-greedy exploration
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Try to load existing model
        if model_path:
            # Always check the models/ directory, since save_model() saves there
            full_path = os.path.join("models", os.path.basename(model_path))

            if os.path.exists(full_path):
                print(f"Model found: {full_path}")
                self.load_model(os.path.basename(model_path))
            else:
                print(f"Model not found at {full_path}, creating new model")

        else:
            # If no model_path provided, use the default path
            default_path = os.path.join("models", "mlp_model.pth")

            if os.path.exists(default_path):
                print(f"Model found: {default_path}")
                self.load_model("mlp_model.pth")
            else:
                print(f"Model not found at {default_path}, creating new model")

        
        # Training stats
        self.training_stats = {
            'battles_completed': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'total_reward': 0.0,
            'avg_reward_per_battle': [],
            'win_rate_history': []
        }

    # -------------------------
    # POLICY / ACTION SELECTION
    # -------------------------

    def _init_model(self, state_size, silent=False):
        """Initialize MLP model on first use."""
        if self.model is None:
            self.state_size = state_size
            self.model = MLPPolicy(state_size, self.action_size, hidden_size=512)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.model.train()
            if not silent:
                print(f"Created new model (state_size={state_size}, hidden_size=512)")
    
    def compute_policy(self, state_vec: np.ndarray) -> np.ndarray:
        """
        Compute policy using MLP.
        Returns action probabilities.
        """
        # Initialize model on first call with max expected size
        if self.model is None:
            # Use a large fixed size to handle variable state vectors
            # Pad/truncate to this size
            max_state_size = 2000  # Large enough for any battle state
            self._init_model(max_state_size)
        
        # Pad or truncate state vector to match model input size
        current_size = len(state_vec)
        if current_size < self.state_size:
            # Pad with zeros
            padded = np.zeros(self.state_size, dtype=np.float32)
            padded[:current_size] = state_vec
            state_vec = padded
        elif current_size > self.state_size:
            # Truncate
            state_vec = state_vec[:self.state_size]
        
        # Convert to tensor and forward pass
        self.model.eval()
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(state_tensor)
            probs = torch.softmax(logits, dim=1)
        
        return probs.squeeze(0).numpy()

    def reset_team(self, team_string):
        self.set_team(team_string)   # correct way to update the team


    def choose_move(self, battle):
        """Main hook called by poke-env to choose an action."""
        # Compute reward from previous turn if we have previous state
        if self.prev_battle_state is not None and not battle.finished:
            turn_reward = self.reward_calc.compute_turn_reward(
                self.prev_battle_state,
                battle,
                action_was_switch=self.last_action_was_switch
            )
            self.current_reward_score += turn_reward
            
            # Store experience: (prev_state, action, reward)
            if len(self.battle_history) > 0:
                # Update last entry with reward
                prev_state, prev_action, _ = self.battle_history[-1]
                self.battle_history[-1] = (prev_state, prev_action, turn_reward)
        
        # Encode current state
        state_vec = encode_state(battle)
        legal = self.legal_actions(battle)
        
        # Ensure model is initialized (needed even for exploration to track state)
        if self.model is None:
            max_state_size = 2000
            self._init_model(max_state_size)
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Explore: random legal action
            action_index = np.random.choice(legal) if legal else 0
            move = self.action_to_move(action_index, battle)
        else:
            # Exploit: use policy
            policy = self.compute_policy(state_vec)
            
            # Mask illegal actions
            masked_policy = policy.copy()
            for i in range(self.action_size):
                if i not in legal:
                    masked_policy[i] = 0.0
            
            # Normalize and sample from policy
            if masked_policy.sum() > 0:
                masked_policy = masked_policy / masked_policy.sum()
                action_index = np.random.choice(self.action_size, p=masked_policy)
            else:
                # Fallback if all actions masked
                action_index = np.random.choice(legal) if legal else 0

            if action_index not in legal:
                # Fallback: random legal action
                move = self.choose_random_legal_action(battle)
                action_index = self.legal_actions(battle)[0] if self.legal_actions(battle) else 0
            else:
                move = self.action_to_move(action_index, battle)
        
        # Store state and action for this turn (reward will be added next turn)
        self.battle_history.append((state_vec.copy(), action_index, 0.0))
        
        # Track if this is a switch action
        self.last_action_was_switch = (action_index >= 4)
        
        # Store current battle state for next turn (shallow copy key attributes)
        self.prev_battle_state = self._copy_battle_state(battle)
        
        return move
    
    def _copy_battle_state(self, battle):
        """Create a minimal copy of battle state for reward calculation."""
        # Store key metrics we need for reward calculation
        # This avoids issues with battle object mutation
        state = {
            'turn': battle.turn,
            'team': {k: {
                'current_hp': v.current_hp,
                'max_hp': v.max_hp,
                'fainted': v.fainted,
                'status': v.status
            } for k, v in battle.team.items()},
            'opponent_team': {k: {
                'current_hp': v.current_hp if v.current_hp is not None else 0,
                'max_hp': v.max_hp if v.max_hp else 1,
                'fainted': v.fainted,
                'status': v.status
            } for k, v in battle.opponent_team.items()}
        }
        return state

    # -------------------------
    # ACTION SPACE / MAPPING
    # -------------------------

    def legal_actions(self, battle):
        """
        Return list of legal action indices in [0, 9]:
        - 0–3: move slots
        - 4–9: switch slots
        """
        indices = []

        # Moves
        for i, move in enumerate(battle.available_moves):
            if move is not None and i < 4:
                indices.append(i)

        # Switches
        for i, poke in enumerate(battle.available_switches):
            if poke is not None and i < 6:
                indices.append(4 + i)

        return indices

    def action_to_move(self, action, battle):
        """Map an integer action index to a poke-env Order object."""
        # 0–3: use move in slot (if exists)
        if 0 <= action <= 3:
            moves = battle.available_moves
            if action < len(moves) and moves[action] is not None:
                return self.create_order(moves[action])
            # Illegal or missing -> fallback
            return self.choose_random_legal_action(battle)

        # 4–9: switch to team slot (if exists)
        if 4 <= action <= 9:
            idx = action - 4
            switches = battle.available_switches
            if idx < len(switches) and switches[idx] is not None:
                return self.create_order(switches[idx])
            return self.choose_random_legal_action(battle)

        # Completely out-of-range? Just do something legal.
        return self.choose_random_legal_action(battle)

    def choose_random_legal_action(self, battle):
        """Sample uniformly from currently legal actions."""
        legal = self.legal_actions(battle)
        if not legal:
            # Let poke-env pick something completely random as a last resort
            return self.choose_random_move(battle)
        a = np.random.choice(legal)
        move = self.action_to_move(a, battle)
        self.last_action_was_switch = (a >= 4)
        return move
    
    def on_battle_end(self, battle):
        """Called when battle ends — compute final reward, update episode buffer, 
        train, update exploration schedule, and reset state."""

        # ---------------------------------------------------
        # 1) Final turn reward (if previous state exists)
        # ---------------------------------------------------
        if self.prev_battle_state is not None:
            turn_reward = self.reward_calc.compute_turn_reward(
                self.prev_battle_state,
                battle,
                action_was_switch=self.last_action_was_switch
            )
            self.current_reward_score += turn_reward
            
            if len(self.battle_history) > 0:
                prev_state, prev_action, _ = self.battle_history[-1]
                self.battle_history[-1] = (prev_state, prev_action, turn_reward)

        # ---------------------------------------------------
        # 2) Terminal win/loss/draw reward
        # ---------------------------------------------------
        terminal_reward = self.reward_calc.compute_terminal_reward(battle)
        self.current_reward_score += terminal_reward

        if len(self.battle_history) > 0:
            prev_state, prev_action, prev_r = self.battle_history[-1]
            self.battle_history[-1] = (prev_state, prev_action, prev_r + terminal_reward)

        # ---------------------------------------------------
        # 3) Add completed episode to buffer + compute returns
        # ---------------------------------------------------
        if len(self.battle_history) > 0:
            self.episode_buffer.append(self.battle_history.copy())
            self._process_episodes()

        # ---------------------------------------------------
        # 4) Update training statistics
        # ---------------------------------------------------
        self.training_stats['battles_completed'] += 1

        if battle.won:
            self.training_stats['wins'] += 1
        elif battle.lost:
            self.training_stats['losses'] += 1
        else:
            self.training_stats['draws'] += 1

        self.training_stats['total_reward'] += self.current_reward_score
        self.training_stats['avg_reward_per_battle'].append(self.current_reward_score)

        win_rate = self.training_stats['wins'] / self.training_stats['battles_completed']
        self.training_stats['win_rate_history'].append(win_rate)

        # ---------------------------------------------------
        # 5) IMPROVED EPSILON DECAY SCHEDULE (Patch #1)
        # ---------------------------------------------------
        # Exponential decay early, shallow decay later
        # Ensures exploration doesn't vanish too early
        decay_factor = self.epsilon_decay ** (1 + self.training_stats['battles_completed'] / 5000)
        self.epsilon = max(self.epsilon_end, self.epsilon * decay_factor)

        # ---------------------------------------------------
        # 6) Train if batch ready
        # ---------------------------------------------------
        if len(self.experience_buffer) >= self.batch_size:
            self._train_batch()

        # ---------------------------------------------------
        # 7) Logging improvements
        # ---------------------------------------------------
        if self.training_stats['battles_completed'] % 50 == 0:
            print(
                f"Battles: {self.training_stats['battles_completed']}, "
                f"Win Rate: {win_rate:.2%}, "
                f"Epsilon: {self.epsilon:.4f}, "
                f"Reward (last): {self.current_reward_score:.2f}"
            )

        # ---------------------------------------------------
        # 8) Reset per-battle values
        # ---------------------------------------------------
        self.reward_calc.reset()
        self.prev_battle_state = None
        self.current_reward_score = 0.0
        self.last_action_was_switch = False
        self.battle_history = []

    
    def _train_batch(self):
        """Train MLP on a batch of experiences using REINFORCE + entropy bonus + gradient clipping."""
        if self.model is None or len(self.experience_buffer) < self.batch_size:
            return
        
        # Sample batch (already has computed returns)
        batch = self.experience_buffer[:self.batch_size]
        self.experience_buffer = self.experience_buffer[self.batch_size:]
        
        # Extract states, actions, and returns
        states = []
        actions = []
        returns = []
        
        for state, action, return_val in batch:
            # Pad/truncate state to match model input size
            state = np.array(state, dtype=np.float32)
            current_size = len(state)

            if current_size < self.state_size:
                padded = np.zeros(self.state_size, dtype=np.float32)
                padded[:current_size] = state
                state = padded
            elif current_size > self.state_size:
                state = state[:self.state_size]
            
            states.append(state)
            actions.append(action)
            returns.append(return_val)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(actions)
        returns_tensor = torch.FloatTensor(returns)
        
        # Normalize returns (baseline)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        
        # Forward pass
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(states_tensor)
        probs = torch.softmax(logits, dim=1)
        
        # REINFORCE loss
        log_probs = torch.log(probs + 1e-8)
        selected_log_probs = log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        reinforce_loss = -(selected_log_probs * returns_tensor).mean()
        
        # ---------------------------------------------------
        # 🔥 ENTROPY BONUS (encourages exploration)
        # ---------------------------------------------------
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        entropy_bonus = 0.01 * entropy
        
        # TOTAL LOSS
        loss = reinforce_loss - entropy_bonus
        
        # ---------------------------------------------------
        # GRADIENT CLIPPING (stability)
        # ---------------------------------------------------
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        print(
            f"Training batch | Loss: {loss.item():.4f} | "
            f"Entropy: {entropy.item():.4f} | "
            f"Epsilon: {self.epsilon:.3f}"
        )

        self.model.eval()

    def save_results(self, filename="training_results.json"):
        """Save training statistics to JSON file."""
        if self.training_stats['battles_completed'] == 0:
            print("Warning: No battles completed. Results file will contain zeros.")
        
        results = {
            'total_battles': self.training_stats['battles_completed'],
            'wins': self.training_stats['wins'],
            'losses': self.training_stats['losses'],
            'draws': self.training_stats['draws'],
            'win_rate': self.training_stats['wins'] / max(self.training_stats['battles_completed'], 1),
            'total_reward': self.training_stats['total_reward'],
            'avg_reward_per_battle': np.mean(self.training_stats['avg_reward_per_battle']) if self.training_stats['avg_reward_per_battle'] else 0.0,
            'win_rate_history': self.training_stats['win_rate_history'],
            'rewards_per_battle': self.training_stats['avg_reward_per_battle']
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filename}")
        if self.training_stats['battles_completed'] > 0:
            print(f"  Completed: {self.training_stats['battles_completed']} battles")
            print(f"  Win rate: {results['win_rate']:.2%}")
    
    def save_model(self, filename="mlp_model.pth"):
        """Save the trained model."""
        if self.model is None:
            print("Warning: Model not initialized. Cannot save.")
            return
        
        # Ensure models directory exists
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)
            print(f"Created models directory: {models_dir}")
        
        # Save to models folder
        filepath = os.path.join(models_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_size': self.state_size,
            'action_size': self.action_size,
            'epsilon': self.epsilon,
            'training_stats': self.training_stats,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filename="mlp_model.pth"):
        """Load a trained model."""
        # Load from models folder
        filepath = os.path.join("models", filename)
        if not os.path.exists(filepath):
            print(f"Model file not found: {filepath}")
            return False
        
        checkpoint = torch.load(filepath)
        self.state_size = checkpoint['state_size']
        self.action_size = checkpoint['action_size']
        self._init_model(self.state_size, silent=True)  # Silent when loading existing model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']
        print(f"Model loaded from {filepath}")
        return True
