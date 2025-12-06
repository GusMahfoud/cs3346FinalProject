from poke_env.player.player import Player

import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

# Correct engine modules
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.move import Move
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.weather import Weather
from poke_env.battle.field import Field
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

WEATHER_KEYS = [
    Weather.SUNNYDAY,
    Weather.RAINDANCE,
    Weather.SANDSTORM,
    Weather.HAIL,
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

            # Base power normalized
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
    Only Stealth Rock and Spikes are relevant for our 20-mon pool.
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
    """
    vec = []

    # Weather one-hot + normalized duration
    weather_one_hot = [0.0] * len(WEATHER_KEYS)
    weather_duration = 0.0
    if battle.weather:
        if isinstance(battle.weather, tuple) and len(battle.weather) == 2:
            w, turns = battle.weather
        else:
            w = battle.weather
            turns = 0
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

    # Opponent's 6 Pokémon (public info only)
    for poke in battle.opponent_team.values():
        vec.extend(encode_pokemon(poke))

    # Our side conditions
    vec.extend(encode_side_conditions(battle.side_conditions))

    # Opponent side conditions
    vec.extend(encode_side_conditions(battle.opponent_side_conditions))

    # Global weather + terrain
    vec.extend(encode_weather_and_terrain(battle))

    # Trick Room – generic future-proofing
    tr = battle.fields.get("trickroom", 0)
    vec.append(1.0 if tr > 0 else 0.0)
    vec.append(min(tr, 8) / 8.0)

    # Turn number normalized
    vec.append(battle.turn / 100.0)

    return np.array(vec, dtype=np.float32)

# -----------------------------
# ACTOR-CRITIC NETWORK
# -----------------------------

class ActorCriticNet(nn.Module):
    """
    Shared trunk with:
      - Policy head: logits over actions
      - Value head: scalar V(s)
    """
    def __init__(self, state_size, action_size=10, hidden_size=512):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        """
        state: (B, state_size)
        returns:
          logits: (B, action_size)
          values: (B,)  value estimates
        """
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        logits = self.policy_head(x)
        values = self.value_head(x).squeeze(-1)
        return logits, values

# -----------------------------
# RL AGENT (A2C)
# -----------------------------

class MyRLAgent(Player):
    """
    A2C-based agent:
    - 10 discrete actions (4 moves + 6 switch options)
    - Rich game state encoding tailored to the 20-Pokémon mini-meta
    - Actor-Critic (shared backbone) with entropy bonus + gradient clipping
    - Uses RewardCalculator for dense shaping.
    """

    def __init__(
        self,
        battle_format="gen9ubers",
        learning_rate=0.0005,
        batch_size=128,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.10,
        epsilon_decay=0.995,
        model_path=None,
        **kwargs
    ):
        super().__init__(battle_format=battle_format, **kwargs)

        self.action_size = 10
        self.reward_calc = RewardCalculator()
        self.battle_history = []      # [(state, action, reward), ...]
        self.episode_buffer = []      # list of episodes
        self.experience_buffer = []   # flattened (state, action, return)
        self.prev_battle_state = None
        self.current_reward_score = 0.0
        self.last_action_was_switch = False

        # Model setup
        self.state_size = None
        self.model: ActorCriticNet | None = None
        self.optimizer = None
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate

        # A2C loss weights
        self.value_coef = 0.5
        self.entropy_coef = 0.01

        # Epsilon-greedy exploration
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Training stats
        self.training_stats = {
            "battles_completed": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "total_reward": 0.0,
            "avg_reward_per_battle": [],
            "win_rate_history": [],
        }

        # Try to load existing model
        if model_path:
            full_path = os.path.join("models", os.path.basename(model_path))
            if os.path.exists(full_path):
                print(f"Model found: {full_path}")
                self.load_model(os.path.basename(model_path))
            else:
                print(f"Model not found at {full_path}, creating new model")
        else:
            default_path = os.path.join("models", "mlp_model.pth")
            if os.path.exists(default_path):
                print(f"Model found: {default_path}")
                self.load_model("mlp_model.pth")
            else:
                print(f"Model not found at {default_path}, creating new model")

    # -------------------------
    # MODEL INIT / POLICY
    # -------------------------

    def _init_model(self, state_size, silent=False):
        """Initialize Actor-Critic model on first use."""
        if self.model is None:
            self.state_size = state_size
            self.model = ActorCriticNet(state_size, self.action_size, hidden_size=512)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.model.train()
            if not silent:
                print(f"Created new ActorCritic model (state_size={state_size}, hidden_size=512)")

    def compute_policy(self, state_vec: np.ndarray) -> np.ndarray:
        """
        Compute policy using A2C network.
        Returns action probabilities.
        """
        # Initialize model with a fixed max size if needed
        if self.model is None:
            max_state_size = 2000
            self._init_model(max_state_size)

        current_size = len(state_vec)
        if current_size < self.state_size:
            padded = np.zeros(self.state_size, dtype=np.float32)
            padded[:current_size] = state_vec
            state_vec = padded
        elif current_size > self.state_size:
            state_vec = state_vec[:self.state_size]

        self.model.eval()
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(state_tensor)
            probs = torch.softmax(logits, dim=1)

        return probs.squeeze(0).numpy()

    def reset_team(self, team_string):
        self.set_team(team_string)

    # -------------------------
    # MAIN DECISION HOOK
    # -------------------------

    def choose_move(self, battle):
        """Main hook called by poke-env to choose an action."""
        # Reward from previous transition
        if self.prev_battle_state is not None and not battle.finished:
            turn_reward = self.reward_calc.compute_turn_reward(
                self.prev_battle_state,
                battle,
                action_was_switch=self.last_action_was_switch
            )
            self.current_reward_score += turn_reward

            if len(self.battle_history) > 0:
                prev_state, prev_action, _ = self.battle_history[-1]
                self.battle_history[-1] = (prev_state, prev_action, turn_reward)

        # Encode current state
        state_vec = encode_state(battle)
        legal = self.legal_actions(battle)

        # Ensure model initialized
        if self.model is None:
            max_state_size = 2000
            self._init_model(max_state_size)

        # Epsilon-greedy on top of policy
        if np.random.random() < self.epsilon:
            action_index = np.random.choice(legal) if legal else 0
            move = self.action_to_move(action_index, battle)
        else:
            policy = self.compute_policy(state_vec)

            masked_policy = policy.copy()
            for i in range(self.action_size):
                if i not in legal:
                    masked_policy[i] = 0.0

            if masked_policy.sum() > 0:
                masked_policy = masked_policy / masked_policy.sum()
                action_index = np.random.choice(self.action_size, p=masked_policy)
            else:
                action_index = np.random.choice(legal) if legal else 0

            if action_index not in legal:
                move = self.choose_random_legal_action(battle)
                legal_now = self.legal_actions(battle)
                action_index = legal_now[0] if legal_now else 0
            else:
                move = self.action_to_move(action_index, battle)

        # Store (state, action, reward placeholder)
        self.battle_history.append((state_vec.copy(), action_index, 0.0))
        self.last_action_was_switch = (action_index >= 4)
        self.prev_battle_state = self._copy_battle_state(battle)

        return move

    def _copy_battle_state(self, battle):
        """Create a minimal copy of battle state for reward calculation."""
        state = {
            'turn': battle.turn,
            'team': {
                k: {
                    'current_hp': v.current_hp,
                    'max_hp': v.max_hp,
                    'fainted': v.fainted,
                    'status': v.status
                }
                for k, v in battle.team.items()
            },
            'opponent_team': {
                k: {
                    'current_hp': v.current_hp if v.current_hp is not None else 0,
                    'max_hp': v.max_hp if v.max_hp else 1,
                    'fainted': v.fainted,
                    'status': v.status
                }
                for k, v in battle.opponent_team.items()
            }
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
        # 0–3: moves
        if 0 <= action <= 3:
            moves = battle.available_moves
            if action < len(moves) and moves[action] is not None:
                return self.create_order(moves[action])
            return self.choose_random_legal_action(battle)

        # 4–9: switches
        if 4 <= action <= 9:
            idx = action - 4
            switches = battle.available_switches
            if idx < len(switches) and switches[idx] is not None:
                return self.create_order(switches[idx])
            return self.choose_random_legal_action(battle)

        return self.choose_random_legal_action(battle)

    def choose_random_legal_action(self, battle):
        """Sample uniformly from currently legal actions."""
        legal = self.legal_actions(battle)
        if not legal:
            return self.choose_random_move(battle)
        a = np.random.choice(legal)
        move = self.action_to_move(a, battle)
        self.last_action_was_switch = (a >= 4)
        return move

    # -------------------------
    # EPISODE END / TRAINING ENTRY
    # -------------------------

    def on_battle_end(self, battle):
        """Called when battle ends — compute final reward, update buffers, train, decay epsilon."""
        # Final turn reward
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

        # Terminal reward
        terminal_reward = self.reward_calc.compute_terminal_reward(battle)
        self.current_reward_score += terminal_reward

        if len(self.battle_history) > 0:
            prev_state, prev_action, prev_r = self.battle_history[-1]
            self.battle_history[-1] = (prev_state, prev_action, prev_r + terminal_reward)

        # Store episode + compute returns
        if len(self.battle_history) > 0:
            self.episode_buffer.append(self.battle_history.copy())
            self._process_episodes()

        # Stats
        self.training_stats["battles_completed"] += 1

        if battle.won:
            self.training_stats["wins"] += 1
        elif battle.lost:
            self.training_stats["losses"] += 1
        else:
            self.training_stats["draws"] += 1

        self.training_stats["total_reward"] += self.current_reward_score
        self.training_stats["avg_reward_per_battle"].append(self.current_reward_score)

        win_rate = self.training_stats["wins"] / self.training_stats["battles_completed"]
        self.training_stats["win_rate_history"].append(win_rate)

        # Epsilon decay
        decay_factor = self.epsilon_decay ** (1 + self.training_stats["battles_completed"] / 5000)
        self.epsilon = max(self.epsilon_end, self.epsilon * decay_factor)

        # Train if batch ready
        if len(self.experience_buffer) >= self.batch_size:
            self._train_batch()

        # Logging
        if self.training_stats["battles_completed"] % 50 == 0:
            print(
                f"Battles: {self.training_stats['battles_completed']}, "
                f"Win Rate: {win_rate:.2%}, "
                f"Epsilon: {self.epsilon:.4f}, "
                f"Reward (last): {self.current_reward_score:.2f}"
            )

        # Reset per-battle values
        self.reward_calc.reset()
        self.prev_battle_state = None
        self.current_reward_score = 0.0
        self.last_action_was_switch = False
        self.battle_history = []

    # -------------------------
    # EPISODE PROCESSING
    # -------------------------

    def _process_episodes(self):
        """Compute discounted returns per episode and flatten into experience_buffer."""
        while len(self.episode_buffer) > 0:
            episode = self.episode_buffer.pop(0)

            returns = []
            G = 0.0
            for i in range(len(episode) - 1, -1, -1):
                _, _, reward = episode[i]
                G = reward + self.gamma * G
                returns.insert(0, G)

            for (state, action, _), return_val in zip(episode, returns):
                self.experience_buffer.append((state, action, return_val))

    # -------------------------
    # A2C TRAINING STEP
    # -------------------------

    def _train_batch(self):
        """
        Train Actor-Critic on a batch of experiences:
          - policy loss = -logπ(a|s) * advantage
          - value loss = MSE(V(s), return)
          - entropy bonus encourages exploration
          - gradient clipping for stability
        """
        if self.model is None or len(self.experience_buffer) < self.batch_size:
            return

        batch = self.experience_buffer[:self.batch_size]
        self.experience_buffer = self.experience_buffer[self.batch_size:]

        states = []
        actions = []
        returns = []

        for state, action, return_val in batch:
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

        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(actions)
        returns_tensor = torch.FloatTensor(returns)

        # Normalize returns
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

        self.model.train()
        self.optimizer.zero_grad()

        logits, values = self.model(states_tensor)
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-8)

        selected_log_probs = log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Advantage = G_t - V(s_t)
        advantages = returns_tensor - values.detach()

        # Loss terms
        policy_loss = -(selected_log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns_tensor)
        entropy = -(probs * log_probs).sum(dim=1).mean()

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        print(
            f"Training batch | "
            f"Loss: {loss.item():.4f} | "
            f"Policy: {policy_loss.item():.4f} | "
            f"Value: {value_loss.item():.4f} | "
            f"Entropy: {entropy.item():.4f} | "
            f"Epsilon: {self.epsilon:.3f}"
        )

        self.model.eval()

    # -------------------------
    # SAVE / LOAD
    # -------------------------

    def save_results(self, filename="training_results.json"):
        """Save training statistics to JSON file."""
        if self.training_stats["battles_completed"] == 0:
            print("Warning: No battles completed. Results file will contain zeros.")

        results = {
            "total_battles": self.training_stats["battles_completed"],
            "wins": self.training_stats["wins"],
            "losses": self.training_stats["losses"],
            "draws": self.training_stats["draws"],
            "win_rate": self.training_stats["wins"] / max(self.training_stats["battles_completed"], 1),
            "total_reward": self.training_stats["total_reward"],
            "avg_reward_per_battle": (
                np.mean(self.training_stats["avg_reward_per_battle"])
                if self.training_stats["avg_reward_per_battle"] else 0.0
            ),
            "win_rate_history": self.training_stats["win_rate_history"],
            "rewards_per_battle": self.training_stats["avg_reward_per_battle"],
        }

        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {filename}")
        if self.training_stats["battles_completed"] > 0:
            print(f"  Completed: {self.training_stats['battles_completed']} battles")
            print(f"  Win rate: {results['win_rate']:.2%}")

    def save_model(self, folder="models/a2c_default"):
        os.makedirs(folder, exist_ok=True)

        ckpt = {
            "state_size": self.state_size,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_stats": self.training_stats,
        }

        path = os.path.join(folder, "checkpoint.pth")
        torch.save(ckpt, path)
        print(f"[A2C] Model saved → {path}")


    def load_model(self, folder="models/a2c_default"):
        path = os.path.join(folder, "checkpoint.pth")
        if not os.path.exists(path):
            print(f"[A2C] No checkpoint found at {path}. Starting fresh.")
            return False

        ckpt = torch.load(path)

        self.state_size = ckpt["state_size"]
        self._init_model(self.state_size, silent=True)

        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])

        self.epsilon = ckpt.get("epsilon", self.epsilon)
        self.training_stats = ckpt.get("training_stats", self.training_stats)

        print(f"[A2C] Loaded model from {path}")
        return True
