from poke_env.player.player import Player

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import os

from rewards import RewardCalculator
from state_encoder import encode_state


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
    - Uses compact, meta-relevant state from state_encoder.encode_state
    - Actor-Critic (shared backbone) with entropy bonus + gradient clipping
    - Uses RewardCalculator for dense shaping.
    """

    def __init__(
        self,
        battle_format="gen9ubers",
        learning_rate=0.00015,        # lower LR → more stable A2C with bucketed features
        batch_size=256,               # bigger batch → more stable advantage estimates
        gamma=0.995,                  # longer horizon (switching matters)
        epsilon_start=1.0,            # start purely exploratory
        epsilon_end=0.05,             # 5% final exploration
        epsilon_decay=0.992,          # slower decay → avoids collapsing into bad local policy
        entropy_coef=0.02,            # stronger entropy → explores better early game
        value_coef=0.5,               # same as standard A2C
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

        # A2C loss weights (use passed arguments)
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef


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
            folder = os.path.join("models", os.path.splitext(os.path.basename(model_path))[0])
            if os.path.exists(folder):
                print(f"Model folder found: {folder}")
                self.load_model(folder)
            else:
                print(f"Model folder not found at {folder}, creating new model")
        else:
            default_folder = "models/a2c_default"
            ckpt_path = os.path.join(default_folder, "checkpoint.pth")
            if os.path.exists(ckpt_path):
                print(f"Checkpoint found: {ckpt_path}")
                self.load_model(default_folder)
            else:
                print(f"No checkpoint at {ckpt_path}, creating new model on first state encode")

    # -------------------------
    # MODEL INIT / POLICY
    # -------------------------

    def _init_model(self, state_size, silent: bool = False):
        """Initialize Actor-Critic model once we know state vector length."""
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
        # Ensure model initialized with correct state size
        if self.model is None:
            self._init_model(len(state_vec))

        # Safety: pad or truncate if mismatch (shouldn't happen with fixed encoder)
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

        # Encode current state with your new encoder
        state_vec = encode_state(battle)
        legal = self.legal_actions(battle)

        # Ensure model initialized
        if self.model is None:
            self._init_model(len(state_vec))

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
        # For RewardCalculator API compatibility (it ignores this snapshot now)
        self.prev_battle_state = {"turn": battle.turn}

        return move

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

            # Safety: pad / truncate to state_size
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

        ckpt = torch.load(path, map_location="cpu")

        self.state_size = ckpt["state_size"]
        self._init_model(self.state_size, silent=True)

        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])

        self.epsilon = ckpt.get("epsilon", self.epsilon)
        self.training_stats = ckpt.get("training_stats", self.training_stats)

        print(f"[A2C] Loaded model from {path}")
        return True
