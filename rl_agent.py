from poke_env.player.player import Player
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter

from rewards import RewardCalculator
from state_encoder import encode_state

# ============================================================
# ACTION SPACE CONSTANTS
# ============================================================

# Move slots 0–3
MOVE_ACTIONS = [0, 1, 2, 3]

# Switch slots start here (up to 6 teammates → 4–9)
SWITCH_OFFSET = 4

# Pivot moves that cause switching on hit
PIVOT_MOVES = {"uturn", "voltturn", "flipturn", "partingshot"}


# ============================================================
# ACTOR–CRITIC NETWORK
# ============================================================

class ActorCriticNet(nn.Module):
    def __init__(self, state_size, action_size=10, hidden_size=512):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.policy = nn.Linear(hidden_size, action_size)
        self.value = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.policy(x)
        value = self.value(x).squeeze(-1)
        return logits, value


# ============================================================
# RL AGENT WITH TRAINING PHASES + MODEL IO + TENSORBOARD
# ============================================================

class MyRLAgent(Player):

    def __init__(
        self,
        battle_format="gen9ubers",
        gamma=0.995,
        lr=1.5e-4,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.992,
        entropy_coef=0.05,
        value_coef=0.5,
        batch_size=256,
        allow_switching=False,            # phase 1 default
        model_folder=None,
        **kwargs
    ):
        super().__init__(battle_format=battle_format, **kwargs)

        # Action space
        self.action_size = 10  # 4 moves + 6 switches

        # RL hyperparameters
        self.gamma = gamma
        self.lr = lr
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.batch_size = batch_size

        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Phase control (Phase 1: moves only)
        self.allow_switching = allow_switching

        # Reward engine
        self.reward_calc = RewardCalculator()

        # Buffers
        self.prev_battle_state = None
        self.battle_history = []
        self.episode_buffer = []
        self.experience_buffer = []

        # Model info
        self.model = None
        self.optimizer = None
        self.state_size = None

        # For reward shaping
        self.last_action_was_switch = False
        self.last_action_priority = False

        # TensorBoard
        self.writer = SummaryWriter(log_dir="runs/rl_training")
        self.global_train_step = 0

        # Model folder
        self.model_folder = model_folder
        if self.model_folder:
            self._try_load_model()

    # ============================================================
    # MODEL INIT
    # ============================================================

    def _init_model(self, state_size):
        self.state_size = state_size
        self.model = ActorCriticNet(state_size, self.action_size, hidden_size=512)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        print(f"[RL] Model created (state_size={state_size})")

    # ============================================================
    # MODEL LOAD / SAVE
    # ============================================================

    def _try_load_model(self):
        ckpt_path = os.path.join(self.model_folder, "checkpoint.pth")
        if not os.path.exists(ckpt_path):
            print("[RL] No checkpoint found, training from scratch.")
            return

        print(f"[RL] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        self._init_model(ckpt["state_size"])
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)

        print("[RL] Model loaded successfully.")

    def save_model(self):
        if self.model is None or not self.model_folder:
            return

        os.makedirs(self.model_folder, exist_ok=True)
        ckpt = {
            "state_size": self.state_size,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }

        ckpt_path = os.path.join(self.model_folder, "checkpoint.pth")
        torch.save(ckpt, ckpt_path)
        print(f"[RL] Saved model → {ckpt_path}")

    # ============================================================
    # LEGAL ACTIONS (PHASE-DEPENDENT, RECURSION-SAFE)
    # ============================================================

    def legal_actions(self, battle):
        legal = []

        # ----------------------------------------------------------
        # If Pokémon is fainted or missing → forced switching only
        # ----------------------------------------------------------
        if (
            battle.active_pokemon is None
            or battle.active_pokemon.fainted
            or battle.active_pokemon.current_hp == 0
        ):
            for i, p in enumerate(battle.available_switches):
                if p is not None:
                    legal.append(SWITCH_OFFSET + i)
            return legal  # ONLY switches allowed

        # ----------------------------------------------------------
        # PHASE 1 — MOVES ONLY (no manual switching, no pivot moves)
        # ----------------------------------------------------------
        if not self.allow_switching:
            for i, m in enumerate(battle.available_moves):
                if m is None or i >= 4:
                    continue
                # Block pivot moves (U-turn, VoltTurn, FlipTurn, Parting Shot)
                if m.id in PIVOT_MOVES:
                    continue
                legal.append(i)

            # If no legal moves (e.g. all PP stalled, Encore corner cases),
            # allow forced switch as a fallback.
            if len(legal) == 0:
                for i, p in enumerate(battle.available_switches):
                    if p is not None:
                        legal.append(SWITCH_OFFSET + i)

            return legal

        # ----------------------------------------------------------
        # PHASE 2 — FULL ACTION SET (moves + switches, pivot allowed)
        # ----------------------------------------------------------
        # Moves 0–3
        for i, m in enumerate(battle.available_moves):
            if m is not None and i < 4:
                legal.append(i)

        # Switches 4–9
        for i, p in enumerate(battle.available_switches):
            if p is not None:
                legal.append(SWITCH_OFFSET + i)

        return legal

    # ============================================================
    # ACTION → MOVE (RECURSION-SAFE)
    # ============================================================

    def action_to_move(self, action, battle):
        # -----------------------------
        # MOVE ACTION (0–3)
        # -----------------------------
        if 0 <= action <= 3:
            moves = battle.available_moves

            if action < len(moves) and moves[action] is not None:
                m = moves[action]
                self.last_action_priority = (m.priority > 0)
                self.last_action_was_switch = False
                return self.create_order(m)

            # Fallback: try any other available move
            for i, m in enumerate(moves):
                if m is not None and i < 4:
                    self.last_action_priority = (m.priority > 0)
                    self.last_action_was_switch = False
                    return self.create_order(m)

            # If no moves are available at all, fallback to random legal choice
            return self.choose_random_legal_action(battle)

        # -----------------------------
        # SWITCH ACTION (4+)
        # -----------------------------
        if action >= SWITCH_OFFSET:
            idx = action - SWITCH_OFFSET
            switches = battle.available_switches

            if idx < len(switches) and switches[idx] is not None:
                self.last_action_priority = False
                self.last_action_was_switch = True
                return self.create_order(switches[idx])

            # Fallback: pick first valid switch
            for i, p in enumerate(switches):
                if p is not None:
                    self.last_action_priority = False
                    self.last_action_was_switch = True
                    return self.create_order(p)

            # If no switches are valid, fallback to random legal choice
            return self.choose_random_legal_action(battle)

        # -----------------------------
        # Completely invalid action index
        # -----------------------------
        return self.choose_random_legal_action(battle)

    # ============================================================
    # RANDOM LEGAL ACTION (uses safe mapping above)
    # ============================================================

    def choose_random_legal_action(self, battle):
        legal = self.legal_actions(battle)

        # Extremely defensive guard (should basically never be empty)
        if not legal:
            # Try to default to any available move or switch
            if battle.available_moves:
                for m in battle.available_moves:
                    if m is not None:
                        self.last_action_priority = (m.priority > 0)
                        self.last_action_was_switch = False
                        return self.create_order(m)
            if battle.available_switches:
                for p in battle.available_switches:
                    if p is not None:
                        self.last_action_priority = False
                        self.last_action_was_switch = True
                        return self.create_order(p)
            # If literally nothing, just return a pass (shouldn't happen)
            return self.choose_default_move(battle)

        a = np.random.choice(legal)
        return self.action_to_move(a, battle)

    # ============================================================
    # CHOOSE MOVE
    # ============================================================

    def choose_move(self, battle):

        # Reward from previous state
        if self.prev_battle_state is not None and not battle.finished:
            r = self.reward_calc.compute_turn_reward(
                self.prev_battle_state,
                battle,
                action_was_switch=self.last_action_was_switch,
            )
            if self.battle_history:
                s, a, _ = self.battle_history[-1]
                self.battle_history[-1] = (s, a, r)

        # Encode state
        state_vec = encode_state(battle).astype(np.float32)
        if self.model is None:
            self._init_model(len(state_vec))

        legal = self.legal_actions(battle)

        # Defensive guard
        if not legal:
            move = self.choose_random_legal_action(battle)
            self.prev_battle_state = battle
            self.battle_history.append((state_vec.copy(), 0, 0.0))
            return move

        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            action = np.random.choice(legal)
            move = self.action_to_move(action, battle)

        else:
            st = torch.FloatTensor(state_vec).unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                logits, _ = self.model(st)
                probs = torch.softmax(logits, dim=1).squeeze().numpy()

            mask = np.zeros_like(probs)
            mask[legal] = probs[legal]

            if mask.sum() == 0:
                action = np.random.choice(legal)
            else:
                mask /= mask.sum()
                temperature = max(0.3, self.epsilon)
                dist = mask ** (1 / temperature)
                dist /= dist.sum()
                action = np.random.choice(len(dist), p=dist)

            move = self.action_to_move(action, battle)

        # Mark switch usage based on action index (forced or manual)
        self.last_action_was_switch = (action >= SWITCH_OFFSET)

        # Store transition
        self.battle_history.append((state_vec.copy(), action, 0.0))
        self.prev_battle_state = battle

        return move

    # ============================================================
    # BATTLE END
    # ============================================================

    def on_battle_end(self, battle):

        # Final intermediate reward
        if self.prev_battle_state is not None and self.battle_history:
            r = self.reward_calc.compute_turn_reward(
                self.prev_battle_state,
                battle,
                action_was_switch=self.last_action_was_switch,
            )
            s, a, old = self.battle_history[-1]
            self.battle_history[-1] = (s, a, old + r)

        # Terminal reward
        terminal = self.reward_calc.terminal_reward(battle)
        if self.battle_history:
            s, a, old = self.battle_history[-1]
            self.battle_history[-1] = (s, a, old + terminal)

        # Store episode
        if self.battle_history:
            self.episode_buffer.append(self.battle_history.copy())
            self._process_episodes()

        # Reset
        self.reward_calc.reset()
        self.prev_battle_state = None
        self.battle_history = []
        self.last_action_was_switch = False
        self.last_action_priority = False

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Train if enough data
        if len(self.experience_buffer) >= self.batch_size:
            self._train_batch()

    # ============================================================
    # EPISODE PROCESSING
    # ============================================================

    def _process_episodes(self):
        while self.episode_buffer:
            ep = self.episode_buffer.pop(0)

            returns = []
            G = 0.0
            for _, _, r in reversed(ep):
                G = r + self.gamma * G
                returns.insert(0, G)

            for (s, a, _), R in zip(ep, returns):
                self.experience_buffer.append((s, a, R))

    # ============================================================
    # TRAINING STEP
    # ============================================================

    def _train_batch(self):
        batch = self.experience_buffer[:self.batch_size]
        self.experience_buffer = self.experience_buffer[self.batch_size:]

        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        returns = np.array([b[2] for b in batch], dtype=np.float32)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        st = torch.FloatTensor(states)
        at = torch.LongTensor(actions)
        rt = torch.FloatTensor(returns)

        logits, values = self.model(st)
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-8)

        selected_logprobs = log_probs[range(len(at)), at]

        # Advantage
        advantages = rt - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(selected_logprobs * advantages).mean()
        value_loss = F.mse_loss(values, rt)
        entropy = -(probs * log_probs).sum(dim=1).mean()

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # TensorBoard logging
        self.writer.add_scalar("Loss/Total", loss.item(), self.global_train_step)
        self.writer.add_scalar("Loss/Policy", policy_loss.item(), self.global_train_step)
        self.writer.add_scalar("Loss/Value", value_loss.item(), self.global_train_step)
        self.writer.add_scalar("Entropy", entropy.item(), self.global_train_step)
        self.writer.add_scalar("Epsilon", self.epsilon, self.global_train_step)

        self.global_train_step += 1

        print(
            f"[TRAIN] loss={loss.item():.4f} "
            f"policy={policy_loss.item():.4f} "
            f"value={value_loss.item():.4f} "
            f"entropy={entropy.item():.4f} "
            f"eps={self.epsilon:.3f} "
            f"switching={'ON' if self.allow_switching else 'OFF'}"
        )
