# ============================================================
# FIXED & CLEANED RL AGENT WITH AUTO STATE SIZE DETECTION
# + PER-PHASE LR/ENTROPY CONTROL
# + SAFE STATE SIZE MISMATCH HANDLING
# + RETURN CLIPPING
# ============================================================

from poke_env.player.player import Player
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter

from advanced_switcher import SwitchHeuristics
from rewards import RewardCalculator
from state_encoder import encode_state

MOVE_ACTIONS = [0, 1, 2, 3]
SWITCH_OFFSET = 4


# ============================================================
# NETWORK
# ============================================================

class ActorCriticNet(nn.Module):
    def __init__(self, state_size, action_size=10, hidden_size=512):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.policy = nn.Linear(hidden_size, action_size)
        self.value = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

        # Orthogonal initialization
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.policy.bias, 0)
        nn.init.constant_(self.value.bias, 0)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.policy(x)
        value = self.value(x).squeeze(-1)
        return logits, value


# ============================================================
# RL AGENT
# ============================================================

class MyRLAgent(Player):
    def __init__(
        self,
        battle_format="gen9ubers",
        gamma=0.995,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.999,
        entropy_coef=0.05,
        value_coef=0.5,
        batch_size=256,
        allow_switching=False,
        use_expert_switching=False,
        rl_switch_enabled=False,
        expert_imitation_bonus=3.0,   # kept for future use if you want imitation learning
        model_folder=None,
        **kwargs
    ):
        super().__init__(battle_format=battle_format, **kwargs)

        # Core hyperparams
        self.gamma = gamma
        self.lr = lr
        self.base_lr = lr  # for reference
        self.entropy_coef = entropy_coef
        self.base_entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.batch_size = batch_size

        # Epsilon config (per-phase values will override these)
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Switching logic
        self.allow_switching = allow_switching
        self.use_expert_switching = use_expert_switching
        self.rl_switch_enabled = rl_switch_enabled
        self.expert_imitation_bonus = expert_imitation_bonus
        self.switch_expert = SwitchHeuristics()

        # Reward system
        self.reward_calc = RewardCalculator()
        self.prev_turn = None

        # Buffers
        # battle_history entries are ALWAYS:
        #   (state_vec, action_index, cumulative_reward, skip_flag)
        self.prev_battle_state = None
        self.battle_history = []
        self.episode_buffer = []
        self.experience_buffer = []

        # State/caches
        self.last_used_move = None
        self.last_action_was_switch = False
        self.last_action_priority = False

        # NN setup
        self.model = None
        self.optimizer = None
        self.state_size = None   # detected dynamically

        # TensorBoard
        self.writer = SummaryWriter(log_dir="runs/rl_training")
        self.global_train_step = 0

        # Optional checkpoint folder
        self.model_folder = model_folder
        if model_folder:
            self._try_load_model()

    # ============================================================
    # MODEL INIT / SAVE / LOAD
    # ============================================================

    def _init_model(self, state_size):
        """Called exactly once when first state arrives, or after load."""
        self.state_size = state_size
        self.model = ActorCriticNet(state_size, action_size=10, hidden_size=512)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        print(f"[RL] Model initialized with state_size={state_size}, lr={self.lr}")

    def choose_team(self, battle):
        return "/team 213456"

    def teampreview(self, battle):
        return "/team 213456"

    def _try_load_model(self):
        if not self.model_folder:
            return

        path = os.path.join(self.model_folder, "checkpoint.pth")
        if not os.path.exists(path):
            print("[RL] No checkpoint found.")
            return

        ckpt = torch.load(path, map_location="cpu")
        saved_state_size = ckpt["state_size"]

        # ALWAYS rebuild network shape to match saved_state_size
        self._init_model(saved_state_size)

        try:
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.epsilon = ckpt.get("epsilon", self.epsilon)
            print(f"[RL] Loaded checkpoint (state_size={saved_state_size}).")
        except Exception as e:
            print("[RL] ERROR loading checkpoint — mismatched dimensions!")
            print("→ Delete the old checkpoint and retrain.")
            print(e)

    def save_model(self):
        if not self.model_folder or self.model is None:
            return
        os.makedirs(self.model_folder, exist_ok=True)
        ckpt = {
            "state_size": self.state_size,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }
        torch.save(ckpt, os.path.join(self.model_folder, "checkpoint.pth"))
        print("[RL] Model saved.")

    # ============================================================
    # PHASE-LEVEL HYPERPARAM CONTROL
    # ============================================================

    def set_lr(self, lr: float):
        """Update optimizer LR safely."""
        self.lr = lr
        if self.optimizer is not None:
            for g in self.optimizer.param_groups:
                g["lr"] = lr
        print(f"[RL] Learning rate set to {lr:g}")

    def set_entropy_coef(self, coef: float):
        self.entropy_coef = coef
        print(f"[RL] Entropy coefficient set to {coef:g}")

    # ============================================================
    # LEGAL ACTIONS
    # ============================================================

    def legal_actions(self, battle):
        legal = []

        # Forced switch (only switches allowed)
        if (
            battle.active_pokemon is None
            or battle.active_pokemon.fainted
            or battle.active_pokemon.current_hp == 0
        ):
            for i, p in enumerate(battle.available_switches):
                if p is not None:
                    legal.append(SWITCH_OFFSET + i)
            return legal

        # Switching fully off → moves only
        if not self.allow_switching:
            for i, m in enumerate(battle.available_moves):
                if m is not None and i < 4:
                    legal.append(i)
            # Fallback: allow forced switches if literally no moves
            if not legal:
                for i, p in enumerate(battle.available_switches):
                    if p is not None:
                        legal.append(SWITCH_OFFSET + i)
            return legal

        # Switching allowed, but RL not allowed to choose switch
        if self.allow_switching and not self.rl_switch_enabled:
            for i, m in enumerate(battle.available_moves):
                if m is not None and i < 4:
                    legal.append(i)
            return legal

        # Full action set: moves + switches
        for i, m in enumerate(battle.available_moves):
            if m is not None and i < 4:
                legal.append(i)
        for i, p in enumerate(battle.available_switches):
            if p is not None:
                legal.append(SWITCH_OFFSET + i)
        return legal

    # ============================================================
    # ACTION → MOVE
    # ============================================================

    def action_to_move(self, action, battle):
        # MOVE
        if 0 <= action <= 3:
            moves = battle.available_moves
            if action < len(moves) and moves[action] is not None:
                m = moves[action]
                self.last_used_move = m
                self.last_action_was_switch = False
                self.last_action_priority = (m.priority > 0)
                return self.create_order(m)

            # Fallback: first valid move
            for m in moves:
                if m is not None:
                    self.last_used_move = m
                    self.last_action_was_switch = False
                    self.last_action_priority = (m.priority > 0)
                    return self.create_order(m)

        # SWITCH
        if action >= SWITCH_OFFSET:
            switches = battle.available_switches
            idx = action - SWITCH_OFFSET
            if idx < len(switches) and switches[idx] is not None:
                p = switches[idx]
                self.last_used_move = None
                self.last_action_was_switch = True
                self.last_action_priority = False
                return self.create_order(p)

            # Fallback: first valid switch
            for p in switches:
                if p is not None:
                    self.last_used_move = None
                    self.last_action_was_switch = True
                    self.last_action_priority = False
                    return self.create_order(p)

        # If something went very wrong, just random legal
        return self.choose_random_legal_action(battle)

    # ============================================================
    # RANDOM ACTION
    # ============================================================

    def choose_random_legal_action(self, battle):
        legal = self.legal_actions(battle)
        if not legal:
            # Very defensive fallback: just random move_order()
            return self.choose_random_move(battle)
        a = np.random.choice(legal)
        return self.action_to_move(a, battle)

    # ============================================================
    # CHOOSE MOVE (MAIN)
    # ============================================================

    def choose_move(self, battle):
        """
        Main decision logic.

        Key invariants:
        - battle_history ALWAYS stores 4-tuples (s, a, cumulative_r, skip_flag)
        - RewardCalculator:
            * compute_turn_reward(prev_state, current_state, last_action_was_switch)
              is called on EVERY decision after the first.
            * flush() is only called when the SHOWDOWN TURN changes.
              The flushed reward is added to the LAST logged (s, a, r, skip).
        - Forced switches + expert switches are logged with skip_flag=True
          so they don't go into the RL experience buffer.
        """

        current_turn = battle.turn

        # --------------------------------------------------------
        # 1. Compute reward for transition (prev_battle_state -> battle)
        # --------------------------------------------------------
        if self.prev_battle_state is not None:
            self.reward_calc.compute_turn_reward(
                self.prev_battle_state,
                battle,
                action_was_switch=self.last_action_was_switch,
                last_used_move=self.last_used_move,
            )

        # If the SHOWDOWN turn number advanced, flush accumulated reward
        if self.prev_turn is not None and current_turn != self.prev_turn:
            if self.battle_history:
                r = self.reward_calc.flush()
                s, a, old, skip = self.battle_history[-1]
                self.battle_history[-1] = (s, a, old + r, skip)

        self.prev_turn = current_turn

        # --------------------------------------------------------
        # 2. Detect forced switching (KO, U-turn into empty, Roar, etc.)
        # --------------------------------------------------------
        forced_switch = (
            battle.active_pokemon is None
            or battle.active_pokemon.fainted
            or battle.force_switch
            or battle.active_pokemon.current_hp == 0
        )

        # --------------------------------------------------------
        # 3. Encode current state
        # --------------------------------------------------------
        raw_state_vec = encode_state(battle).astype(np.float32)

        # If model already exists, but encoder changed length, pad/truncate safely.
        if self.model is not None and self.state_size is not None:
            if len(raw_state_vec) != self.state_size:
                # Non-ideal, but allows you to keep old checkpoints after encoder tweaks.
                fixed = np.zeros(self.state_size, dtype=np.float32)
                n = min(self.state_size, len(raw_state_vec))
                fixed[:n] = raw_state_vec[:n]
                state_vec = fixed
            else:
                state_vec = raw_state_vec
        else:
            state_vec = raw_state_vec

        # Initialize model on first ever state
        if self.model is None:
            print(f"[RL] Detected state_vec size = {len(state_vec)}")
            self._init_model(len(state_vec))

        # --------------------------------------------------------
        # 4. Handle forced switches with heuristic ONLY
        # --------------------------------------------------------
        if forced_switch:
            switches = battle.available_switches
            if not switches:
                # Safety fallback
                move = self.choose_random_legal_action(battle)
                # Log as skip=True since not RL-driven
                self.battle_history.append((state_vec.copy(), 0, 0.0, True))
                self.prev_battle_state = battle
                self.last_action_was_switch = True
                self.last_action_priority = False
                return move

            # Ask heuristic to pick best target
            try:
                best = self.switch_expert.best_switch_target(battle)
            except Exception:
                best = None

            if best is None:
                # Fallback: first non-None switch target
                for mon in switches:
                    if mon is not None:
                        best = mon
                        break

            if best is None:
                # As a last resort, random legal action
                move = self.choose_random_legal_action(battle)
                self.battle_history.append((state_vec.copy(), 0, 0.0, True))
                self.prev_battle_state = battle
                self.last_action_was_switch = True
                self.last_action_priority = False
                return move

            # Determine action index for logging (purely for consistency)
            action = SWITCH_OFFSET
            try:
                idx = battle.available_switches.index(best)
                action = SWITCH_OFFSET + idx
            except ValueError:
                pass

            # LOG: skip_flag=True → do NOT use as RL training data
            self.battle_history.append((state_vec.copy(), action, 0.0, True))

            self.last_action_was_switch = True
            self.last_action_priority = False
            self.prev_battle_state = battle

            return self.create_order(best)

        # --------------------------------------------------------
        # 5. Normal turn: pick action (possibly with expert override)
        # --------------------------------------------------------

        legal = self.legal_actions(battle)
        if not legal:
            move = self.choose_random_legal_action(battle)
            self.battle_history.append((state_vec.copy(), 0, 0.0, False))
            self.prev_battle_state = battle
            self.last_action_was_switch = False
            return move

        # 5a. Voluntary expert switching (if enabled)
        if (
            self.allow_switching
            and self.use_expert_switching
            and battle.available_switches
        ):
            try:
                if self.switch_expert.should_switch_out(battle):
                    best = self.switch_expert.best_switch_target(battle)
                    if best is not None:
                        try:
                            idx = battle.available_switches.index(best)
                            action = SWITCH_OFFSET + idx
                        except ValueError:
                            action = SWITCH_OFFSET

                        # LOG: expert action → skip_flag=True
                        self.battle_history.append((state_vec.copy(), action, 0.0, True))

                        self.last_action_was_switch = True
                        self.last_action_priority = False
                        self.prev_battle_state = battle
                        return self.create_order(best)
            except Exception:
                # If heuristic fails, fall back to RL / random
                pass

        # 5b. RL decision (ε-greedy)
        if np.random.random() < self.epsilon:
            # Exploration
            action = np.random.choice(legal)
            move = self.action_to_move(action, battle)
        else:
            # Policy
            st = torch.FloatTensor(state_vec).unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                logits, _ = self.model(st)
                probs = torch.softmax(logits, dim=1).squeeze().numpy()

            # Mask illegal actions
            mask = np.zeros_like(probs)
            mask[legal] = probs[legal]

            if mask.sum() == 0:
                action = np.random.choice(legal)
            else:
                mask /= mask.sum()
                # Simple temperature schedule tied to epsilon
                temp = max(0.3, float(self.epsilon))
                dist = mask ** (1.0 / temp)
                dist_sum = dist.sum()
                if dist_sum <= 0:
                    action = np.random.choice(legal)
                else:
                    dist /= dist_sum
                    action = np.random.choice(len(dist), p=dist)

            move = self.action_to_move(action, battle)

        # LOG RL-chosen action: skip_flag=False → used for training
        self.battle_history.append((state_vec.copy(), action, 0.0, False))
        self.prev_battle_state = battle

        return move

    # ============================================================
    # BATTLE END
    # ============================================================

    def _battle_finished_callback(self, battle):
        self.on_battle_end(battle)

    def on_battle_end(self, battle):
        # Final transition reward (prev_battle_state -> terminal battle)
        if self.prev_battle_state is not None:
            self.reward_calc.compute_turn_reward(
                self.prev_battle_state,
                battle,
                action_was_switch=self.last_action_was_switch,
                last_used_move=self.last_used_move,
            )

        # Flush remaining turn-level reward into last action
        if self.battle_history:
            r = self.reward_calc.flush()
            s, a, old, skip = self.battle_history[-1]
            self.battle_history[-1] = (s, a, old + r, skip)

        # Terminal reward (win/loss, etc.)
        term = self.reward_calc.compute_terminal_reward(battle)
        if self.battle_history:
            s, a, old, skip = self.battle_history[-1]
            self.battle_history[-1] = (s, a, old + term, skip)

        # Store full episode & convert into experience
        if self.battle_history:
            self.episode_buffer.append(self.battle_history.copy())
            self._process_episodes()

        # Reset per-battle state
        self.reward_calc.reset()
        self.battle_history = []
        self.prev_battle_state = None
        self.last_used_move = None
        self.last_action_was_switch = False
        self.last_action_priority = False
        self.prev_turn = None

        # ε decay: once per battle (as intended)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Train if we have a full batch
        if len(self.experience_buffer) >= self.batch_size:
            self._train_batch()

    # ============================================================
    # PROCESS EPISODES
    # ============================================================

    def _process_episodes(self):
        while self.episode_buffer:
            ep = self.episode_buffer.pop(0)

            # Compute returns (backwards)
            G = 0.0
            returns = []
            for (_, _, r, skip) in reversed(ep):
                G = r + self.gamma * G
                returns.insert(0, G)

            # Push non-expert examples only (skip_flag=False)
            for (s, a, r, skip_flag), R in zip(ep, returns):
                if not skip_flag:
                    # Mild return clipping for stability
                    R_clipped = max(-10.0, min(10.0, R))
                    self.experience_buffer.append((s, a, R_clipped))

    # ============================================================
    # TRAINING
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

        selected_log = log_probs[range(len(at)), at]

        # Advantage
        adv = rt - values.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        policy_loss = -(selected_log * adv).mean()
        value_loss = F.mse_loss(values, rt)
        entropy = -(probs * log_probs).sum(dim=1).mean()

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Logging
        self.writer.add_scalar("Loss/Total", loss.item(), self.global_train_step)
        self.writer.add_scalar("Loss/Policy", policy_loss.item(), self.global_train_step)
        self.writer.add_scalar("Loss/Value", value_loss.item(), self.global_train_step)
        self.writer.add_scalar("Entropy", entropy.item(), self.global_train_step)
        self.writer.add_scalar("Epsilon", self.epsilon, self.global_train_step)
        self.writer.add_scalar("LR", self.lr, self.global_train_step)

        self.global_train_step += 1