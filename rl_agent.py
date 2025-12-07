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

# ============================================================
# ACTION SPACE CONSTANTS
# ============================================================

# Move slots 0–3
MOVE_ACTIONS = [0, 1, 2, 3]

# Switch slots start here (up to 6 teammates → 4–9)
SWITCH_OFFSET = 4

# Pivot moves that cause switching on hit
PIVOT_MOVES = {"uturn", "voltswitch", "flipturn", "partingshot"}


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

        # Small init to avoid huge logits at start
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.constant_(self.policy.bias, 0.0)
        nn.init.constant_(self.value.bias, 0.0)

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
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.03,
        epsilon_decay=0.9995,
        entropy_coef=0.05,
        value_coef=0.5,
        batch_size=256,
        allow_switching=False,       # phase 1: moves only
        use_expert_switching=False, 
        rl_switch_enabled=False, # when True, use SwitchHeuristics
        expert_imitation_bonus=3.0,
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

        # Switching / expert control
        self.allow_switching = allow_switching
        self.use_expert_switching = use_expert_switching
        self.expert_imitation_bonus = expert_imitation_bonus
        self.rl_switch_enabled = rl_switch_enabled
        self.switch_expert = SwitchHeuristics()

        # Reward engine
        self.reward_calc = RewardCalculator()
        self.prev_turn = None


        # Buffers
        self.prev_battle_state = None
        self.battle_history = []      # list of (state_vec, action, reward)
        self.episode_buffer = []      # list of episodes
        self.experience_buffer = []   # flattened (s, a, R)

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

    def teampreview(self, battle):
        return "/team 213456"
    # ============================================================
    # MODEL LOAD / SAVE
    # ============================================================
    def choose_team(self, battle):
        # Always pick the given team order: Dragapult lead
        return "/team 213456"
    
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
        # 1. FORCED SWITCH (KO, Roar, Encore lock, etc.)
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
        # 2. SWITCHING DISABLED (Phase 1)
        #    RL may use pivot moves, but NOT manual switching
        # ----------------------------------------------------------
        if not self.allow_switching:
            for i, m in enumerate(battle.available_moves):
                if m is None or i >= 4:
                    continue
                # IMPORTANT: pivot moves are allowed
                legal.append(i)

            # If no attacking moves: fallback to forced switching
            if len(legal) == 0:
                for i, p in enumerate(battle.available_switches):
                    if p is not None:
                        legal.append(SWITCH_OFFSET + i)

            return legal

        # ----------------------------------------------------------
        # 3. SWITCHING ENABLED,
        #    BUT RL IS NOT ALLOWED TO CHOOSE SWITCHES (Phase 2A)
        #    Expert switching still works separately
        # ----------------------------------------------------------
        if self.allow_switching and not self.rl_switch_enabled:
            # RL can ONLY choose moves (including pivots)
            for i, m in enumerate(battle.available_moves):
                if m is not None and i < 4:
                    legal.append(i)

            # DO NOT expose switch actions to RL
            return legal

        # ----------------------------------------------------------
        # 4. FULL ACTION SET (Phase 2B / 3)
        #    RL AND expert both may switch
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
    # ACTION → MOVE (RECURSION-SAFE, WITH EXPERT OVERRIDE)
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
            switches = battle.available_switches

            # If expert switching is enabled, always switch into best heuristic target
            if self.allow_switching and self.use_expert_switching and switches:
                try:
                    best_mon = self.switch_expert.best_switch_target(battle)
                except Exception:
                    best_mon = None

                if best_mon is not None:
                    self.last_action_priority = False
                    self.last_action_was_switch = True
                    return self.create_order(best_mon)

            # Otherwise, respect the index when possible
            idx = action - SWITCH_OFFSET
            if idx < len(switches) and switches[idx] is not None:
                self.last_action_priority = False
                self.last_action_was_switch = True
                return self.create_order(switches[idx])

            # Fallback: pick first valid switch
            for p in switches:
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
        # --------------------------------------------------------
        # 1. PROCESS REWARD FOR LAST TURN
        # --------------------------------------------------------
        current_turn = battle.turn

        # If we have previous state AND turn changed, flush accumulated reward
        if self.prev_battle_state is not None and self.prev_turn is not None:
            if current_turn != self.prev_turn:
                # flush reward for previous turn
                r = self.reward_calc.flush()

                # assign reward to last (s,a)
                if self.battle_history:
                    s, a, old = self.battle_history[-1]
                    self.battle_history[-1] = (s, a, old + r)

        # Always update prev_turn
        self.prev_turn = current_turn

        # Continue gathering reward data for new events in this turn
        self.reward_calc.compute_turn_reward(
            self.prev_battle_state,
            battle,
            action_was_switch=self.last_action_was_switch,
)


        # --------------------------------------------------------
        # 2. Encode current state
        # --------------------------------------------------------
        state_vec = encode_state(battle).astype(np.float32)
        if self.model is None:
            self._init_model(len(state_vec))

        legal = self.legal_actions(battle)

        # Defensive guard
        if not legal:
            move = self.choose_random_legal_action(battle)
            self.prev_battle_state = battle
            # store dummy action 0, reward 0
            self.battle_history.append((state_vec.copy(), 0, 0.0))
            return move

        # --------------------------------------------------------
        # 3. Expert override BEFORE RL sampling (optional)
        #    If expert strongly says "switch out now", we obey.
        # --------------------------------------------------------
        if (
            self.allow_switching
            and self.use_expert_switching
            and battle.available_switches
        ):
            try:
                if self.switch_expert.should_switch_out(battle):
                    best_mon = self.switch_expert.best_switch_target(battle)
                else:
                    best_mon = None
            except Exception:
                best_mon = None

            if best_mon is not None:
                # Determine a consistent action index to record
                try:
                    idx = battle.available_switches.index(best_mon)
                    action = SWITCH_OFFSET + idx
                except ValueError:
                    # If somehow not in available_switches list, just pick first
                    action = SWITCH_OFFSET

                self.last_action_was_switch = True
                self.last_action_priority = False

                move = self.create_order(best_mon)
                self.battle_history.append((state_vec.copy(), action, 0.0))
                self.prev_battle_state = battle
                return move

        # --------------------------------------------------------
        # 4. RL policy: epsilon-greedy over legal actions
        # --------------------------------------------------------
        if np.random.random() < self.epsilon:
            # Exploration
            action = np.random.choice(legal)
            move = self.action_to_move(action, battle)
        else:
            # Policy sampling
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
                # Small temperature to avoid over-confident early picks
                temperature = max(0.3, self.epsilon)
                dist = mask ** (1 / temperature)
                dist /= dist.sum()
                action = np.random.choice(len(dist), p=dist)

            move = self.action_to_move(action, battle)

        # NOTE: last_action_was_switch is set inside action_to_move.
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
            # Flush any leftover turn reward
            r = self.reward_calc.flush()
            if self.battle_history:
                s, a, old = self.battle_history[-1]
                self.battle_history[-1] = (s, a, old + r)


            if self.use_expert_switching:
                try:
                    if self.last_action_was_switch:
                        if self.switch_expert.should_switch_out(self.prev_battle_state):
                            r += self.expert_imitation_bonus
                except Exception:
                    pass

            s, a, old = self.battle_history[-1]
            self.battle_history[-1] = (s, a, old + r)

        # Terminal reward
        terminal = self.reward_calc.compute_terminal_reward(battle)
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
            f"switching={'ON' if self.allow_switching else 'OFF'} "
            f"expert={'ON' if self.use_expert_switching else 'OFF'}"
        )
