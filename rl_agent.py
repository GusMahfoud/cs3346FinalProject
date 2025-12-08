# ============================================================
# FIXED & CLEANED RL AGENT WITH CORRECT EXPERT OVERRIDES
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

# Move indices 0–3
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

        # Orthogonal init
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
        epsilon_decay=0.9993,
        entropy_coef=0.05,
        value_coef=0.5,
        batch_size=256,
        allow_switching=False,
        use_expert_switching=False,
        rl_switch_enabled=False,
        expert_imitation_bonus=3.0,
        model_folder=None,
        **kwargs
    ):
        super().__init__(battle_format=battle_format, **kwargs)

        # RL config
        self.gamma = gamma
        self.lr = lr
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.batch_size = batch_size

        # Epsilon
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Switching controllers
        self.allow_switching = allow_switching
        self.use_expert_switching = use_expert_switching
        self.rl_switch_enabled = rl_switch_enabled
        self.expert_imitation_bonus = expert_imitation_bonus
        self.switch_expert = SwitchHeuristics()

        # Reward engine
        self.reward_calc = RewardCalculator()
        self.prev_turn = None

        # Buffers
        self.prev_battle_state = None
        self.battle_history = []      # (state, action, reward, expert_skip)
        self.episode_buffer = []      # list of episodes
        self.experience_buffer = []   # flattened

        # State caches
        self.last_used_move = None
        self.last_action_was_switch = False
        self.last_action_priority = False

        # NN
        self.model = None
        self.optimizer = None
        self.state_size = None

        # TensorBoard
        self.writer = SummaryWriter(log_dir="runs/rl_training")
        self.global_train_step = 0

        # File I/O
        self.model_folder = model_folder
        if model_folder:
            self._try_load_model()

    # ============================================================
    # MODEL INIT / SAVE / LOAD
    # ============================================================

    def _init_model(self, state_size):
        self.state_size = state_size
        self.model = ActorCriticNet(state_size, action_size=10, hidden_size=512)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        print(f"[RL] Model created, state_size={state_size}")

    def choose_team(self, battle):
        return "/team 213456"

    def teampreview(self, battle):
        return "/team 213456"

    def _try_load_model(self):
        path = os.path.join(self.model_folder, "checkpoint.pth")
        if not os.path.exists(path):
            print("[RL] No checkpoint found.")
            return

        ckpt = torch.load(path, map_location="cpu")
        self._init_model(ckpt["state_size"])
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)
        print("[RL] Loaded checkpoint successfully.")

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
    # LEGAL ACTIONS
    # ============================================================

    def legal_actions(self, battle):
        legal = []

        # Forced switch only
        if (
            battle.active_pokemon is None
            or battle.active_pokemon.fainted
            or battle.active_pokemon.current_hp == 0
        ):
            for i, p in enumerate(battle.available_switches):
                if p is not None:
                    legal.append(SWITCH_OFFSET + i)
            return legal

        # Switching disabled
        if not self.allow_switching:
            for i, m in enumerate(battle.available_moves):
                if m and i < 4:
                    legal.append(i)

            if not legal:
                for i, p in enumerate(battle.available_switches):
                    if p:
                        legal.append(SWITCH_OFFSET + i)
            return legal

        # Switching enabled but RL cannot switch
        if self.allow_switching and not self.rl_switch_enabled:
            for i, m in enumerate(battle.available_moves):
                if m and i < 4:
                    legal.append(i)
            return legal

        # Full action set
        for i, m in enumerate(battle.available_moves):
            if m and i < 4:
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
            if action < len(moves) and moves[action]:
                m = moves[action]
                self.last_used_move = m
                self.last_action_was_switch = False
                self.last_action_priority = (m.priority > 0)
                return self.create_order(m)

            # fallback
            for m in moves:
                if m:
                    self.last_used_move = m
                    self.last_action_was_switch = False
                    self.last_action_priority = (m.priority > 0)
                    return self.create_order(m)

        # SWITCH
        if action >= SWITCH_OFFSET:
            switches = battle.available_switches

            # Expert override target selection
            idx = action - SWITCH_OFFSET
            if idx < len(switches) and switches[idx]:
                p = switches[idx]
                self.last_used_move = None
                self.last_action_was_switch = True
                self.last_action_priority = False
                return self.create_order(p)

            # fallback
            for p in switches:
                if p:
                    self.last_used_move = None
                    self.last_action_was_switch = True
                    return self.create_order(p)

        return self.choose_random_legal_action(battle)

    # ============================================================
    # RANDOM ACTION
    # ============================================================

    def choose_random_legal_action(self, battle):
        legal = self.legal_actions(battle)
        a = np.random.choice(legal)
        return self.action_to_move(a, battle)

    # ============================================================
    # CHOOSE MOVE (MAIN LOGIC)
    # ============================================================

    def choose_move(self, battle):
        """
        FIXED VERSION:
        - Forced switches ALWAYS use best heuristic switch.
        - Heuristic is NEVER asked whether to switch during forced switches.
        - RL is prevented from taking illegal switch actions depending on phase.
        - Expert override only applies when switching is allowed AND not forced.
        - Rewards are flushed exactly once per turn.
        """

        # --------------------------------------------------------
        # 1. PROCESS REWARD FOR LAST TURN
        # --------------------------------------------------------
        current_turn = battle.turn
        if self.prev_battle_state is not None and self.prev_turn is not None:
            if current_turn != self.prev_turn:
                r = self.reward_calc.flush()
                if self.battle_history:
                    s, a, old = self.battle_history[-1]
                    self.battle_history[-1] = (s, a, old + r)

        self.prev_turn = current_turn

        # --------------------------------------------------------
        # 2. Detect forced switching (KO, Roar, Encore-locked, etc.)
        # --------------------------------------------------------
        forced_switch = (
            battle.active_pokemon is None
            or battle.active_pokemon.fainted
            or battle.force_switch
            or battle.active_pokemon.current_hp == 0
        )

        # --------------------------------------------------------
        # 3. Forced switch → ALWAYS let heuristic pick best switch
        # --------------------------------------------------------
        if forced_switch:
            switches = battle.available_switches

            if not switches:
                # extreme emergency: fallback
                return self.choose_random_legal_action(battle)

            try:
                best = self.switch_expert.best_switch_target(battle)
            except Exception:
                best = None

            if best is None:
                # fallback to first valid switch
                for mon in switches:
                    if mon is not None:
                        best = mon
                        break

            # Record reward transition
            state_vec = encode_state(battle).astype(np.float32)
            action = SWITCH_OFFSET  # heuristic-controlled forced switch
            self.battle_history.append((state_vec, action, 0.0))

            self.last_action_was_switch = True
            self.last_action_priority = False
            self.prev_battle_state = battle

            return self.create_order(best)

        # --------------------------------------------------------
        # 4. Normal-turn reward tracking
        # --------------------------------------------------------
        self.reward_calc.compute_turn_reward(
            self.prev_battle_state,
            battle,
            action_was_switch=self.last_action_was_switch
        )

        # --------------------------------------------------------
        # 5. Encode state
        # --------------------------------------------------------
        state_vec = encode_state(battle).astype(np.float32)
        if self.model is None:
            self._init_model(len(state_vec))

        legal = self.legal_actions(battle)
        if not legal:
            # recursion-safe fallback
            move = self.choose_random_legal_action(battle)
            self.battle_history.append((state_vec, 0, 0.0))
            self.prev_battle_state = battle
            return move

        # --------------------------------------------------------
        # 6. EXPERT SWITCH OVERRIDE (ONLY ON NON-FORCED TURNS)
        # --------------------------------------------------------
        if (
            self.allow_switching
            and self.use_expert_switching
            and battle.available_switches
            and not forced_switch
        ):
            try:
                if self.switch_expert.should_switch_out(battle):
                    # voluntary heuristic switch
                    best = self.switch_expert.best_switch_target(battle)

                    # determine index for logging
                    try:
                        idx = battle.available_switches.index(best)
                        action = SWITCH_OFFSET + idx
                    except Exception:
                        action = SWITCH_OFFSET

                    self.last_action_was_switch = True
                    self.last_action_priority = False

                    self.battle_history.append((state_vec, action, 0.0))
                    self.prev_battle_state = battle
                    return self.create_order(best)

            except Exception:
                pass

        # --------------------------------------------------------
        # 7. RL ACTION SELECTION (EPSILON GREEDY)
        # --------------------------------------------------------
        if np.random.random() < self.epsilon:
            # exploration
            action = np.random.choice(legal)
            move = self.action_to_move(action, battle)
        else:
            # policy
            st = torch.FloatTensor(state_vec).unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                logits, _ = self.model(st)
                probs = torch.softmax(logits, dim=1).squeeze().numpy()

            # mask illegal actions
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

        # --------------------------------------------------------
        # 8. Record RL-selected action
        # --------------------------------------------------------
        self.battle_history.append((state_vec.copy(), action, 0.0))
        self.prev_battle_state = battle

        return move
    # ============================================================
    # BATTLE END
    # ============================================================

    def on_battle_end(self, battle):
        # flush last turn reward
        if self.battle_history:
            r = self.reward_calc.flush()
            s, a, old, skip = self.battle_history[-1]
            self.battle_history[-1] = (s, a, old + r, skip)

        # terminal reward
        terminal = self.reward_calc.compute_terminal_reward(battle)
        if self.battle_history:
            s, a, old, skip = self.battle_history[-1]
            self.battle_history[-1] = (s, a, old + terminal, skip)

        # store episode
        if self.battle_history:
            self.episode_buffer.append(self.battle_history.copy())
            self._process_episodes()

        # reset tracking
        self.reward_calc.reset()
        self.battle_history = []
        self.prev_battle_state = None
        self.last_used_move = None
        self.last_action_was_switch = False

        # epsilon decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # train
        if len(self.experience_buffer) >= self.batch_size:
            self._train_batch()

    # ============================================================
    # PROCESS EPISODES → ADVANTAGE LEARNING BUFFER
    # ============================================================

    def _process_episodes(self):
        while self.episode_buffer:
            ep = self.episode_buffer.pop(0)

            # compute returns
            G = 0
            returns = []
            for (_, _, r, skip) in reversed(ep):
                G = r + self.gamma * G
                returns.insert(0, G)

            # store into training buffer (skip expert actions)
            for (s, a, r, skip_flag), R in zip(ep, returns):
                if skip_flag:
                    continue
                self.experience_buffer.append((s, a, R))

    # ============================================================
    # TRAINING
    # ============================================================

    def _train_batch(self):
        batch = self.experience_buffer[:self.batch_size]
        self.experience_buffer = self.experience_buffer[self.batch_size:]

        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        returns = np.array([b[2] for b in batch], dtype=np.float32)

        # normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        st = torch.FloatTensor(states)
        at = torch.LongTensor(actions)
        rt = torch.FloatTensor(returns)

        logits, values = self.model(st)
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-8)

        selected_logprobs = log_probs[range(len(at)), at]

        advantages = rt - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(selected_logprobs * advantages).mean()
        value_loss = F.mse_loss(values, rt)
        entropy = -(probs * log_probs).sum(dim=1).mean()

        loss = (
            policy_loss
            + self.value_coef * value_loss
            - self.entropy_coef * entropy
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Log
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
            f"eps={self.epsilon:.3f}"
        )