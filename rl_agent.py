# ============================================================
# FIXED & CLEANED RL AGENT WITH AUTO STATE SIZE DETECTION
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

        self.gamma = gamma
        self.lr = lr
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.batch_size = batch_size

        # Epsilon config
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
        self.prev_battle_state = None
        self.battle_history = []      # (state, action, reward, skip_flag)
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
        """Called exactly once when first state arrives."""
        self.state_size = state_size
        self.model = ActorCriticNet(state_size, action_size=10, hidden_size=512)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        print(f"[RL] Model initialized with state_size={state_size}")

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
    # LEGAL ACTIONS
    # ============================================================

    def legal_actions(self, battle):
        legal = []

        # Forced switch
        if (
            battle.active_pokemon is None
            or battle.active_pokemon.fainted
            or battle.active_pokemon.current_hp == 0
        ):
            for i, p in enumerate(battle.available_switches):
                if p:
                    legal.append(SWITCH_OFFSET + i)
            return legal

        # switching off
        if not self.allow_switching:
            for i, m in enumerate(battle.available_moves):
                if m and i < 4:
                    legal.append(i)
            if not legal:
                for i, p in enumerate(battle.available_switches):
                    if p:
                        legal.append(SWITCH_OFFSET + i)
            return legal

        # RL not allowed to switch
        if self.allow_switching and not self.rl_switch_enabled:
            for i, m in enumerate(battle.available_moves):
                if m:
                    legal.append(i)
            return legal

        # full legal set
        for i, m in enumerate(battle.available_moves):
            if m:
                legal.append(i)
        for i, p in enumerate(battle.available_switches):
            if p:
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
                    return self.create_order(m)

        # SWITCH
        if action >= SWITCH_OFFSET:
            switches = battle.available_switches
            idx = action - SWITCH_OFFSET
            if idx < len(switches) and switches[idx]:
                p = switches[idx]
                self.last_used_move = None
                self.last_action_was_switch = True
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
    # CHOOSE MOVE
    # ============================================================

    def choose_move(self, battle):
        """Main logic — now fully dimension-safe."""

        # --------------------------------------------------------
        # 1. Flush last-turn rewards
        # --------------------------------------------------------
        current_turn = battle.turn
        if self.prev_battle_state is not None and self.prev_turn is not None:
            if current_turn != self.prev_turn:
                r = self.reward_calc.flush()
                if self.battle_history:
                    s, a, old, skip = self.battle_history[-1]
                    self.battle_history[-1] = (s, a, old + r, skip)

        self.prev_turn = current_turn

        # --------------------------------------------------------
        # 2. Forced switch detection
        # --------------------------------------------------------
        forced_switch = (
            battle.active_pokemon is None
            or battle.active_pokemon.fainted
            or battle.force_switch
            or battle.active_pokemon.current_hp == 0
        )


        # --------------------------------------------------------
        # 3. FORCED SWITCH → ALWAYS heuristics
        # --------------------------------------------------------
        if forced_switch:
            switches = battle.available_switches
            if not switches:
                return self.choose_random_legal_action(battle)

            try:
                best = self.switch_expert.best_switch_target(battle)
            except Exception:
                best = None

            if best is None:
                best = switches[0]

            # STATE VECTOR
            state_vec = encode_state(battle).astype(np.float32)

            if self.model is None:
                self._init_model(len(state_vec))

            # LOG FOR TRAINING
            action = SWITCH_OFFSET
            self.battle_history.append((state_vec, action, 0.0, True))

            self.last_action_was_switch = True
            self.prev_battle_state = battle

            return self.create_order(best)


        # --------------------------------------------------------
        # 4. Normal-turn reward update
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

        # Dynamically detect state_size on first call
        if self.model is None:
            print(f"[RL] Detected state_vec size = {len(state_vec)}")
            self._init_model(len(state_vec))


        legal = self.legal_actions(battle)
        if not legal:
            move = self.choose_random_legal_action(battle)
            self.battle_history.append((state_vec, 0, 0.0, False))
            self.prev_battle_state = battle
            return move


        # --------------------------------------------------------
        # 6. Expert voluntary switching
        # --------------------------------------------------------
        if (
            self.allow_switching
            and self.use_expert_switching
            and battle.available_switches
            and not forced_switch
        ):
            try:
                if self.switch_expert.should_switch_out(battle):
                    best = self.switch_expert.best_switch_target(battle)
                    idx = battle.available_switches.index(best)
                    action = SWITCH_OFFSET + idx
                    self.battle_history.append((state_vec, action, 0.0, True))
                    self.last_action_was_switch = True
                    self.prev_battle_state = battle
                    return self.create_order(best)
            except Exception:
                pass


        # --------------------------------------------------------
        # 7. RL policy (ε-greedy)
        # --------------------------------------------------------
        if np.random.random() < self.epsilon:
            action = np.random.choice(legal)
            move = self.action_to_move(action, battle)
        else:
            st = torch.FloatTensor(state_vec).unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                logits, _ = self.model(st)
                probs = torch.softmax(logits, dim=1).squeeze().numpy()

            # mask illegal
            mask = np.zeros_like(probs)
            mask[legal] = probs[legal]

            if mask.sum() == 0:
                action = np.random.choice(legal)
            else:
                mask /= mask.sum()
                temp = max(0.3, self.epsilon)
                dist = mask ** (1 / temp)
                dist /= dist.sum()
                action = np.random.choice(len(dist), p=dist)

            move = self.action_to_move(action, battle)

        # LOG ACTION
        self.battle_history.append((state_vec.copy(), action, 0.0, False))
        self.prev_battle_state = battle

        return move


    # ============================================================
    # BATTLE END
    # ============================================================

    def on_battle_end(self, battle):

        # Flush final-step reward
        if self.battle_history:
            r = self.reward_calc.flush()
            s, a, old, skip = self.battle_history[-1]
            self.battle_history[-1] = (s, a, old + r, skip)

        # Terminal reward
        term = self.reward_calc.compute_terminal_reward(battle)
        if self.battle_history:
            s, a, old, skip = self.battle_history[-1]
            self.battle_history[-1] = (s, a, old + term, skip)

        # Store episode
        if self.battle_history:
            self.episode_buffer.append(self.battle_history.copy())
            self._process_episodes()

        # Reset
        self.reward_calc.reset()
        self.battle_history = []
        self.prev_battle_state = None
        self.last_used_move = None
        self.last_action_was_switch = False

        # ε decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Training when enough data
        if len(self.experience_buffer) >= self.batch_size:
            self._train_batch()


    # ============================================================
    # PROCESS EPISODES
    # ============================================================

    def _process_episodes(self):
        while self.episode_buffer:
            ep = self.episode_buffer.pop(0)

            # Compute returns
            G = 0
            returns = []
            for (_, _, r, skip) in reversed(ep):
                G = r + self.gamma * G
                returns.insert(0, G)

            # Push non-expert examples
            for (s, a, r, skip_flag), R in zip(ep, returns):
                if not skip_flag:
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

        self.global_train_step += 1

        print(
            f"[TRAIN] loss={loss.item():.4f} "
            f"policy={policy_loss.item():.4f} "
            f"value={value_loss.item():.4f} "
            f"entropy={entropy.item():.4f} "
            f"eps={self.epsilon:.3f}"
        )