# ============================================================
# rl_agent.py — Move-Only A2C Agent
# Heuristic controls ALL switching (voluntary + involuntary)
# Supports: warmup (no switching), phase1, phase2, phase3
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
from state_encoder import encode_state, estimate_damage


# ============================================================
# ACTION SPACE — 4 MOVES ONLY
# ============================================================

MOVE_ACTIONS = [0, 1, 2, 3]
ACTION_SIZE = 4      # NO SWITCH ACTION
# MUST MATCH EXACTLY move.id from PokeEnv = lowercase, no hyphens or spaces
PRIORITY_MOVES = {
    "iceshard", "shadowsneak", "extremespeed",
    "machpunch", "suckerpunch"
}

BOOSTING_MOVES = {
    "nastyplot", "swordsdance", "calmmind",
    "quiverdance", "bulkup", "torchsong"
}

PIVOT_MOVES = {"uturn", "voltswitch", "flipturn", "partingshot", "batonpass"}

STAT_DROP_MOVES = {
    "makeitrain", "dracometeor", "overheat",
    "leafstorm", "psychoboost"
}
def norm_move_id(move):
    if move is None:
        return ""
    return move.id.lower().replace(" ", "").replace("-", "")
# ============================================================
# NETWORK
# ============================================================

class ActorCriticNet(nn.Module):
    def __init__(self, state_size, hidden_size=512):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.policy = nn.Linear(hidden_size, ACTION_SIZE)
        self.value = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

        # Initialize weights orthogonally
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
        lr=5e-4,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.999,
        entropy_coef=0.01,
        value_coef=0.5,
        batch_size=256,
        allow_switching=False,     # set by training phase
        use_expert_switching=False,
        model_folder=None,
        **kwargs
    ):
        super().__init__(battle_format=battle_format, **kwargs)

        # Hyperparameters
        self.gamma = gamma
        self.lr = lr
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.batch_size = batch_size

        # ε parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Switching control
        self.allow_switching = allow_switching
        self.use_expert_switching = use_expert_switching
        self.switcher = SwitchHeuristics()

        # Reward calculator
        self.reward_calc = RewardCalculator()

        # Internal tracking
        self.prev_turn = None
        self.prev_battle_state = None
        self.last_used_move = None
        self.last_action_priority = False
        self.last_action_was_switch = False

        # Buffers
        self.battle_history = []
        self.episode_buffer = []
        self.experience_buffer = []

        # Model
        self.model = None
        self.optimizer = None
        self.state_size = None

        # TensorBoard
        self.writer = SummaryWriter(log_dir="runs/rl_training")
        self.global_train_step = 0

        # Model load
        self.model_folder = model_folder
        if model_folder:
            self._try_load_model()

    # ============================================================
    # Model load/save/init
    # ============================================================


    def override_with_best_damage(self, battle, chosen_move):
        """Override RL move with best-damage option unless exempt."""

        if chosen_move is None:
            return None

        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        chosen_id = norm_move_id(chosen_move)

        # ---------------------------------------------------------
        # 1. Moves never overridden
        # ---------------------------------------------------------
        if (
            chosen_id in PRIORITY_MOVES
            or chosen_id in BOOSTING_MOVES
            or chosen_id in PIVOT_MOVES
        ):
            (f"[OVERRIDE] Skip override: {chosen_id} is protected.")
            return chosen_move

        # ---------------------------------------------------------
        # 2. Evaluate chosen move
        # ---------------------------------------------------------
        chosen_dmg, _, _, _ = estimate_damage(chosen_move, me, opp)

        best_move = chosen_move
        best_dmg = chosen_dmg

        # ---------------------------------------------------------
        # 3. Search for best damage move
        # ---------------------------------------------------------
        for mv in battle.available_moves:
            if mv is None:
                continue

            mv_id = norm_move_id(mv)

            # skip non-attacks
            if mv.category.name == "STATUS":
                continue

            dmg, _, _, _ = estimate_damage(mv, me, opp)

            if dmg > best_dmg:
                best_dmg = dmg
                best_move = mv
                (f"[OVERRIDE] Better move found: {mv_id} ({dmg:.3f}) vs {chosen_id} ({chosen_dmg:.3f})")

        # no change
        if best_move == chosen_move:
            return chosen_move

        best_id = norm_move_id(best_move)

        # ---------------------------------------------------------
        # 4. Stat-drop moves require threshold advantage
        # ---------------------------------------------------------
        THRESHOLD = 0.25

        if best_id in STAT_DROP_MOVES:
            if not (best_dmg >= chosen_dmg * (1 + THRESHOLD)):
                (f"[OVERRIDE] Reject stat-drop {best_id}: improvement {best_dmg:.3f} insufficient.")
                return chosen_move

        # ---------------------------------------------------------
        # 5. Return overridden move
        # ---------------------------------------------------------
        
        return best_move

    def _init_model(self, state_size):
        self.state_size = state_size
        self.model = ActorCriticNet(state_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        print(f"[RL] Model initialized with state_size={state_size}, lr={self.lr}")

    #def choose_team(self, battle):
        # keep your team selection logic
    #    return "/team 213456"

   #def teampreview(self, battle):
        # keep your team preview logic
    #    return "/team 213456"
    def _try_load_model(self):
        path = os.path.join(self.model_folder, "checkpoint.pth")
        if not os.path.exists(path):
            print("[RL] No checkpoint found.")
            return

        ckpt = torch.load(path, map_location="cpu")
        saved_size = ckpt["state_size"]

        self._init_model(saved_size)

        try:
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.epsilon = ckpt.get("epsilon", self.epsilon)
            print("[RL] Loaded checkpoint successfully.")
        except Exception as e:
            print("[RL] ERROR loading checkpoint.")
            print(e)

    def save_model(self):
        if self.model is None:
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
    # Hyperparameter updates (per-phase)
    # ============================================================

    def set_lr(self, lr):
        self.lr = lr
        if self.optimizer:
            for g in self.optimizer.param_groups:
                g["lr"] = lr

    def set_entropy_coef(self, coef):
        self.entropy_coef = coef

    # ============================================================
    # MOVE-ONLY ACTION SPACE
    # ============================================================

    def legal_actions(self, battle):
        legal = []
        for i, m in enumerate(battle.available_moves):
            if m is not None and i < 4:
                legal.append(i)
        if not legal:
            return [0]
        return legal

    # ============================================================
    # MOVE EXECUTION
    # ============================================================

    def _make_move(self, action_idx, battle):
        moves = battle.available_moves

        # ------------------------------------------------------------
        # 1. If action index maps to a real move, pick it (RL choice)
        # ------------------------------------------------------------
        if action_idx < len(moves) and moves[action_idx] is not None:
            chosen = moves[action_idx]
        else:
            # fallback: just pick first available move
            chosen = None
            for m in moves:
                if m is not None:
                    chosen = m
                    break

            if chosen is None:
                return self.choose_random_move(battle)

        # ------------------------------------------------------------
        # 2. Apply override heuristic AFTER RL picks the move
        # ------------------------------------------------------------
        overridden = self.override_with_best_damage(battle, chosen)

        # ------------------------------------------------------------
        # 3. Store this as the move actually taken (critical for PPO)
        # ------------------------------------------------------------
        final_move = overridden

        # RL must learn that THIS was the selected action
        self.last_used_move = final_move
        self.last_action_priority = final_move.priority > 0
        self.last_action_was_switch = False

        return self.create_order(final_move)

    # ============================================================
    # SWITCH HANDLING — ALWAYS HEURISTIC CONTROL
    # ============================================================

    def _do_switch(self, battle, voluntary: bool):
        try:
            target = self.switcher.best_switch_target(battle, voluntary=voluntary)
        except Exception:
            target = None

        self.last_used_move = None
        self.last_action_was_switch = True
        self.last_action_priority = False

        if target:
            return self.create_order(target)

        # fallback random
        return self.choose_random_move(battle)

    # ============================================================
    # MAIN DECISION FUNCTION
    # ============================================================

    def choose_move(self, battle):
        turn_now = battle.turn
        # reward delivery between prev_state → battle
        if self.prev_battle_state is not None:
            self.reward_calc.compute_turn_reward(
                self.prev_battle_state,
                battle,
                action_was_switch=self.last_action_was_switch,
                last_used_move=self.last_used_move,
            )

        # flush reward when turn increments
        if self.prev_turn is not None and turn_now != self.prev_turn:
            if self.battle_history:
                r = self.reward_calc.flush()
                s, a, old, skip = self.battle_history[-1]
                self.battle_history[-1] = (s, a, old + r, skip)

        self.prev_turn = turn_now

        # Encode state
        raw = encode_state(battle).astype(np.float32)
        if self.model is None:
            self._init_model(len(raw))
        elif len(raw) != self.state_size:
            fixed = np.zeros(self.state_size, dtype=np.float32)
            fixed[:min(self.state_size, len(raw))] = raw[:min(self.state_size, len(raw))]
            raw = fixed

        state_vec = raw

        # ========================================================
        # FORCED SWITCH (KO, phazing, etc.) → involuntary
        # ========================================================
        forced = (
            battle.active_pokemon is None
            or battle.active_pokemon.fainted
            or battle.force_switch
            or battle.active_pokemon.current_hp == 0
        )

        if forced:
            self.battle_history.append((state_vec.copy(), -1, 0.0, True))
            move = self._do_switch(battle, voluntary=False)
            self.prev_battle_state = battle
            return move

        # ========================================================
        # VOLUNTARY SWITCH (PHASE 1/2/3)
        # ========================================================
        if self.allow_switching and self.use_expert_switching:
            try:
                shouldSwitch=self.switcher.should_switch_out(battle)
                if shouldSwitch:
                    self.battle_history.append((state_vec.copy(), -1, 0.0, True))
                    move = self._do_switch(battle, voluntary=True)
                    self.prev_battle_state = battle
                    return move
            except Exception as e:
                import traceback
                print("[SWITCH ERROR]", e)
                traceback.print_exc()
                pass

        # ========================================================
        # RL MOVE DECISION
        # ========================================================
        legal = self.legal_actions(battle)

        if np.random.random() < self.epsilon:
            action = np.random.choice(legal)
        else:
            st = torch.FloatTensor(state_vec).unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                logits, _ = self.model(st)
                probs = torch.softmax(logits, dim=1).squeeze().numpy()

            mask = np.zeros_like(probs)
            mask[:4] = probs[:4]
            mask = mask[legal]

            if mask.sum() == 0:
                action = np.random.choice(legal)
            else:
                action = np.random.choice(
                    legal,
                    p=mask / mask.sum()
                )

        move = self._make_move(action, battle)
        # Determine final move index (so RL learns correctly)
        final_idx = battle.available_moves.index(self.last_used_move)

        self.battle_history.append((state_vec.copy(), final_idx, 0.0, False))
        self.prev_battle_state = battle
        return move

    # ============================================================
    # BATTLE END → Compute returns + training
    # ============================================================
    def _battle_finished_callback(self, battle):
        self.on_battle_end(battle)
    def on_battle_end(self, battle):
        # finalize transition
        if self.prev_battle_state is not None:
            self.reward_calc.compute_turn_reward(
                self.prev_battle_state,
                battle,
                action_was_switch=self.last_action_was_switch,
                last_used_move=self.last_used_move,
            )

        # apply turn flush
        if self.battle_history:
            r = self.reward_calc.flush()
            s, a, old, skip = self.battle_history[-1]
            self.battle_history[-1] = (s, a, old + r, skip)

        # terminal reward
        term = self.reward_calc.compute_terminal_reward(battle)
        if self.battle_history:
            s, a, old, skip = self.battle_history[-1]
            self.battle_history[-1] = (s, a, old + term, skip)

        # Store full episode
        if self.battle_history:
            self.episode_buffer.append(self.battle_history.copy())
            self._process_episodes()

        # Reset
        self.reward_calc.reset()
        self.battle_history = []
        self.prev_battle_state = None
        self.last_used_move = None
        self.last_action_was_switch = False
        self.last_action_priority = False
        self.prev_turn = None

        # epsilon decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # train if ready
        if len(self.experience_buffer) >= self.batch_size:
            self._train_batch()

    # ============================================================
    # EPISODE PROCESSING
    # ============================================================

    def _process_episodes(self):
        while self.episode_buffer:
            ep = self.episode_buffer.pop(0)

            G = 0.0
            returns = []

            for (_, _, r, _) in reversed(ep):
                G = r + self.gamma * G
                returns.insert(0, G)

            for (s, a, r, skip), R in zip(ep, returns):
                if not skip:  # ignore heuristic switches
                    R = max(-10, min(10, R))
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

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        st = torch.FloatTensor(states)
        at = torch.LongTensor(actions)
        rt = torch.FloatTensor(returns)

        logits, values = self.model(st)
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-8)

        selected = log_probs[range(len(at)), at]

        adv = rt - values.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        policy_loss = -(selected * adv).mean()
        value_loss = F.mse_loss(values, rt)
        entropy = -(probs * log_probs).sum(dim=1).mean()

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # logging
        self.writer.add_scalar("Loss/Total", loss.item(), self.global_train_step)
        self.writer.add_scalar("Loss/Policy", policy_loss.item(), self.global_train_step)
        self.writer.add_scalar("Loss/Value", value_loss.item(), self.global_train_step)
        self.writer.add_scalar("Entropy", entropy.item(), self.global_train_step)
        self.writer.add_scalar("Epsilon", self.epsilon, self.global_train_step)

        self.global_train_step += 1