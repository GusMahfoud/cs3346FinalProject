# ============================================================
# rewards.py — REBUILT REWARD SYSTEM (v2)
# Competitive-quality shaping for RL switching & decision-making
# ============================================================

import numpy as np
from poke_env.data import GenData

from computed_stats import get_real_stats
from advanced_switcher import SwitchHeuristics
from state_encoder import danger_score, kill_threat_score

GEN9_DATA = GenData.from_gen(9)
TYPE_CHART = GEN9_DATA.type_chart

PIVOT_MOVES = {"uturn", "voltswitch", "flipturn", "partingshot", "batonpass"}


# ------------------------------------------------------------
# HELPER: HP bucket (0–9)
# ------------------------------------------------------------
def hp_bucket(frac):
    frac = max(0.0, min(1.0, frac))
    return int(frac * 10)


# ------------------------------------------------------------
# HELPER: diminishing returns on boosts
# ------------------------------------------------------------
def stage_multiplier(stage):
    stage = max(-6, min(6, stage))
    mults = {
        -6: 2/8, -5: 2/7, -4: 2/6, -3: 2/5, -2: 2/4, -1: 2/3,
         0: 1.0,
        +1: 3/2, +2: 2.0, +3: 5/2, +4: 3.0, +5: 7/2, +6: 4.0
    }
    return mults[stage]


def diminishing_boost_value(stage, base=4):
    if stage <= 0:
        return 0
    mult = stage_multiplier(stage)
    return base * np.log2(mult)


BOOSTING_MOVES = {"swordsdance", "nastyplot", "quiverdance", "calmmind", "torchsong"}


# ============================================================
# REWARD CALCULATOR — COMPLETE REWRITE
# ============================================================
class RewardCalculator:

    def __init__(
        self,
        # Game outcome
        terminal_win=80,
        terminal_loss=-90,
        terminal_draw=0,

        # HP shaping (reduced)
        hp_reward=1.5,
        hp_penalty=1.0,

        # KOs
        ko_reward=20,
        faint_penalty=18,

        # Status
        status_reward=3.0,
        status_penalty=3.0,

        # Boosts
        boost_base=8,
        suicide_boost_penalty=10,

        # Switching
        baseline_switch_reward=3.0,
        switch_matchup_weight=4.0,
        switch_danger_weight=3.0,
        switch_threat_weight=2.0,
        wasted_boost_penalty=4.0,
        debuff_escape_reward=10.0,
        oscillation_penalty=10,
        switch_chain_penalty=5,
        switch_spam_penalty=10,
        switch_window=6,

        # Pivot moves
        pivot_reward=4.0,

        # Immunity / wasted move
        immune_move_penalty=8.0,

        # Turn pacing
        turn_penalty=0.02,

        # Danger / threat scaling
        danger_penalty=6.0,
        kill_threat_bonus=8.0,
        danger_reduction_bonus=5.0,
        high_danger_threshold=0.85,
        low_threat_threshold=0.35,
        high_danger_extra_penalty=12.0,
    ):

        # Terminal
        self.terminal_win = terminal_win
        self.terminal_loss = terminal_loss
        self.terminal_draw = terminal_draw

        # HP / status / KOs
        self.hp_reward = hp_reward
        self.hp_penalty = hp_penalty
        self.ko_reward = ko_reward
        self.faint_penalty = faint_penalty
        self.status_reward = status_reward
        self.status_penalty = status_penalty

        # Boosts
        self.boost_base = boost_base
        self.suicide_boost_penalty = suicide_boost_penalty

        # Switching
        self.baseline_switch_reward = baseline_switch_reward
        self.switch_matchup_weight = switch_matchup_weight
        self.switch_danger_weight = switch_danger_weight
        self.switch_threat_weight = switch_threat_weight
        self.wasted_boost_penalty = wasted_boost_penalty
        self.debuff_escape_reward = debuff_escape_reward
        self.oscillation_penalty = oscillation_penalty
        self.switch_chain_penalty = switch_chain_penalty
        self.switch_spam_penalty = switch_spam_penalty
        self.switch_window = switch_window

        # Pivot
        self.pivot_reward = pivot_reward

        # Misc
        self.immune_move_penalty = immune_move_penalty
        self.turn_penalty = turn_penalty

        # Danger / threat
        self.danger_penalty = danger_penalty
        self.kill_threat_bonus = kill_threat_bonus
        self.danger_reduction_bonus = danger_reduction_bonus
        self.high_danger_threshold = high_danger_threshold
        self.low_threat_threshold = low_threat_threshold
        self.high_danger_extra_penalty = high_danger_extra_penalty

        # Internal heuristic for matchup evaluation
        self.heuristic = SwitchHeuristics()

        self.reset()

    # ============================================================
    def reset(self):
        self.prev_my_bucket = None
        self.prev_opp_bucket = None

        self.prev_my_fainted = 0
        self.prev_opp_fainted = 0

        self.prev_my_status = 0
        self.prev_opp_status = 0

        self.prev_my_boosts = {
            "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0
        }
        self.prev_opp_boosts = {
            "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0
        }

        self.prev_matchup_score = None
        self.prev_danger = 0.0
        self.prev_kill_threat = 0.0

        self.last_switch_species = None
        self.last_active_species = None

        self.consecutive_switches = 0
        self.recent_switches = []

        self.turn_reward = 0.0

    def add(self, v):
        self.turn_reward += v

    def flush(self):
        r = self.turn_reward
        self.turn_reward = 0.0
        return r

    # ============================================================
    def compute_terminal_reward(self, battle):
        if battle.won:
            return self.terminal_win
        if battle.lost:
            return self.terminal_loss
        return self.terminal_draw

    # ============================================================
    # MAIN REWARD FUNCTION
    # ============================================================
    def compute_turn_reward(
        self,
        prev_battle,
        battle,
        action_was_switch=False,
        last_used_move=None,
    ):
        
        my = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        # Always compute current matchup score
        new_match = self.heuristic.estimate_matchup(my, opp)
        if my is None or opp is None:
            return 0.0

        reward = 0.0

        # FIRST TURN: initialize tracking
        if self.prev_my_bucket is None:
            self.prev_my_bucket = hp_bucket(my.current_hp_fraction)
            self.prev_opp_bucket = hp_bucket(opp.current_hp_fraction)

            self.prev_my_fainted = sum(p.fainted for p in battle.team.values())
            self.prev_opp_fainted = sum(p.fainted for p in battle.opponent_team.values())

            self.prev_my_status = sum(1 for p in battle.team.values() if p.status)
            self.prev_opp_status = sum(1 for p in battle.opponent_team.values() if p.status)

            self.prev_my_boosts = my.boosts.copy()
            self.prev_opp_boosts = opp.boosts.copy()

            d, _ = danger_score(my, opp)
            self.prev_danger = d
            self.prev_kill_threat = kill_threat_score(my, opp)

            self.prev_matchup_score = self.heuristic.estimate_matchup(my, opp)
            self.last_active_species = my.species
            return 0.0

        # ============================================================
        # 1. HP CHANGES (reduced scale)
        # ============================================================
        my_b = hp_bucket(my.current_hp_fraction)
        opp_b = hp_bucket(opp.current_hp_fraction)

        opp_drop = self.prev_opp_bucket - opp_b
        my_drop = self.prev_my_bucket - my_b

        reward += opp_drop * self.hp_reward
        reward -= my_drop * self.hp_penalty

        # ============================================================
        # 2. KOs
        # ============================================================
        my_fainted = sum(p.fainted for p in battle.team.values())
        opp_fainted = sum(p.fainted for p in battle.opponent_team.values())

        if opp_fainted > self.prev_opp_fainted:
            reward += self.ko_reward
        if my_fainted > self.prev_my_fainted:
            reward -= self.faint_penalty

        # ============================================================
        # 3. Status
        # ============================================================
        my_status = sum(1 for p in battle.team.values() if p.status)
        opp_status = sum(1 for p in battle.opponent_team.values() if p.status)

        reward += (opp_status - self.prev_opp_status) * self.status_reward
        reward -= (my_status - self.prev_my_status) * self.status_penalty

        # ============================================================
        # 4. Boosts (diminishing returns)
        # ============================================================
        for stat in ("atk", "spa"):
            old = self.prev_my_boosts[stat]
            new = my.boosts[stat]
            if new > old:
                reward += diminishing_boost_value(new, self.boost_base) - diminishing_boost_value(old, self.boost_base)

        for stat in ("def", "spd"):
            old = self.prev_my_boosts[stat]
            new = my.boosts[stat]
            if new > old:
                reward += 0.5 * (diminishing_boost_value(new, self.boost_base) - diminishing_boost_value(old, self.boost_base))

        # Speed boosts
        old_spe = self.prev_my_boosts["spe"]
        new_spe = my.boosts["spe"]
        if new_spe > old_spe:
            reward += 3.0 + (new_spe - 1) * 0.5

        # ============================================================
        # 5. Danger / Threat shaping (major driver for switching)
        # ============================================================
        new_danger, _ = danger_score(my, opp)
        new_kill = kill_threat_score(my, opp)

        delta_danger = new_danger - self.prev_danger
        delta_kill = new_kill - self.prev_kill_threat

        # Kill threat increases
        reward += max(0, delta_kill) * self.kill_threat_bonus
        reward += min(0, delta_kill) * (self.kill_threat_bonus * 0.5)

        # Danger penalties
        reward -= max(0, delta_danger) * self.danger_penalty
        reward += max(0, -delta_danger) * self.danger_reduction_bonus

        # Very bad states
        if new_danger >= self.high_danger_threshold and new_kill <= self.low_threat_threshold:
            reward -= self.high_danger_extra_penalty

        # ============================================================
        # 6. Suicide boosting penalty
        # ============================================================
        if my_fainted > self.prev_my_fainted and last_used_move:
            mv = last_used_move.id.lower()
            if mv in BOOSTING_MOVES:
                reward -= self.suicide_boost_penalty

        # ============================================================
        # 7. Immune move penalty
        # ============================================================
        if last_used_move:
            try:
                eff = last_used_move.type.damage_multiplier(opp.type_1, opp.type_2)
            except:
                eff = 1
            if eff == 0:
                reward -= self.immune_move_penalty

        # ============================================================
        # 8. SWITCHING REWARDS (major change)
        # ============================================================
        forced_switch = (my_fainted > self.prev_my_fainted)

        if action_was_switch and not forced_switch:
            # Baseline reward
            reward += self.baseline_switch_reward

            new_match = self.heuristic.estimate_matchup(my, opp)
            matchup_delta = new_match - self.prev_matchup_score

            danger_delta = self.prev_danger - new_danger     # positive if safer
            threat_delta = new_kill - self.prev_kill_threat  # positive if more pressure

            reward += self.switch_matchup_weight * matchup_delta
            reward += self.switch_danger_weight * max(0, danger_delta)
            reward += self.switch_threat_weight * max(0, threat_delta)

            # Penalize if switching makes everything worse
            if danger_delta < 0 and threat_delta <= 0:
                reward -= 2.0

            # wasted boosts
            total_pos = sum(max(0, self.prev_my_boosts[s]) for s in self.prev_my_boosts)
            reward -= total_pos * self.wasted_boost_penalty

            # escape debuffs reward
            total_neg = sum(abs(min(0, self.prev_my_boosts[s])) for s in self.prev_my_boosts)
            reward += total_neg * self.debuff_escape_reward

            # switch spam logic
            self.consecutive_switches += 1
            self.recent_switches.append(1)
            if len(self.recent_switches) > self.switch_window:
                self.recent_switches.pop(0)

            if self.consecutive_switches > 1:
                reward -= (self.consecutive_switches - 1) * self.switch_chain_penalty

            if sum(self.recent_switches) >= 3:
                reward -= self.switch_spam_penalty

            if self.last_switch_species == my.species:
                reward -= self.oscillation_penalty

            self.last_switch_species = self.last_active_species

        else:
            # Non-switch turn
            self.consecutive_switches = 0
            self.recent_switches.append(0)
            if len(self.recent_switches) > self.switch_window:
                self.recent_switches.pop(0)

        # ============================================================
        # 9. Pivot moves
        # ============================================================
        if last_used_move and last_used_move.id.lower() in PIVOT_MOVES:
            reward += self.pivot_reward

        # ============================================================
        # 10. Turn pacing
        # ============================================================
        reward -= self.turn_penalty

        # ============================================================
        # UPDATE MEMORY
        # ============================================================
        self.prev_my_bucket = my_b
        self.prev_opp_bucket = opp_b

        self.prev_my_fainted = my_fainted
        self.prev_opp_fainted = opp_fainted

        self.prev_my_status = my_status
        self.prev_opp_status = opp_status

        self.prev_my_boosts = my.boosts.copy()
        self.prev_opp_boosts = opp.boosts.copy()

        # Always update matchup score (safe, prevents unbound variable)
        self.prev_matchup_score = new_match

        self.prev_danger = new_danger
        self.prev_kill_threat = new_kill

        self.last_active_species = my.species

        self.add(reward)
        return 0.0