# ============================================================
# rewards.py — Clean RL-Shaping (v4.0)
# Removes switching-shape, fixes boost/HP resets, preserves all battle rewards
# ============================================================

import numpy as np
from poke_env.data import GenData
from advanced_switcher import SwitchHeuristics

GEN9_DATA = GenData.from_gen(9)
TYPE_CHART = GEN9_DATA.type_chart

PIVOT_MOVES = {"uturn", "voltswitch", "flipturn", "partingshot", "batonpass"}

BOOSTING_MOVES = {"swordsdance", "nastyplot", "quiverdance", "calmmind", "torchsong"}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def hp_bucket(frac):
    frac = max(0.0, min(1.0, frac))
    return int(frac * 10)


def stable_sigmoid(x):
    if x >= 0:
        z = np.exp(-x); return 1/(1+z)
    else:
        z = np.exp(x); return z/(1+z)


def stage_multiplier(stage):
    stage = max(-6, min(6, stage))
    table = {
        -6: 2/8, -5: 2/7, -4: 2/6, -3: 2/5, -2: 2/4, -1: 2/3,
         0: 1.0,
        +1: 3/2, +2: 2.0, +3: 5/2, +4: 3.0, +5: 7/2, +6: 4.0,
    }
    return table[stage]


def diminishing_boost_value(stage, base=8):
    if stage <= 0:
        return 0.0
    return base * np.log2(stage_multiplier(stage))


# ============================================================
# Reward Calculator
# ============================================================
class RewardCalculator:
    """
    Modern reward system:
    - No switching-shaping (handled heuristically)
    - Correct KO, HP, status, boost, debuff, pivot, immune, turn rewards
    - Correct reset behavior when mon changes (fixes all false penalties)
    """

    def __init__(
        self,
        terminal_win=80,
        terminal_loss=-90,
        terminal_draw=0,

        # HP
        hp_reward=1.5,
        hp_penalty=1.0,

        # KOs
        ko_reward=20,
        faint_penalty=18,

        # Status
        status_reward=3.0,
        status_penalty=3.0,

        # Boosting
        boost_reward_base=14,
        bulk_boost_factor=0.5,

        # Debuffs
        debuff_penalty_atkspa=6.0,
        debuff_penalty_bulk=3.0,

        # Suicide boosting
        suicide_boost_penalty=12.0,

        # Immune hits
        immune_move_penalty=10.0,

        # Turn pacing
        turn_penalty=0.02,
    ):

        self.terminal_win = terminal_win
        self.terminal_loss = terminal_loss
        self.terminal_draw = terminal_draw

        self.hp_reward = hp_reward
        self.hp_penalty = hp_penalty

        self.ko_reward = ko_reward
        self.faint_penalty = faint_penalty

        self.status_reward = status_reward
        self.status_penalty = status_penalty

        self.boost_reward_base = boost_reward_base
        self.bulk_boost_factor = bulk_boost_factor

        self.debuff_penalty_atkspa = debuff_penalty_atkspa
        self.debuff_penalty_bulk = debuff_penalty_bulk

        self.suicide_boost_penalty = suicide_boost_penalty

        self.immune_move_penalty = immune_move_penalty
        self.turn_penalty = turn_penalty

        self.heuristic = SwitchHeuristics()
        self.reset()

    # ------------------------------------------------------------
    def reset(self):
        #self.prev_matchup_score = None

        self.prev_my_fainted = 0
        self.prev_opp_fainted = 0

        self.prev_my_status = 0
        self.prev_opp_status = 0

        self.prev_boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}

        self.prev_hp_buckets = {}
        self.last_active_species = None

        self.turn_reward = 0.0

    def add(self, v):
        self.turn_reward += v

    def flush(self):
        r = self.turn_reward
        self.turn_reward = 0.0
        return r

    # ------------------------------------------------------------
    def compute_terminal_reward(self, battle):
        if battle.won:
            return self.terminal_win
        if battle.lost:
            return self.terminal_loss
        return self.terminal_draw

    # ------------------------------------------------------------
    def compute_turn_reward(
        self,
        prev_battle,
        battle,
        action_was_switch=False,
        last_used_move=None,
    ):

        my = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        if my is None or opp is None:
            return 0.0

        reward = 0.0
        #new_match = self.heuristic.estimate_matchup(my, opp, battle)

        # ---------------- FIRST TURN INIT ----------------
        if self.prev_my_fainted is None:
            #self.prev_matchup_score = new_match
            self.prev_my_fainted = sum(p.fainted for p in battle.team.values())
            self.prev_opp_fainted = sum(p.fainted for p in battle.opponent_team.values())
            self.prev_my_status = sum(1 for p in battle.team.values() if p.status)
            self.prev_opp_status = sum(1 for p in battle.opponent_team.values() if p.status)
            boosts = my.boosts if isinstance(my.boosts, dict) else my.boosts()
            self.prev_boosts = boosts.copy()
            self.prev_hp_buckets[my.species] = hp_bucket(my.current_hp_fraction)
            self.last_active_species = my.species
            return 0.0

        # ============================================================
        # FIX 1 — Reset HP bucket tracking when Pokémon changes
        # ============================================================
        if my.species != self.last_active_species:
            self.prev_hp_buckets[my.species] = hp_bucket(my.current_hp_fraction)
            # prevents fake HP change rewards on switch

        # ============================================================
        # FIX 2 — Reset boost tracking when Pokémon changes
        # ============================================================
        if my.species != self.last_active_species:
            self.prev_boosts = {"atk":0, "def":0, "spa":0, "spd":0, "spe":0}
            # prevents false debuff penalties or false boost rewards

        # ---------------- 1. HP CHANGE -----------------
        current_bucket = hp_bucket(my.current_hp_fraction)
        prev_bucket = self.prev_hp_buckets.get(my.species, current_bucket)

        if current_bucket < prev_bucket:
            reward -= (prev_bucket - current_bucket) * self.hp_penalty
        elif current_bucket > prev_bucket:
            reward += (current_bucket - prev_bucket) * self.hp_reward

        self.prev_hp_buckets[my.species] = current_bucket

        # ---------------- 2. KOs -----------------
        my_fainted = sum(p.fainted for p in battle.team.values())
        opp_fainted = sum(p.fainted for p in battle.opponent_team.values())

        if opp_fainted > self.prev_opp_fainted:
            reward += self.ko_reward

        if my_fainted > self.prev_my_fainted and last_used_move:
            if last_used_move.id.lower() in BOOSTING_MOVES:
                reward -= self.suicide_boost_penalty

        if my_fainted > self.prev_my_fainted:
            reward -= self.faint_penalty

        # ---------------- 3. STATUS CHANGES -----------------
        my_status = sum(1 for p in battle.team.values() if p.status)
        opp_status = sum(1 for p in battle.opponent_team.values() if p.status)

        reward += (opp_status - self.prev_opp_status) * self.status_reward
        reward -= (my_status - self.prev_my_status) * self.status_penalty

        # ---------------- 4. BOOST GAINS -----------------
        for stat in ("atk","spa"):
            old = self.prev_boosts[stat]
            new = my.boosts[stat]
            if new > old:
                reward += diminishing_boost_value(new, self.boost_reward_base) - diminishing_boost_value(old, self.boost_reward_base)

        for stat in ("def","spd"):
            old = self.prev_boosts[stat]
            new = my.boosts[stat]
            if new > old:
                inc = diminishing_boost_value(new, self.boost_reward_base) - diminishing_boost_value(old, self.boost_reward_base)
                reward += inc * self.bulk_boost_factor

        if my.boosts["spe"] > self.prev_boosts["spe"]:
            reward += 2.0

        # ---------------- 5. DEBUFF PENALTIES -----------------
        for stat in ("atk","spa"):
            if my.boosts[stat] < self.prev_boosts[stat]:
                reward -= (self.prev_boosts[stat] - my.boosts[stat]) * self.debuff_penalty_atkspa

        for stat in ("def","spd","spe"):
            if my.boosts[stat] < self.prev_boosts[stat]:
                reward -= (self.prev_boosts[stat] - my.boosts[stat]) * self.debuff_penalty_bulk

        # ---------------- 6. IMMUNE MOVE PENALTY -----------------
        if last_used_move:
            try:
                eff = last_used_move.type.damage_multiplier(opp.type_1, opp.type_2)
            except:
                eff = 1
            if eff == 0:
                reward -= self.immune_move_penalty

        # ---------------- 7. PIVOT BONUS -----------------
        if last_used_move and last_used_move.id.lower() in PIVOT_MOVES:
            reward += 2.0

        # ---------------- 8. TURN PENALTY -----------------
        reward -= self.turn_penalty

        # ---------------- UPDATE MEMORY -----------------
        #self.prev_matchup_score = new_match
        self.prev_my_fainted = my_fainted
        self.prev_opp_fainted = opp_fainted
        self.prev_my_status = my_status
        self.prev_opp_status = opp_status
        self.prev_boosts = my.boosts.copy()
        self.last_active_species = my.species

        self.add(reward)
        return 0.0