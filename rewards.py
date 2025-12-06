import numpy as np
from poke_env.battle.battle import Battle
from poke_env.battle.side_condition import SideCondition
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

# -----------------------------
# TYPE DATA (GEN 9)
# -----------------------------
_GEN9_DATA = GenData.from_gen(9)
_TYPE_CHART = _GEN9_DATA.type_chart


# ========================================
# NEW: HP FRACTION → BUCKET HELPER
# ========================================
def hp_bucket_frac(frac):
    """Convert HP fraction into bucket index:
       0 = 0–25%
       1 = 25–50%
       2 = 50–75%
       3 = 75–100%
    """
    if frac <= 0.25: return 0
    if frac <= 0.50: return 1
    if frac <= 0.75: return 2
    return 3


class RewardCalculator:
    def __init__(
        self,
        enable_intermediate: bool = True,
        # terminal outcomes
        terminal_win: float = 100.0,
        terminal_loss: float = -100.0,
        terminal_draw: float = 0.0,
        # HP / damage shaping
        hp_weight: float = 1.5,
        ko_reward: float = 30.0,
        ko_penalty: float = -25.0,
        status_reward: float = 2.0,
        status_penalty: float = -1.5,
        type_effectiveness_reward: float = 6.0,
        type_effectiveness_penalty: float = -3.0,
        switch_penalty: float = -0.7,
        turn_penalty: float = -0.08,
        momentum_reward: float = 1.0,
        momentum_turns: int = 3,
    ):
        self.enable_intermediate = enable_intermediate
        self.terminal_win = terminal_win
        self.terminal_loss = terminal_loss
        self.terminal_draw = terminal_draw

        # Intermediate reward weights
        self.hp_weight = hp_weight
        self.ko_reward = ko_reward
        self.ko_penalty = ko_penalty
        self.status_reward = status_reward
        self.status_penalty = status_penalty
        self.type_effectiveness_reward = type_effectiveness_reward
        self.type_effectiveness_penalty = type_effectiveness_penalty
        self.switch_penalty = switch_penalty
        self.turn_penalty = turn_penalty
        self.momentum_reward = momentum_reward
        self.momentum_turns = momentum_turns

        # hazard / screen constants
        self.hazard_sr_set_reward = 4.0
        self.hazard_spikes_layer_reward = 2.0
        self.hazard_tspikes_layer_reward = 2.5
        self.hazard_remove_reward = 3.0
        self.hazard_removed_penalty = -3.0
        self.hazard_switch_penalty_scale = 0.7
        self.screen_set_reward = 3.0
        self.screen_lost_penalty = -2.0
        self.forced_switch_reward = 2.0
        self.forced_switch_hazard_bonus = 1.0

        # OLD HP advantage tracking (still used for momentum)
        self.prev_hp_advantage = 0.0
        self.prev_our_hp_pct = None
        self.prev_opp_hp_pct = None

        # NEW: HP bucket tracking
        self.prev_my_bucket = None
        self.prev_opp_bucket = None

        # fainted/status counters
        self.prev_our_fainted = 0
        self.prev_opp_fainted = 0
        self.prev_our_status = 0
        self.prev_opp_status = 0

        # boost tracking
        self.prev_our_boosts = {"atk": 0, "spa": 0, "spe": 0, "def": 0, "spd": 0}
        self.prev_opp_boosts = {"atk": 0, "spa": 0, "spe": 0, "def": 0, "spd": 0}

        # hazards/screens
        self.prev_hazards_our = {"sr": 0, "spikes": 0, "tspikes": 0, "webs": 0}
        self.prev_hazards_opp = {"sr": 0, "spikes": 0, "tspikes": 0, "webs": 0}
        self.prev_screens_our = {"reflect": 0, "lscreen": 0, "veil": 0}
        self.prev_screens_opp = {"reflect": 0, "lscreen": 0, "veil": 0}

        # fields
        self.prev_weather = None
        self.prev_terrain = None
        self.prev_opp_active_id = None

        # momentum system
        self.consecutive_favorable_turns = 0


    # ================================================================
    # (unchanged helper methods omitted for brevity — keep as-is)
    # ================================================================
    def _stat_boost_debuff_reward(self, battle, damage_dealt):
        reward = 0.0
        if battle.active_pokemon:
            boosts = battle.active_pokemon.boosts
            for stat, prev in self.prev_our_boosts.items():
                now = boosts.get(stat, 0)
                if now > prev:
                    reward += 1.5
        if battle.opponent_active_pokemon:
            boosts = battle.opponent_active_pokemon.boosts
            for stat, prev in self.prev_opp_boosts.items():
                now = boosts.get(stat, 0)
                if now > prev:
                    reward -= 1.5
        return reward

    def _hazard_and_screen_reward(self, battle, action_was_switch):
        reward = 0.0
        new_our = self._extract_hazards(battle.side_conditions)
        new_opp = self._extract_hazards(battle.opponent_side_conditions)

        # Stealth Rock
        if new_opp["sr"] > self.prev_hazards_opp["sr"]:
            reward += self.hazard_sr_set_reward
        if new_our["sr"] < self.prev_hazards_our["sr"]:
            reward += self.hazard_removed_penalty

        # Spikes differing
        if new_opp["spikes"] > self.prev_hazards_opp["spikes"]:
            reward += self.hazard_spikes_layer_reward
        if new_our["spikes"] < self.prev_hazards_our["spikes"]:
            reward += self.hazard_removed_penalty

        # Screens
        s_our = self._extract_screens(battle.side_conditions)
        s_opp = self._extract_screens(battle.opponent_side_conditions)
        if s_our["reflect"] > self.prev_screens_our["reflect"]:
            reward += self.screen_set_reward
        if s_opp["reflect"] > self.prev_screens_opp["reflect"]:
            reward -= self.screen_lost_penalty

        return reward

    def _good_switch_reward(self, battle, damage_dealt, damage_taken):
        # Positive reward when switching avoids big damage
        if damage_taken < 0.05 and damage_dealt > 0.15:
            return 2.0
        return self.switch_penalty

    def _forced_switch_reward(self, battle):
        reward = 0.0
        opp_id = self._get_opp_active_id(battle)
        if opp_id != self.prev_opp_active_id:
            reward += self.forced_switch_reward
        return reward

    def _small_weather_terrain_reward(self, battle):
        # reward minor field control, optional
        return 0.0
    def _get_hp_percentage(self, battle, opponent=False):
        team = battle.opponent_team if opponent else battle.team
        total = 0
        maxhp = 0
        for p in team.values():
            if p and p.max_hp:
                maxhp += p.max_hp
                total += p.current_hp or 0
        return (total / maxhp) if maxhp else 0.0

    def _count_fainted(self, battle, opponent=False):
        team = battle.opponent_team if opponent else battle.team
        return sum(1 for p in team.values() if p and p.fainted)

    def _count_status_conditions(self, battle, opponent=False):
        team = battle.opponent_team if opponent else battle.team
        return sum(1 for p in team.values() if p and p.status)

    def _extract_hazards(self, side_conditions):
        return {
            "sr": int(SideCondition.STEALTH_ROCK in side_conditions),
            "spikes": side_conditions.get(SideCondition.SPIKES, 0),
            "tspikes": side_conditions.get(SideCondition.TOXIC_SPIKES, 0),
            "webs": int(SideCondition.STICKY_WEB in side_conditions)
        }

    def _extract_screens(self, side_conditions):
        return {
            "reflect": int(SideCondition.REFLECT in side_conditions),
            "lscreen": int(SideCondition.LIGHT_SCREEN in side_conditions),
            "veil": int(SideCondition.AURORA_VEIL in side_conditions)
        }

    def _snapshot_boosts(self, battle):
        if battle.active_pokemon:
            b = battle.active_pokemon.boosts
            self.prev_our_boosts = {
                "atk": b.get("atk", 0),
                "spa": b.get("spa", 0),
                "spe": b.get("spe", 0),
                "def": b.get("def", 0),
                "spd": b.get("spd", 0)
            }
        if battle.opponent_active_pokemon:
            b = battle.opponent_active_pokemon.boosts
            self.prev_opp_boosts = {
                "atk": b.get("atk", 0),
                "spa": b.get("spa", 0),
                "spe": b.get("spe", 0),
                "def": b.get("def", 0),
                "spd": b.get("spd", 0)
            }

    def _get_opp_active_id(self, battle):
        opp = battle.opponent_active_pokemon
        return opp.species if opp else None


    # ================================================================
    # MAIN TURN REWARD
    # ================================================================
    def compute_turn_reward(
        self,
        prev_battle,
        current_battle: Battle,
        action_was_switch: bool = False
    ) -> float:

        if not self.enable_intermediate:
            return 0.0

        reward = 0.0

        # ============================================================
        # NEW — per-active-mon HP BUCKETS
        # ============================================================
        my_active = current_battle.active_pokemon
        opp_active = current_battle.opponent_active_pokemon

        if my_active and my_active.max_hp:
            my_frac = (my_active.current_hp or 0) / my_active.max_hp
            my_bucket = hp_bucket_frac(my_frac)
        else:
            my_bucket = None

        if opp_active and opp_active.max_hp:
            opp_frac = (opp_active.current_hp or 0) / opp_active.max_hp
            opp_bucket = hp_bucket_frac(opp_frac)
        else:
            opp_bucket = None

        # FIRST TURN: Initialize bucket memory
        if self.prev_my_bucket is None or self.prev_opp_bucket is None:
            self.prev_my_bucket = my_bucket
            self.prev_opp_bucket = opp_bucket
            # also initialize old HP for other parts of reward
            self.prev_our_hp_pct = self._get_hp_percentage(current_battle, False)
            self.prev_opp_hp_pct = self._get_hp_percentage(current_battle, True)
            self.prev_hp_advantage = self.prev_our_hp_pct - self.prev_opp_hp_pct
            self.prev_our_fainted = self._count_fainted(current_battle, False)
            self.prev_opp_fainted = self._count_fainted(current_battle, True)
            self.prev_our_status = self._count_status_conditions(current_battle, False)
            self.prev_opp_status = self._count_status_conditions(current_battle, True)
            self.prev_hazards_our = self._extract_hazards(current_battle.side_conditions)
            self.prev_hazards_opp = self._extract_hazards(current_battle.opponent_side_conditions)
            self.prev_screens_our = self._extract_screens(current_battle.side_conditions)
            self.prev_screens_opp = self._extract_screens(current_battle.opponent_side_conditions)
            self._snapshot_boosts(current_battle)
            self.prev_opp_active_id = self._get_opp_active_id(current_battle)
            return 0.0

        # ============================================================
        # HP BUCKET REWARD (NEW)
        # ============================================================
        # --- Opponent bucket changes ---
        if opp_bucket is not None and self.prev_opp_bucket is not None:
            delta = opp_bucket - self.prev_opp_bucket
            if delta < 0:   # opponent bucket DOWN → good
                reward += abs(delta) * 4.0
            elif delta > 0: # opponent healed → bad
                reward -= abs(delta) * 3.0

        # --- Our bucket changes ---
        if my_bucket is not None and self.prev_my_bucket is not None:
            delta = my_bucket - self.prev_my_bucket
            if delta < 0:   # we got chunked → bad
                reward -= abs(delta) * 4.0
            elif delta > 0: # we healed → good
                reward += abs(delta) * 3.0


        # ============================================================
        # (KEEP ALL YOUR EXISTING REWARD SHAPING — UNTOUCHED)
        # ============================================================

        # HP percentage macro shaping (momentum signal)
        our_hp = self._get_hp_percentage(current_battle, False)
        opp_hp = self._get_hp_percentage(current_battle, True)
        hp_adv = our_hp - opp_hp
        hp_change = hp_adv - self.prev_hp_advantage
        reward += hp_change * self.hp_weight

        if hp_change > 0:
            self.consecutive_favorable_turns += 1
            if self.consecutive_favorable_turns >= self.momentum_turns:
                reward += self.momentum_reward
        else:
            self.consecutive_favorable_turns = 0

        # Damage chunk bonuses
        damage_dealt = max(0.0, self.prev_opp_hp_pct - opp_hp)
        damage_taken = max(0.0, self.prev_our_hp_pct - our_hp)

        if damage_dealt > 0.30:
            reward += self.type_effectiveness_reward
        elif 0 < damage_dealt < 0.10:
            reward += self.type_effectiveness_penalty

        if damage_taken > 0.30:
            reward -= abs(self.type_effectiveness_penalty)

        # KO rewards
        our_fainted = self._count_fainted(current_battle, False)
        opp_fainted = self._count_fainted(current_battle, True)
        if opp_fainted > self.prev_opp_fainted:
            reward += self.ko_reward
        if our_fainted > self.prev_our_fainted:
            reward += self.ko_penalty

        # Status
        our_status = self._count_status_conditions(current_battle, False)
        opp_status = self._count_status_conditions(current_battle, True)
        if our_status > self.prev_our_status:
            reward += self.status_penalty
        if opp_status > self.prev_opp_status:
            reward += self.status_reward

        # Boosts, hazards, good-switch reward, forced switch reward
        reward += self._stat_boost_debuff_reward(current_battle, damage_dealt)
        reward += self._hazard_and_screen_reward(current_battle, action_was_switch)
        if action_was_switch:
            reward += self._good_switch_reward(current_battle, damage_dealt, damage_taken)
        reward += self._forced_switch_reward(current_battle)

        reward += self._small_weather_terrain_reward(current_battle)
        reward += self.turn_penalty

        # ============================================================
        # UPDATE INTERNAL MEMORY
        # ============================================================
        self.prev_my_bucket = my_bucket
        self.prev_opp_bucket = opp_bucket

        self.prev_our_hp_pct = our_hp
        self.prev_opp_hp_pct = opp_hp
        self.prev_hp_advantage = hp_adv

        self.prev_our_fainted = our_fainted
        self.prev_opp_fainted = opp_fainted
        self.prev_our_status = our_status
        self.prev_opp_status = opp_status

        self.prev_hazards_our = self._extract_hazards(current_battle.side_conditions)
        self.prev_hazards_opp = self._extract_hazards(current_battle.opponent_side_conditions)
        self.prev_screens_our = self._extract_screens(current_battle.side_conditions)
        self.prev_screens_opp = self._extract_screens(current_battle.opponent_side_conditions)

        self._snapshot_boosts(current_battle)
        self.prev_opp_active_id = self._get_opp_active_id(current_battle)

        return reward


    # ================================================================
    # RESET FOR NEW BATTLE
    # ================================================================
    def reset(self):
        self.prev_hp_advantage = 0.0
        self.prev_our_hp_pct = None
        self.prev_opp_hp_pct = None

        # NEW
        self.prev_my_bucket = None
        self.prev_opp_bucket = None

        self.prev_our_fainted = 0
        self.prev_opp_fainted = 0
        self.prev_our_status = 0
        self.prev_opp_status = 0

        self.prev_our_boosts = {"atk": 0, "spa": 0, "spe": 0, "def": 0, "spd": 0}
        self.prev_opp_boosts = {"atk": 0, "spa": 0, "spe": 0, "def": 0, "spd": 0}

        self.prev_hazards_our = {"sr": 0, "spikes": 0, "tspikes": 0, "webs": 0}
        self.prev_hazards_opp = {"sr": 0, "spikes": 0, "tspikes": 0, "webs": 0}
        self.prev_screens_our = {"reflect": 0, "lscreen": 0, "veil": 0}
        self.prev_screens_opp = {"reflect": 0, "lscreen": 0, "veil": 0}

        self.prev_weather = None
        self.prev_terrain = None
        self.prev_opp_active_id = None
        self.consecutive_favorable_turns = 0
