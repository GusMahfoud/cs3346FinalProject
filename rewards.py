# rewards.py — TUNED FOR DANGER/THREAT + RL SWITCHING

import numpy as np
from poke_env.data import GenData
from computed_stats import get_real_stats
from advanced_switcher import SwitchHeuristics
from state_encoder import estimate_damage, danger_score, kill_threat_score

GEN9_DATA = GenData.from_gen(9)
TYPE_CHART = GEN9_DATA.type_chart

PIVOT_MOVES = {"uturn", "voltswitch", "flipturn", "partingshot", "batonpass"}


# ------------------------------------------------------------
# HP BUCKET
# ------------------------------------------------------------
def hp_bucket(frac):
    """
    10 buckets from 0 → 9.
    Each bucket = 10% HP.
    """
    frac = max(0.0, min(1.0, frac))
    return int(frac * 10)   # 0–9


# ------------------------------------------------------------
# BOOST DIMINISHING RETURNS
# ------------------------------------------------------------
def stage_multiplier(stage):
    stage = max(-6, min(6, stage))
    mults = {
        -6: 2/8, -5: 2/7, -4: 2/6, -3: 2/5, -2: 2/4, -1: 2/3,
         0: 1.0,
        +1: 3/2, +2: 2.0, +3: 5/2, +4: 3.0, +5: 7/2, +6: 4.0
    }
    return mults[stage]


def diminishing_boost_value(stage, base=8):
    if stage <= 0:
        return 0
    mult = stage_multiplier(stage)
    return base * np.log2(mult)


BOOSTING_MOVES = {"swordsdance", "nastyplot", "quiverdance", "calmmind", "torchsong"}
UTILITY_MOVES = {"willowisp", "recover", "slackoff"}


# ===========================================================
# REWARD CALCULATOR
# ===========================================================
class RewardCalculator:

    def __init__(
        self,
        # Game outcome
        terminal_win=80,
        terminal_loss=-90,
        terminal_draw=0,

        # HP shaping
        hp_reward=3.5,
        hp_penalty=5.0,

        # KOs
        ko_reward=25,
        faint_penalty=22,

        # Status
        status_reward=3.5,
        status_penalty=5.0,

        # Boosts
        boost_base=7,
        suicide_boost_penalty=15,

        # Switching / oscillation
        switch_matchup_weight=4.0,
        switch_danger_weight=5.0,
        switch_threat_weight=3.0,
        wasted_boost_penalty=3.0,
        debuff_escape_reward=3.0,
        oscillation_penalty=4.0,
        switch_chain_penalty=1.0,
        switch_spam_penalty=1.5,
        switch_window=5,

        # Moves
        pivot_reward=3.0,
        immune_move_penalty=8.0,
        turn_penalty=0.04,

        # Danger / threat
        danger_penalty=8.0,
        kill_threat_bonus=7.0,
        danger_reduction_bonus=6.0,
        high_danger_threshold=0.85,
        low_threat_threshold=0.35,
        high_danger_extra_penalty=10.0,
    ):

        # Terminal
        self.terminal_win = terminal_win
        self.terminal_loss = terminal_loss
        self.terminal_draw = terminal_draw

        # HP / KOs / status
        self.hp_reward = hp_reward
        self.hp_penalty = hp_penalty
        self.ko_reward = ko_reward
        self.faint_penalty = faint_penalty
        self.status_reward = status_reward
        self.status_penalty = status_penalty

        # Boosts
        self.boost_base = boost_base
        self.suicide_boost_penalty = suicide_boost_penalty

        # Switch shaping
        self.switch_matchup_weight = switch_matchup_weight
        self.switch_danger_weight = switch_danger_weight
        self.switch_threat_weight = switch_threat_weight
        self.wasted_boost_penalty = wasted_boost_penalty
        self.debuff_escape_reward = debuff_escape_reward
        self.oscillation_penalty = oscillation_penalty
        self.switch_chain_penalty = switch_chain_penalty
        self.switch_spam_penalty = switch_spam_penalty
        self.switch_window = switch_window

        # Moves
        self.pivot_reward = pivot_reward
        self.immune_move_penalty = immune_move_penalty
        self.turn_penalty = turn_penalty

        # Danger / threat weights
        self.danger_penalty = danger_penalty
        self.kill_threat_bonus = kill_threat_bonus
        self.danger_reduction_bonus = danger_reduction_bonus
        self.high_danger_threshold = high_danger_threshold
        self.low_threat_threshold = low_threat_threshold
        self.high_danger_extra_penalty = high_danger_extra_penalty

        # Heuristic only for evaluating matchup quality (not for decisions)
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

        self.prev_my_boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
        self.prev_opp_boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}

        self.prev_matchup_score = None
        self.last_switch_species = None
        self.last_active_species = None

        self.prev_danger = 0.0
        self.prev_kill_threat = 0.0

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
    def compute_switch_heuristic_score(self, battle):
        a = battle.active_pokemon
        o = battle.opponent_active_pokemon
        if a is None or o is None:
            return 0.0
        return self.heuristic.estimate_matchup(a, o)

    # ============================================================
    def compute_terminal_reward(self, battle):
        if battle.won:
            return self.terminal_win
        if battle.lost:
            return self.terminal_loss
        return self.terminal_draw

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
        if my is None or opp is None:
            return 0.0

        reward = 0.0

        # ============================================================
        # FIRST TURN: initialize memory only
        # ============================================================
        if self.prev_my_bucket is None:
            self.prev_my_bucket = hp_bucket(my.current_hp_fraction)
            self.prev_opp_bucket = hp_bucket(opp.current_hp_fraction)

            self.prev_my_fainted = sum(p.fainted for p in battle.team.values())
            self.prev_opp_fainted = sum(
                p.fainted for p in battle.opponent_team.values()
            )

            self.prev_my_status = sum(1 for p in battle.team.values() if p.status)
            self.prev_opp_status = sum(
                1 for p in battle.opponent_team.values() if p.status
            )

            self.prev_my_boosts = my.boosts.copy()
            self.prev_opp_boosts = opp.boosts.copy()

            d, _ = danger_score(my, opp)
            kt = kill_threat_score(my, opp)
            self.prev_danger = d
            self.prev_kill_threat = kt

            self.prev_matchup_score = self.compute_switch_heuristic_score(battle)
            self.last_active_species = my.species
            return 0.0

        # ============================================================
        # 1. HP bucket change (per 10% chunks)
        # ============================================================
        my_b = hp_bucket(my.current_hp_fraction)
        opp_b = hp_bucket(opp.current_hp_fraction)

        opp_drop = self.prev_opp_bucket - opp_b   # >0 => we damaged them
        my_drop = self.prev_my_bucket - my_b      # >0 => we took damage

        if opp_drop > 0:
            reward += self.hp_reward * opp_drop

        if my_drop > 0:
            reward -= self.hp_penalty * my_drop

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

        if opp_status > self.prev_opp_status:
            reward += self.status_reward
        if my_status > self.prev_my_status:
            reward -= self.status_penalty

        # ============================================================
        # 4. Boosts with diminishing returns
        # ============================================================
        for stat in ("atk", "spa"):
            old = self.prev_my_boosts.get(stat, 0)
            new = my.boosts.get(stat, 0)
            if new > old:
                reward += diminishing_boost_value(
                    new, self.boost_base
                ) - diminishing_boost_value(old, self.boost_base)

        for stat in ("def", "spd"):
            old = self.prev_my_boosts.get(stat, 0)
            new = my.boosts.get(stat, 0)
            if new > old:
                reward += 0.5 * (
                    diminishing_boost_value(new, self.boost_base)
                    - diminishing_boost_value(old, self.boost_base)
                )

        old_spe = self.prev_my_boosts.get("spe", 0)
        new_spe = my.boosts.get("spe", 0)
        if new_spe > old_spe:
            reward += 5.0
            if new_spe > 1:
                reward += (new_spe - 1) * 1.0

        # ============================================================
        # 5. Danger & Kill Threat shaping
        # ============================================================
        new_danger, _ = danger_score(my, opp)
        new_kill_threat = kill_threat_score(my, opp)

        delta_kill = new_kill_threat - self.prev_kill_threat
        delta_danger = new_danger - self.prev_danger

        # Reward for gaining more kill pressure
        if delta_kill > 0:
            reward += self.kill_threat_bonus * delta_kill
        elif delta_kill < 0:
            # slight penalty for losing pressure
            reward += 0.5 * self.kill_threat_bonus * delta_kill  # negative

        # Penalize increased danger, reward decreased danger
        if delta_danger > 0:
            reward -= self.danger_penalty * delta_danger
        elif delta_danger < 0:
            reward += self.danger_reduction_bonus * (-delta_danger)

        # If we end up in a very bad state (opponent can likely kill us and we can’t kill back)
        if (
            new_danger >= self.high_danger_threshold
            and new_kill_threat <= self.low_threat_threshold
        ):
            reward -= self.high_danger_extra_penalty

        # ============================================================
        # 6. Suicide boosting (die after boosting when you were faster)
        # ============================================================
        if my_fainted > self.prev_my_fainted and last_used_move:
            mv = last_used_move.id.lower()
            if mv in BOOSTING_MOVES:
                my_speed = get_real_stats(my.species)["spe"]
                opp_speed = get_real_stats(opp.species)["spe"]
                if my_speed >= opp_speed:
                    reward -= self.suicide_boost_penalty

        # ============================================================
        # 7. Useless / immune move penalty
        # ============================================================
        if last_used_move:
            eff = 1
            try:
                eff = last_used_move.type.damage_multiplier(opp.type_1, opp.type_2)
            except Exception:
                pass

            # full immunity: hard punish (teaches reading immunities)
            if eff == 0:
                reward -= self.immune_move_penalty

        # ============================================================
        # 8. Switch rewards (only for voluntary switches)
        # ============================================================
        forced_switch = (my_fainted > self.prev_my_fainted)

        if action_was_switch and not forced_switch:
            new_score = self.compute_switch_heuristic_score(battle)
            matchup_delta = new_score - self.prev_matchup_score

            # Compare danger / threat across the switch
            d_delta = self.prev_danger - new_danger         # >0 => safer
            t_delta = new_kill_threat - self.prev_kill_threat  # >0 => more threatening

            # Reward switches that both:
            # - improve matchup
            # - reduce danger
            # - increase kill threat
            reward += self.switch_matchup_weight * matchup_delta
            reward += self.switch_danger_weight * max(d_delta, 0.0)
            reward += self.switch_threat_weight * max(t_delta, 0.0)

            # Penalize "coward switches" that keep you just as threatened and don't pressure
            if d_delta < 0 and t_delta <= 0:
                reward -= 2.0

            # Wasted boosts (switching out after stacking boosts)
            total_pos = sum(
                max(0, self.prev_my_boosts.get(s, 0))
                for s in ("atk", "def", "spa", "spd", "spe")
            )
            if total_pos > 0:
                reward -= self.wasted_boost_penalty * total_pos

            # Reward escaping bad debuffs
            total_neg = sum(
                abs(min(0, self.prev_my_boosts.get(s, 0)))
                for s in ("atk", "def", "spa", "spd", "spe")
            )
            if total_neg > 0:
                reward += self.debuff_escape_reward * total_neg

            # Switch spam / oscillation tracking
            self.consecutive_switches += 1
            self.recent_switches.append(1)

            if self.consecutive_switches > 1:
                reward -= self.switch_chain_penalty * (self.consecutive_switches - 1)

            if len(self.recent_switches) > self.switch_window:
                self.recent_switches.pop(0)

            if sum(self.recent_switches) >= 3:
                reward -= self.switch_spam_penalty

            if self.last_switch_species == my.species:
                reward -= self.oscillation_penalty

            self.last_switch_species = self.last_active_species

        else:
            # Not a switch: decay the switch chain count slowly
            self.consecutive_switches = 0
            self.recent_switches.append(0)
            if len(self.recent_switches) > self.switch_window:
                self.recent_switches.pop(0)

        # ============================================================
        # 9. Pivot moves (U-turn / Volt Switch / etc.)
        # ============================================================
        if last_used_move and last_used_move.id.lower() in PIVOT_MOVES:
            reward += self.pivot_reward

        # ============================================================
        # 10. Turn pacing penalty (keep games from dragging forever)
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
        self.prev_matchup_score = self.compute_switch_heuristic_score(battle)

        self.prev_danger = new_danger
        self.prev_kill_threat = new_kill_threat

        self.last_active_species = my.species

        self.add(reward)
        return 0.0