# rewards.py
import numpy as np
from poke_env.battle.battle import Battle
from poke_env.data import GenData

from computed_stats import get_real_stats

GEN9_DATA = GenData.from_gen(9)
TYPE_CHART = GEN9_DATA.type_chart

PIVOT_MOVES = {"uturn","voltswitch","flipturn","partingshot","batonpass"}


def hp_bucket(frac):
    if frac <= 0.25: return 0
    if frac <= 0.50: return 1
    if frac <= 0.75: return 2
    return 3


class RewardCalculator:

    def __init__(
        self,

        # Terminal
        terminal_win=100,
        terminal_loss=-120,
        terminal_draw=0,

        # HP buckets
        hp_reward=4,
        hp_penalty=6,

        # KO
        ko_reward=30,
        faint_penalty=35,

        # Status
        status_reward=4,
        status_penalty=6,

        # Boosts
        boost_reward=6,
        boost_penalty=8,
        suicide_boost_penalty=20,

        # Switching
        base_switch_penalty=1.0,
        switch_chain_penalty=2.5,
        switch_spam_penalty=3.0,
        oscillation_penalty=5.0,
        switch_window=5,

        # Matchup switching
        matchup_reward=4.0,
        matchup_penalty=4.0,

        # Speed advantage switching
        switch_speed_bonus=4.0,

        # Pivot moves
        pivot_reward=3.0,

        # Turn pacing
        turn_penalty=0.05,
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

        self.boost_reward = boost_reward
        self.boost_penalty = boost_penalty
        self.suicide_boost_penalty = suicide_boost_penalty

        self.base_switch_penalty = base_switch_penalty
        self.switch_chain_penalty = switch_chain_penalty
        self.switch_spam_penalty = switch_spam_penalty
        self.oscillation_penalty = oscillation_penalty
        self.switch_window = switch_window

        self.matchup_reward = matchup_reward
        self.matchup_penalty = matchup_penalty

        self.switch_speed_bonus = switch_speed_bonus

        self.pivot_reward = pivot_reward

        self.turn_penalty = turn_penalty

        self.reset()


    # ------------------------------------------------------
    def reset(self):
        self.prev_my_bucket = None
        self.prev_opp_bucket = None

        self.prev_my_fainted = 0
        self.prev_opp_fainted = 0

        self.prev_my_status = 0
        self.prev_opp_status = 0

        self.prev_my_boosts = {"atk":0,"spa":0}
        self.prev_opp_boosts = {"atk":0,"spa":0}

        self.consecutive_switches = 0
        self.recent_switches = []
        self.last_switch_species = None
        self.last_active_species = None

        self.prev_matchup_score = None


    # ------------------------------------------------------
    def matchup_score(self, my, opp):
        my_types = my.types or []
        opp_types = opp.types or []

        my_eff = 1.0
        opp_eff = 1.0

        for t in my_types:
            for ot in opp_types:
                if t in TYPE_CHART and ot in TYPE_CHART[t]:
                    my_eff *= TYPE_CHART[t][ot]

        for ot in opp_types:
            for t in my_types:
                if ot in TYPE_CHART and t in TYPE_CHART[ot]:
                    opp_eff *= TYPE_CHART[ot][t]

        return my_eff - opp_eff


    # ------------------------------------------------------
    def compute_terminal_reward(self, battle):
        if battle.won: return self.terminal_win
        if battle.lost: return self.terminal_loss
        return self.terminal_draw


    # ------------------------------------------------------
    def compute_turn_reward(self, prev_battle, battle, action_was_switch=False, last_used_move=None):
        my = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        if my is None or opp is None:
            return 0.0

        reward = 0.0

        # --------------------------------------------------
        # FIRST TURN INITIALIZATION
        # --------------------------------------------------
        if self.prev_my_bucket is None:
            self.prev_my_bucket = hp_bucket(my.current_hp_fraction)
            self.prev_opp_bucket = hp_bucket(opp.current_hp_fraction)

            self.prev_my_fainted = sum(p.fainted for p in battle.team.values())
            self.prev_opp_fainted = sum(p.fainted for p in battle.opponent_team.values())

            self.prev_my_status = sum(1 for p in battle.team.values() if p.status)
            self.prev_opp_status = sum(1 for p in battle.opponent_team.values() if p.status)

            self.prev_my_boosts = my.boosts.copy()
            self.prev_opp_boosts = opp.boosts.copy()

            self.prev_matchup_score = self.matchup_score(my, opp)
            self.last_active_species = my.species

            return 0.0

        # --------------------------------------------------
        # HP REWARDS
        # --------------------------------------------------
        my_b = hp_bucket(my.current_hp_fraction)
        opp_b = hp_bucket(opp.current_hp_fraction)

        if opp_b < self.prev_opp_bucket:
            reward += self.hp_reward
        if my_b < self.prev_my_bucket:
            reward -= self.hp_penalty

        # --------------------------------------------------
        # KO / FAINT
        # --------------------------------------------------
        my_fainted = sum(p.fainted for p in battle.team.values())
        opp_fainted = sum(p.fainted for p in battle.opponent_team.values())

        if opp_fainted > self.prev_opp_fainted:
            reward += self.ko_reward
        if my_fainted > self.prev_my_fainted:
            reward -= self.faint_penalty

        # --------------------------------------------------
        # STATUS
        # --------------------------------------------------
        my_status = sum(1 for p in battle.team.values() if p.status)
        opp_status = sum(1 for p in battle.opponent_team.values() if p.status)

        if opp_status > self.prev_opp_status:
            reward += self.status_reward
        if my_status > self.prev_my_status:
            reward -= self.status_penalty

        # --------------------------------------------------
        # BOOSTS
        # --------------------------------------------------
        # BOOSTS (all major stats)
        boost_stats = ("atk", "def", "spa", "spd", "spe")

        for stat in boost_stats:
            # My boosts gained → reward
            if my.boosts.get(stat, 0) > self.prev_my_boosts.get(stat, 0):
                reward += self.boost_reward

            # Opponent boosts gained → penalty
            if opp.boosts.get(stat, 0) > self.prev_opp_boosts.get(stat, 0):
                reward -= self.boost_penalty


        # Suicide boosting penalty
        if my_fainted > self.prev_my_fainted and last_used_move and last_used_move.category.name == "STATUS":
            reward -= self.suicide_boost_penalty

        # --------------------------------------------------
        # SWITCH EVALUATION
        # --------------------------------------------------
        forced_switch = (my_fainted > self.prev_my_fainted)
        new_matchup = self.matchup_score(my, opp)

        if action_was_switch and not forced_switch:
            # Base cost
            reward -= self.base_switch_penalty

            # Matchup improvement
            delta = new_matchup - self.prev_matchup_score
            if delta > 0.1:
                reward += self.matchup_reward
            elif delta < -0.1:
                reward -= self.matchup_penalty
                # --- NEW: Punish switching out when you have boosts ---
            boost_stats = ("atk", "def", "spa", "spd", "spe")

            # Count only positive boosts (switching wastes them)
            total_positive_boosts = sum(
                max(0, my.boosts.get(stat, 0)) for stat in boost_stats
            )

            if total_positive_boosts > 0:
                # Scaled penalty: each boost level increases punishment
                reward -= self.boost_penalty * total_positive_boosts
            # --- NEW: Reward switching out when significantly debuffed ---
            debuff_stats = ("atk", "def", "spa", "spd", "spe")

            total_debuff = sum(
                abs(min(0, my.boosts.get(stat, 0)))
                for stat in debuff_stats
            )

            if total_debuff > 0:
                reward += self.matchup_reward * total_debuff


            # Speed advantage bonus
            my_spe = get_real_stats(my.species)["spe"]
            opp_spe = get_real_stats(opp.species)["spe"]

            if my_spe > opp_spe:
                reward += self.switch_speed_bonus

            # Switch chain
            self.consecutive_switches += 1
            self.recent_switches.append(1)

            if self.consecutive_switches >= 2:
                reward -= self.switch_chain_penalty * (self.consecutive_switches - 1)

            # Spam
            if len(self.recent_switches) > self.switch_window:
                self.recent_switches.pop(0)
            if sum(self.recent_switches) >= 3:
                reward -= self.switch_spam_penalty

            # Oscillation (A → B → A)
            if self.last_switch_species == my.species:
                reward -= self.oscillation_penalty

            self.last_switch_species = self.last_active_species

        elif forced_switch:
            # Reset chains, never penalize
            self.consecutive_switches = 0
            self.recent_switches.append(0)

            if new_matchup > self.prev_matchup_score:
                reward += self.matchup_reward * 0.5

        else:
            self.consecutive_switches = 0
            self.recent_switches.append(0)

        # --------------------------------------------------
        # PIVOT MOVE REWARD
        # --------------------------------------------------
        if last_used_move and last_used_move.id.lower() in PIVOT_MOVES:
            reward += self.pivot_reward

        # --------------------------------------------------
        # TURN PENALTY
        # --------------------------------------------------
        reward -= self.turn_penalty

        # --------------------------------------------------
        # UPDATE INTERNAL STATE
        # --------------------------------------------------
        self.prev_my_bucket = my_b
        self.prev_opp_bucket = opp_b
        self.prev_my_fainted = my_fainted
        self.prev_opp_fainted = opp_fainted

        self.prev_my_status = my_status
        self.prev_opp_status = opp_status

        self.prev_my_boosts = my.boosts.copy()
        self.prev_opp_boosts = opp.boosts.copy()

        self.prev_matchup_score = new_matchup
        self.last_active_species = my.species

        return reward
