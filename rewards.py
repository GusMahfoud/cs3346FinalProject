import numpy as np
from poke_env.battle.battle import Battle
from poke_env.data import GenData

# Load the correct Gen 9 type chart
GEN9_DATA = GenData.from_gen(9)
TYPE_CHART = GEN9_DATA.type_chart


# ----------------------------------------------------------
# Utility bucket
# ----------------------------------------------------------
def hp_bucket(frac):
    if frac <= 0.25: return 0
    if frac <= 0.50: return 1
    if frac <= 0.75: return 2
    return 3


PIVOT_MOVES = {
    "uturn", "voltturn", "voltswitch", "flipturn", "partingshot", "batonpass"
}


class RewardCalculator:
    """
    Smart, switch-aware reward engine engineered for your fixed mini-meta.

    Captures:
    - HP trades
    - KO / faint value
    - Status usage
    - Boost progression
    - Switch misuse penalties (spam, chain, no-benefit)
    - Good switching rewards (matchup improvement)
    - Pivot move bonus (U-turn, Volt Switch, Flip Turn)
    - Turn-time pressure
    """

    def __init__(
        self,
        # terminal
        terminal_win=100,
        terminal_loss=-100,
        terminal_draw=0,

        # hp buckets
        hp_bucket_reward=4,
        hp_bucket_penalty=4,

        # ko/faint
        ko_reward=30,
        faint_penalty=25,

        # status
        status_reward=4,
        status_penalty=4,

        # boosts
        boost_reward=6,
        boost_penalty=6,

        # switching
        base_switch_penalty=0.5,
        switch_chain_penalty=1.5,
        switch_spam_penalty=2.0,
        switch_window=6,

        # matchup switching
        matchup_improve_reward=4.0,
        bad_switch_penalty=4.0,

        # pivot moves
        pivot_reward=3.0,

        # pace
        turn_penalty=0.05,
    ):
        self.terminal_win = terminal_win
        self.terminal_loss = terminal_loss
        self.terminal_draw = terminal_draw

        self.hp_bucket_reward = hp_bucket_reward
        self.hp_bucket_penalty = hp_bucket_penalty

        self.ko_reward = ko_reward
        self.faint_penalty = faint_penalty

        self.status_reward = status_reward
        self.status_penalty = status_penalty

        self.boost_reward = boost_reward
        self.boost_penalty = boost_penalty

        self.base_switch_penalty = base_switch_penalty
        self.switch_chain_penalty = switch_chain_penalty
        self.switch_spam_penalty = switch_spam_penalty
        self.switch_window = switch_window

        self.matchup_improve_reward = matchup_improve_reward
        self.bad_switch_penalty = bad_switch_penalty
        self.pivot_reward = pivot_reward

        self.turn_penalty = turn_penalty

        self.reset()


    # ----------------------------------------------------------
    def reset(self):
        self.prev_my_bucket = None
        self.prev_opp_bucket = None

        self.prev_my_fainted = 0
        self.prev_opp_fainted = 0

        self.prev_my_status = 0
        self.prev_opp_status = 0

        self.prev_my_boosts = {"atk": 0, "spa": 0}
        self.prev_opp_boosts = {"atk": 0, "spa": 0}

        # switching traces
        self.consecutive_switches = 0
        self.recent_switches = []

        # matchup
        self.prev_matchup_score = None


    # ----------------------------------------------------------
    # Type matchup score using Gen 9 type chart
    # ----------------------------------------------------------
    def matchup_score(self, my, opp):
        my_types = my.types or []
        opp_types = opp.types or []

        my_eff = 1.0
        opp_eff = 1.0

        # my → opp effectiveness
        for t in my_types:
            for ot in opp_types:
                if t in TYPE_CHART and ot in TYPE_CHART[t]:
                    my_eff *= TYPE_CHART[t][ot]

        # opp → my effectiveness
        for ot in opp_types:
            for t in my_types:
                if ot in TYPE_CHART and t in TYPE_CHART[ot]:
                    opp_eff *= TYPE_CHART[ot][t]

        return my_eff - opp_eff


    # ----------------------------------------------------------
    def compute_terminal_reward(self, battle):
        if battle.won: return self.terminal_win
        if battle.lost: return self.terminal_loss
        return self.terminal_draw


    # ----------------------------------------------------------
    def compute_turn_reward(
        self,
        prev_battle,
        battle,
        action_was_switch=False,
        action_was_priority=False,
        last_used_move=None,
    ):
        my = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        if my is None or opp is None:
            return 0.0

        reward = 0.0

        # ------------------------------------------------------
        # FIRST TURN INIT
        # ------------------------------------------------------
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
            return 0.0

        # ------------------------------------------------------
        # 1. HP BUCKETS
        # ------------------------------------------------------
        my_b = hp_bucket(my.current_hp_fraction)
        opp_b = hp_bucket(opp.current_hp_fraction)

        if opp_b < self.prev_opp_bucket:
            reward += self.hp_bucket_reward
        if my_b < self.prev_my_bucket:
            reward -= self.hp_bucket_penalty

        # ------------------------------------------------------
        # 2. KOs / FAINTS
        # ------------------------------------------------------
        my_fainted = sum(p.fainted for p in battle.team.values())
        opp_fainted = sum(p.fainted for p in battle.opponent_team.values())

        if opp_fainted > self.prev_opp_fainted:
            reward += self.ko_reward
        if my_fainted > self.prev_my_fainted:
            reward -= self.faint_penalty

        # ------------------------------------------------------
        # 3. STATUS EFFECTS
        # ------------------------------------------------------
        my_status = sum(1 for p in battle.team.values() if p.status)
        opp_status = sum(1 for p in battle.opponent_team.values() if p.status)

        if opp_status > self.prev_opp_status:
            reward += self.status_reward
        if my_status > self.prev_my_status:
            reward -= self.status_penalty

        # ------------------------------------------------------
        # 4. BOOSTS (ATK/SPA only)
        # ------------------------------------------------------
        for stat in ("atk", "spa"):
            if my.boosts.get(stat, 0) > self.prev_my_boosts.get(stat, 0):
                reward += self.boost_reward
            if opp.boosts.get(stat, 0) > self.prev_opp_boosts.get(stat, 0):
                reward -= self.boost_penalty

        # ------------------------------------------------------
        # 5. MATCHUP-BASED SWITCH LOGIC
        # ------------------------------------------------------
        new_matchup = self.matchup_score(my, opp)

        if action_was_switch:
            if new_matchup > self.prev_matchup_score:
                reward += self.matchup_improve_reward
            elif new_matchup < self.prev_matchup_score:
                reward -= self.bad_switch_penalty

        # ------------------------------------------------------
        # 6. PIVOT MOVE (U-turn etc)
        # ------------------------------------------------------
        if last_used_move is not None:
            move_id = last_used_move.id.lower()
            if move_id in PIVOT_MOVES:
                reward += self.pivot_reward

        # ------------------------------------------------------
        # 7. SWITCH ABUSE PENALTIES
        # ------------------------------------------------------
        if action_was_switch:
            reward -= self.base_switch_penalty
            self.consecutive_switches += 1
            self.recent_switches.append(1)
        else:
            self.consecutive_switches = 0
            self.recent_switches.append(0)

        # maintain sliding window
        if len(self.recent_switches) > self.switch_window:
            self.recent_switches.pop(0)

        # chain penalty: switching 2+ turns in a row
        if self.consecutive_switches >= 2:
            reward -= self.switch_chain_penalty * (self.consecutive_switches - 1)

        # spam penalty: 3 switches in last N turns
        if sum(self.recent_switches) >= 3:
            reward -= self.switch_spam_penalty

        # ------------------------------------------------------
        # 8. TURN PRESSURE
        # ------------------------------------------------------
        reward -= self.turn_penalty

        # ------------------------------------------------------
        # TRACK STATE
        # ------------------------------------------------------
        self.prev_my_bucket = my_b
        self.prev_opp_bucket = opp_b
        self.prev_my_fainted = my_fainted
        self.prev_opp_fainted = opp_fainted
        self.prev_my_status = my_status
        self.prev_opp_status = opp_status
        self.prev_my_boosts = my.boosts.copy()
        self.prev_opp_boosts = opp.boosts.copy()
        self.prev_matchup_score = new_matchup

        return reward
