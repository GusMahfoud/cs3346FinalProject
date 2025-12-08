# rewards.py  (FULLY PATCHED WITH DIMINISHING BOOST REWARDS)

import numpy as np
from poke_env.data import GenData
from computed_stats import get_real_stats
from advanced_switcher import SwitchHeuristics

GEN9_DATA = GenData.from_gen(9)
TYPE_CHART = GEN9_DATA.type_chart

PIVOT_MOVES = {"uturn", "voltswitch", "flipturn", "partingshot", "batonpass"}


# ------------------------------------------------------------
# HP BUCKET
# ------------------------------------------------------------
def hp_bucket(frac):
    if frac <= 0.25: return 0
    if frac <= 0.50: return 1
    if frac <= 0.75: return 2
    return 3


# ------------------------------------------------------------
# DAMAGE-STAGE MULTIPLIERS (actual Pokémon mechanics)
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
    """
    Diminishing returns curve for boosting:
    +1 → moderate, +2 → strong, +3-6 → flattening
    """
    if stage <= 0:
        return 0
    mult = stage_multiplier(stage)
    return base * np.log2(mult)   # log scaling = diminishing returns


# ------------------------------------------------------------
# BOOSTING MOVE CATEGORIES
# ------------------------------------------------------------
BOOSTING_MOVES = {"swordsdance", "nastyplot", "quiverdance", "calmmind", "torchsong"}
UTILITY_MOVES = {"willowisp", "recover", "slackoff"}


# ------------------------------------------------------------
# REWARD CALCULATOR
# ------------------------------------------------------------
class RewardCalculator:

    def __init__(
        self,
        terminal_win=100,
        terminal_loss=-120,
        terminal_draw=0,

        hp_reward=4,
        hp_penalty=6,

        ko_reward=30,
        faint_penalty=25,

        status_reward=4,
        status_penalty=6,

        boost_base=8,   # controls how strong boosts are globally

        suicide_boost_penalty=15,

        switch_weight=5.0,
        wasted_boost_penalty=4.0,
        debuff_escape_reward=4.0,
        oscillation_penalty=5.0,
        switch_chain_penalty=1.5,
        switch_spam_penalty=2.0,
        switch_window=5,

        pivot_reward=3.0,
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

        self.boost_base = boost_base
        self.suicide_boost_penalty = suicide_boost_penalty

        self.switch_weight = switch_weight
        self.wasted_boost_penalty = wasted_boost_penalty
        self.debuff_escape_reward = debuff_escape_reward
        self.oscillation_penalty = oscillation_penalty
        self.switch_chain_penalty = switch_chain_penalty
        self.switch_spam_penalty = switch_spam_penalty
        self.switch_window = switch_window

        self.pivot_reward = pivot_reward
        self.turn_penalty = turn_penalty

        self.heuristic = SwitchHeuristics()

        self.reset()


    # ============================================================
    # RESET
    # ============================================================
    def reset(self):
        self.prev_my_bucket = None
        self.prev_opp_bucket = None

        self.prev_my_fainted = 0
        self.prev_opp_fainted = 0

        self.prev_my_status = 0
        self.prev_opp_status = 0

        self.prev_my_boosts = {"atk":0,"def":0,"spa":0,"spd":0,"spe":0}
        self.prev_opp_boosts = {"atk":0,"def":0,"spa":0,"spd":0,"spe":0}

        self.prev_matchup_score = None
        self.last_switch_species = None
        self.last_active_species = None

        self.consecutive_switches = 0
        self.recent_switches = []

        self.turn_reward = 0.0


    def add(self, v): self.turn_reward += v
    def flush(self): r = self.turn_reward; self.turn_reward = 0; return r


    # ============================================================
    # SWITCH MATCHUP SCORE
    # ============================================================
    def compute_switch_heuristic_score(self, battle):
        a = battle.active_pokemon
        o = battle.opponent_active_pokemon
        if a is None or o is None:
            return 0.0
        return self.heuristic.estimate_matchup(a, o)


    # ============================================================
    # TERMINAL REWARD
    # ============================================================
    def compute_terminal_reward(self, battle):
        if battle.won: return self.terminal_win
        if battle.lost: return self.terminal_loss
        return self.terminal_draw


    # ============================================================
    # TURN REWARD
    # ============================================================
    def compute_turn_reward(self, prev_battle, battle, action_was_switch=False, last_used_move=None):
        
        my = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        if my is None or opp is None:
            return 0.0

        reward = 0.0

        # ------------------------------------------------------------
        # FIRST TURN INIT
        # ------------------------------------------------------------
        if self.prev_my_bucket is None:

            self.prev_my_bucket = hp_bucket(my.current_hp_fraction)
            self.prev_opp_bucket = hp_bucket(opp.current_hp_fraction)

            self.prev_my_fainted = sum(p.fainted for p in battle.team.values())
            self.prev_opp_fainted = sum(p.fainted for p in battle.opponent_team.values())

            self.prev_my_status = sum(1 for p in battle.team.values() if p.status)
            self.prev_opp_status = sum(1 for p in battle.opponent_team.values() if p.status)

            self.prev_my_boosts = my.boosts.copy()
            self.prev_opp_boosts = opp.boosts.copy()

            self.prev_matchup_score = self.compute_switch_heuristic_score(battle)
            self.last_active_species = my.species

            return 0.0

        # ============================================================
        # 1. HP CHANGE REWARDS
        # ============================================================
        my_b = hp_bucket(my.current_hp_fraction)
        opp_b = hp_bucket(opp.current_hp_fraction)

        if opp_b < self.prev_opp_bucket:
            reward += self.hp_reward

        if my_b < self.prev_my_bucket:
            reward -= self.hp_penalty

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
        # 3. STATUS EFFECTS
        # ============================================================
        my_status = sum(1 for p in battle.team.values() if p.status)
        opp_status = sum(1 for p in battle.opponent_team.values() if p.status)

        if opp_status > self.prev_opp_status:
            reward += self.status_reward

        if my_status > self.prev_my_status:
            reward -= self.status_penalty

        # ============================================================
        # 4. DIMINISHING BOOST REWARDS (MAIN FIX)
        # ============================================================
        for stat in ("atk", "spa"):
            old = self.prev_my_boosts.get(stat, 0)
            new = my.boosts.get(stat, 0)
            if new > old:
                reward += diminishing_boost_value(new, self.boost_base) - diminishing_boost_value(old, self.boost_base)

        # Defense boosts (half value)
        for stat in ("def", "spd"):
            old = self.prev_my_boosts.get(stat, 0)
            new = my.boosts.get(stat, 0)
            if new > old:
                reward += 0.5 * (diminishing_boost_value(new, self.boost_base) - diminishing_boost_value(old, self.boost_base))

        # Speed boosts (binary usefulness)
        old = self.prev_my_boosts.get("spe", 0)
        new = my.boosts.get("spe", 0)
        if new > old:
            reward += 6   # outspeed bonus
            if new > 1:
                reward += (new - 1)  # diminishing extra

        # ============================================================
        # 5. SUICIDE BOOSTING FIXED
        # ============================================================
        if my_fainted > self.prev_my_fainted and last_used_move:
            move_id = last_used_move.id.lower()

            # Punish wasted boosting only
            if move_id in BOOSTING_MOVES:
                # "Could have acted"
                my_speed = get_real_stats(my.species)["spe"]
                opp_speed = get_real_stats(opp.species)["spe"]

                if my_speed >= opp_speed:
                    reward -= self.suicide_boost_penalty

            # Utility moves -> never punish
            elif move_id in UTILITY_MOVES:
                pass

        # ============================================================
        # 6. USELESS / IMMUNE MOVE PENALTY
        # ============================================================
        if last_used_move:
            imm = False

            try:
                eff = last_used_move.type.damage_multiplier(opp.type_1, opp.type_2)
                if eff == 0:
                    imm = True
            except:
                pass

            # Dragon vs Fairy immunity
            if last_used_move.type and last_used_move.type.name == "Dragon":
                if (opp.type_1 and opp.type_1.name == "Fairy") or \
                   (opp.type_2 and opp.type_2.name == "Fairy"):
                    imm = True

            # Will-O-Wisp vs Fire or statused target
            if last_used_move.id == "willowisp":
                if (opp.type_1 and opp.type_1.name == "Fire") or \
                   (opp.type_2 and opp.type_2.name == "Fire") or opp.status:
                    imm = True

            if imm:
                reward -= 8.0

        # ============================================================
        # 7. SWITCH REWARDS (use heuristic delta)
        # ============================================================
        forced_switch = (my_fainted > self.prev_my_fainted)

        if action_was_switch and not forced_switch:

            new_score = self.compute_switch_heuristic_score(battle)
            delta = new_score - self.prev_matchup_score

            reward += delta * self.switch_weight

            # waste boosts
            total_pos = sum(max(0, my.boosts.get(s, 0)) for s in ("atk","def","spa","spd","spe"))
            reward -= self.wasted_boost_penalty * total_pos

            # escape debuffs
            total_neg = sum(abs(min(0, my.boosts.get(s, 0))) for s in ("atk","def","spa","spd","spe"))
            reward += self.debuff_escape_reward * total_neg

            # switch chain penalties
            self.consecutive_switches += 1
            self.recent_switches.append(1)

            if self.consecutive_switches > 1:
                reward -= self.switch_chain_penalty * (self.consecutive_switches - 1)

            # spam
            if len(self.recent_switches) > self.switch_window:
                self.recent_switches.pop(0)
            if sum(self.recent_switches) >= 3:
                reward -= self.switch_spam_penalty

            # oscillation
            if self.last_switch_species == my.species:
                reward -= self.oscillation_penalty

            self.last_switch_species = self.last_active_species

        elif forced_switch:
            self.consecutive_switches = 0
            self.recent_switches.append(0)

        else:
            self.consecutive_switches = 0
            self.recent_switches.append(0)

        # ============================================================
        # 8. PIVOT MOVES
        # ============================================================
        if last_used_move and last_used_move.id.lower() in PIVOT_MOVES:
            reward += self.pivot_reward

        # ============================================================
        # 9. TURN TICK
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
        self.last_active_species = my.species

        self.add(reward)
        return 0.0