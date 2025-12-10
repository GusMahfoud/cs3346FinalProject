# advanced_switcher.py — Predictive Switching with Shared Damage Model

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.pokemon import Pokemon

from computed_stats import get_real_stats
from move_pool_loader import get_species_moves
from damage_model import estimate_damage   # (frac, raw, sig1, sig2)


class SwitchHeuristics:
    # ================================================================
    # CONFIG CONSTANTS
    # ================================================================
    DEBUG = False

    # How much better a switch must be than staying to trigger a switch
    SWITCH_OUT_MATCHUP_THRESHOLD = 1.8

    # How hard to lean on matchup in stay vs switch scoring
    MATCHUP_STAY_WEIGHT = 1.0
    MATCHUP_SWITCH_WEIGHT = 1.3

    # Base components for matchup score
    SPEED_COEFF = 1.6
    HP_COEFF = 0.4
    OFFENSE_COEFF = 4.0   # self_best_frac - opp_best_frac

    # Bonus / penalty for lethal situations, etc.
    HOPLESS_STAY_PENALTY = 6.0   # staying when you do nothing and die
    LETHAL_ADVANTAGE_BONUS = 4.0 # when you OHKO and they can’t

    BOOST_RETENTION_PENALTY = 0.7
    DEBUFF_ESCAPE_REWARD = 1.2

    DANGER_WEIGHT = 9
    THREAT_WEIGHT = 6.0

    OHKO_BONUS_FAST = 8.0
    OHKO_BONUS_SLOW = 5.0

    VOLUNTARY_DEATH_PENALTY = -12.0
    FORCED_DEATH_PENALTY = -8.0
    SAFE_SWITCH_BONUS = 2.5

    IMMEDIATE_TURN_WEIGHT = 6.0

    # thresholds for “hopeless” offense and “lethal” risk
    MIN_USEFUL_FRACTION = 0.25   # below this = 4HKO or worse
    LETHAL_INCOMING_FRACTION = 0.9

    # ================================================================
    # BASIC HELPERS
    # ================================================================
    def _stage_mult(self, stage: int):
        table = [
            2/8, 2/7, 2/6, 2/5, 2/4, 2/3,
            1.0,
            3/2, 2.0, 5/2, 3.0, 7/2, 4.0
        ]
        return table[stage + 6]

    def _package_stats(self, mon: Pokemon):
        base = get_real_stats(mon.species)
        boosts = mon.boosts
        return {
            "atk": base["atk"] * self._stage_mult(boosts.get("atk", 0)),
            "def": base["def"] * self._stage_mult(boosts.get("def", 0)),
            "spa": base["spa"] * self._stage_mult(boosts.get("spa", 0)),
            "spd": base["spd"] * self._stage_mult(boosts.get("spd", 0)),
            "spe": base["spe"] * self._stage_mult(boosts.get("spe", 0)),
            "hp": mon.current_hp or 1,
            "max_hp": mon.max_hp or 1,
        }

    def _speed_bucket(self, stats_dict):
        spe = stats_dict["spe"]
        if spe < 140:
            return 0     # very slow (Gambit tier)
        if spe < 200:
            return 1     # slow
        if spe < 280:
            return 2     # mid
        if spe < 350:
            return 3     # fast
        return 4         # extremely fast (Dragapult / Weavile tier)

    def _get_moves_from_battle(self, species: str, battle):
        if battle is None or getattr(battle, "team", None) is None:
            return []
        for mon in battle.team.values():
            if mon and mon.species.lower() == species.lower():
                return list(mon.moves.values())
        return []

    # ================================================================
    # DAMAGE HELPERS (ONLY USE FRACTION)
    # ================================================================
    def _best_frac(self, attacker, defender, moves):
        """Return (best_fraction, is_lethal) for attacker using this move list."""
        best = 0.0
        lethal = False

        for mv in moves:
            frac, _, _, _ = estimate_damage(mv, attacker, defender)
            if frac is None:
                continue
            frac = max(0.0, min(1.0, frac))
            best = max(best, frac)
            if frac >= 1.0:
                lethal = True

        return best, lethal

    def _predict_opponent_move(self, opp: Pokemon, me: Pokemon):
        """Use species-default move pool ONLY."""
        moves = get_species_moves(opp.species)
        if not moves:
            if self.DEBUG:
                print(f"[PREDICT] No species-default moves for {opp.species}")
            return None, 0.0

        best_mv = None
        best_frac = -1.0
        for mv in moves:
            frac, _, _, _ = estimate_damage(mv, opp, me)
            frac = max(0.0, min(1.0, frac or 0.0))

            if self.DEBUG:
                try:
                    mid = mv.id
                except AttributeError:
                    mid = getattr(mv, "name", "<?>")
                print(f"   - {mid}: frac={frac:.3f}")

            if frac > best_frac:
                best_frac = frac
                best_mv = mv

        if self.DEBUG:
            print(f"[PREDICT] {opp.species} → {best_mv} (frac={best_frac:.3f}) vs {me.species}")

        return best_mv, best_frac

    def _danger(self, opp, me):
        moves = get_species_moves(opp.species)
        if not moves:
            return 0.0
        best_frac, lethal = self._best_frac(opp, me, moves)
        return self.DANGER_WEIGHT if lethal else best_frac * self.DANGER_WEIGHT

    def _threat(self, me, opp, battle):
        moves = self._get_moves_from_battle(me.species, battle)
        if not moves:
            return 0.0
        best_frac, lethal = self._best_frac(me, opp, moves)
        return self.THREAT_WEIGHT if lethal else best_frac * self.THREAT_WEIGHT

    # ================================================================
    # MATCHUP SCORE
    # ================================================================
    def estimate_matchup(self, me, opp, battle=None):
        """
        Positive = favorable matchup for me.
        Negative = bad matchup (e.g., Weavile into Kingambit).
        """
        if me is None or opp is None:
            return 0.0

        me_stats = self._package_stats(me)
        opp_stats = self._package_stats(opp)

        me_hp_frac = me_stats["hp"] / me_stats["max_hp"]
        opp_hp_frac = opp_stats["hp"] / opp_stats["max_hp"]

        # Speed term
        faster = me_stats["spe"] > opp_stats["spe"]
        slower = me_stats["spe"] < opp_stats["spe"]
        if faster:
            speed_term = self.SPEED_COEFF
        elif slower:
            speed_term = -self.SPEED_COEFF
        else:
            speed_term = 0.0

        # HP/bulk term
        hp_term = (me_hp_frac - opp_hp_frac) * self.HP_COEFF

        # Offense vs danger: my best damage vs their best damage
        my_moves = self._get_moves_from_battle(me.species, battle)
        opp_moves = get_species_moves(opp.species)

        my_best_frac, my_lethal = self._best_frac(me, opp, my_moves) if my_moves else (0.0, False)
        opp_best_frac, opp_lethal = self._best_frac(opp, me, opp_moves) if opp_moves else (0.0, False)

        offense_term = (my_best_frac - opp_best_frac) * self.OFFENSE_COEFF

        # Threat / danger global scale (same as before but explicit)
        threat_term = self._threat(me, opp, battle)
        danger_term = self._danger(opp, me)

        score = speed_term + hp_term + offense_term + threat_term - danger_term

        # Boosts: still keep your original idea
        for val in me.boosts.values():
            if val > 0:
                score += val * self.BOOST_RETENTION_PENALTY
            elif val < 0:
                score += abs(val) * self.DEBUFF_ESCAPE_REWARD

        

        # Hopeless matchup: I do nothing, they delete me
        if my_best_frac < self.MIN_USEFUL_FRACTION and opp_best_frac >= self.LETHAL_INCOMING_FRACTION:
            score -= self.HOPLESS_STAY_PENALTY

        # Lethal advantage: I OHKO and they don't
        if my_lethal and not opp_lethal:
            score += self.LETHAL_ADVANTAGE_BONUS

        return score

    # ================================================================
    # STAY EVALUATION
    # ================================================================
    def _evaluate_stay(self, me, opp, battle):
        pred_mv, _ = self._predict_opponent_move(opp, me)
        me_stats = self._package_stats(me)
        me_hp_frac = me_stats["hp"] / me_stats["max_hp"]
        if pred_mv:
            incoming_frac, _, incoming_lethal, _ = estimate_damage(pred_mv, opp, me)
        else:
            incoming_frac = 0.0
            incoming_lethal = False

        moves_me = self._get_moves_from_battle(me.species, battle)
        best_out_frac = 0.0
        outgoing_lethal = False

        for mv in moves_me:
            frac, _, leth, _ = estimate_damage(mv, me, opp)
            frac = max(0.0, min(1.0, frac or 0.0))
            best_out_frac = max(best_out_frac, frac)
            if frac >= 1.0:
                outgoing_lethal = True

        me_stats = self._package_stats(me)
        opp_stats = self._package_stats(opp)
        faster = me_stats["spe"] > opp_stats["spe"]

        immediate = 0.0
        if faster:
            if outgoing_lethal:
                # I OHKO before they move → huge incentive to stay
                immediate += self.OHKO_BONUS_FAST * 5
            else:
                immediate += (best_out_frac - incoming_frac) * self.IMMEDIATE_TURN_WEIGHT
        else:
            if incoming_lethal:
                # Slower and I die → strong penalty for staying
                immediate += self.VOLUNTARY_DEATH_PENALTY
            else:
                immediate += (best_out_frac - incoming_frac) * self.IMMEDIATE_TURN_WEIGHT
        # Low-HP preservation for FAST mons (use speed buckets)
        my_bucket = self._speed_bucket(me_stats)
        if me_hp_frac < 0.25 and my_bucket <= 1:
            immediate -= 2
        matchup = self.estimate_matchup(me, opp, battle)
        return immediate + self.MATCHUP_STAY_WEIGHT * matchup

    # ================================================================
    # SWITCH TARGET EVALUATION
    # ================================================================
    def _evaluate_switch_target(self, cand, opp, me, battle, voluntary):
        pred_mv, _ = self._predict_opponent_move(opp, me)
       
        if pred_mv:
            dmg_me, _, _, _ = estimate_damage(pred_mv, opp, me)
            dmg_cand, _, lethal_cand, _ = estimate_damage(pred_mv, opp, cand)
        else:
            dmg_me = 0.0
            dmg_cand = 0.0
            lethal_cand = False

        # Immediate benefit: how much damage you "save" by switching
        if voluntary:
            immediate = (dmg_me - dmg_cand) * self.IMMEDIATE_TURN_WEIGHT
        else:
            immediate = self.SAFE_SWITCH_BONUS

        if lethal_cand:
            # Don’t pivot into something that just dies
            immediate += self.VOLUNTARY_DEATH_PENALTY * 3 if voluntary else self.FORCED_DEATH_PENALTY

        post = self.estimate_matchup(cand, opp, battle) * self.MATCHUP_SWITCH_WEIGHT

        # Extra juice if candidate can immediately threaten a big hit
        moves = self._get_moves_from_battle(cand.species, battle)
        best_frac = 0.0
        lethal = False
        for mv in moves:
            frac, _, leth_flag, _ = estimate_damage(mv, cand, opp)
            frac = max(0.0, min(1.0, frac or 0.0))
            best_frac = max(best_frac, frac)
            if frac >= 1.0:
                lethal = True

        cand_stats = self._package_stats(cand)
        opp_stats = self._package_stats(opp)
        faster = cand_stats["spe"] > opp_stats["spe"]

        if lethal:
            post += self.OHKO_BONUS_FAST if faster else self.OHKO_BONUS_SLOW
        elif best_frac >= 0.5 and faster:
            post += 2.0

        return immediate + post

    # ================================================================
    # DECISION + TARGET
    # ================================================================
    def should_switch_out(self, battle):
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        if me is None or opp is None:
            return False
        if not battle.available_switches:
            return False

        stay = self._evaluate_stay(me, opp, battle)

        best_switch = max(
            (self._evaluate_switch_target(mon, opp, me, battle, True)
             for mon in battle.available_switches),
            default=None
        )

        if best_switch is None:
            return False

        return best_switch - stay >= self.SWITCH_OUT_MATCHUP_THRESHOLD

    def best_switch_target(self, battle, voluntary=True):
        if not battle.available_switches:
            return None

        scored = [
            (
                self._evaluate_switch_target(
                    mon,
                    battle.opponent_active_pokemon,
                    battle.active_pokemon,
                    battle,
                    voluntary,
                ),
                mon,
            )
            for mon in battle.available_switches
        ]

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]