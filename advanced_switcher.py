# advanced_switcher.py — Predictive Switching with Shared Damage Model
# Uses real moves for our side, species-default move pools for opponent.

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.pokemon import Pokemon

from computed_stats import get_real_stats
from move_pool_loader import get_species_moves
from damage_model import estimate_damage   # (frac, raw, sig1, sig2)


class SwitchHeuristics:

    # ================================================================
    # CONFIG CONSTANTS — YOUR ORIGINAL VALUES
    # ================================================================
    DEBUG = True   # set True if you want printouts, otherwise stays silent
    SWITCH_OUT_MATCHUP_THRESHOLD = 3

    TYPE_COEFF = 1.0
    SPEED_COEFF = 1.6
    HP_COEFF = 0.25
    STAB_BONUS = 1.3

    BOOST_RETENTION_PENALTY = 0.7
    DEBUFF_ESCAPE_REWARD = 1.2

    DANGER_WEIGHT = 8
    THREAT_WEIGHT = 6.0

    OHKO_BONUS_FAST = 8.0
    OHKO_BONUS_SLOW = 5.0

    VOLUNTARY_DEATH_PENALTY = -12.0
    FORCED_DEATH_PENALTY = -8.0
    SAFE_SWITCH_BONUS = 2.5

    IMMEDIATE_TURN_WEIGHT = 4.0

    def _computed_stat(self, mon: Pokemon, stat: str):
        """
        Returns boosted real stat (atk/def/spa/spd/spe) using get_real_stats.
        """
        base = get_real_stats(mon.species)[stat]
        boost = mon.boosts.get(stat, 0)

        stage_mult = [
            2/8, 2/7, 2/6, 2/5, 2/4, 2/3,
            1.0,
            3/2, 2.0, 5/2, 3.0, 7/2, 4.0
        ]
        mult = stage_mult[boost + 6]
        return base * mult
    # ================================================================
    # GET REAL MOVES
    # ================================================================
    def _get_moves_from_battle(self, species: str, battle):
        if battle is None or getattr(battle, "team", None) is None:
            return []
        for mon in battle.team.values():
            if mon and mon.species.lower() == species.lower():
                return list(mon.moves.values())
        return []


    # ================================================================
    # BOOSTED STAT PACKAGE
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


    # ================================================================
    # MAX DAMAGE (patched to use frac ONLY)
    # ================================================================
    def _max_damage_frac(self, attacker, defender, moves):
        best = 0.0
        lethal = False

        for mv in moves:
            frac, _, _, _ = estimate_damage(mv, attacker, defender)

            # ensure valid range
            if frac is None:
                continue
            if frac < 0:
                frac = 0.0
            if frac > 1:
                frac = 1.0

            best = max(best, frac)

            if frac >= 1.0:
                lethal = True

        return best, lethal


    # ================================================================
    # PREDICT OPP MOVE (frac only)
    # ================================================================
    def _predict_opponent_move(self, opp: Pokemon, me: Pokemon):
        """
        Predict opponent move *always* from species-default moves.
        Never use revealed moves, never read opp.moves.
        """
        moves = get_species_moves(opp.species)
        
        if not moves:
            if self.DEBUG:
                print(f"[PREDICT] No species-default moves found for {opp.species}")
            return None, 0.0

        best_mv = None
        best_frac = -1.0

        if self.DEBUG:
            print(f"\n[PREDICT] Evaluating species-default moves for {opp.species} vs {me.species}")

        for mv in moves:
            frac, _, _, _ = estimate_damage(mv, opp, me)
            frac = max(0.0, min(1.0, frac or 0.0))

            if self.DEBUG:
                print(f"   - {mv.id}: frac={frac:.3f}")

            if frac > best_frac:
                best_frac = frac
                best_mv = mv

        if self.DEBUG:
            print(f"[PREDICT] Selected {best_mv.id} with predicted frac={best_frac:.3f}")

        return best_mv, best_frac
    # ================================================================
    # DANGER (frac only)
    # ================================================================
    def _danger(self, opp, me):
        moves = get_species_moves(opp.species)
        if not moves:
            return 0.0

        worst_frac = 0.0
        lethal = False

        for mv in moves:
            frac, _, _, _ = estimate_damage(mv, opp, me)
            if frac < 0: frac = 0.0
            if frac > 1: frac = 1.0

            worst_frac = max(worst_frac, frac)
            if frac >= 1.0:
                lethal = True

        return self.DANGER_WEIGHT if lethal else worst_frac * self.DANGER_WEIGHT


    # ================================================================
    # THREAT (frac only)
    # ================================================================
    def _threat(self, me, opp, battle):
        moves = self._get_moves_from_battle(me.species, battle)
        if not moves:
            return 0.0

        best_frac = 0.0
        lethal = False

        for mv in moves:
            frac, _, _, _ = estimate_damage(mv, me, opp)
            if frac < 0: frac = 0.0
            if frac > 1: frac = 1.0

            best_frac = max(best_frac, frac)
            if frac >= 1.0:
                lethal = True

        return self.THREAT_WEIGHT if lethal else best_frac * self.THREAT_WEIGHT


    # ================================================================
    # MATCHUP SCORE (unchanged, just uses patched threat/danger)
    # ================================================================
    def estimate_matchup(self, me, opp, battle=None):
        if me is None or opp is None:
            return 0.0

        me_stats = self._package_stats(me)
        opp_stats = self._package_stats(opp)

        score = 0.0

        # speed
        score += self.SPEED_COEFF if me_stats["spe"] > opp_stats["spe"] else -self.SPEED_COEFF

        # HP difference
        me_frac = me_stats["hp"] / me_stats["max_hp"]
        opp_frac = opp_stats["hp"] / opp_stats["max_hp"]
        score += (me_frac - opp_frac) * self.HP_COEFF

        # threat vs danger
        score += self._threat(me, opp, battle)
        score -= self._danger(opp, me)

        # boosts
        for val in me.boosts.values():
            if val > 0:
                score += val * self.BOOST_RETENTION_PENALTY
            elif val < 0:
                score += abs(val) * self.DEBUFF_ESCAPE_REWARD

        return score


    # ================================================================
    # STAY (patched but same logic)
    # ================================================================
    def _evaluate_stay(self, me, opp, battle):
        pred_mv, _ = self._predict_opponent_move(opp, me)

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
            best_out_frac = max(best_out_frac, frac)
            if frac >= 1.0:
                outgoing_lethal = True

        # Use packaged boosted stats
        me_stats = self._package_stats(me)
        opp_stats = self._package_stats(opp)

        faster = me_stats["spe"] > opp_stats["spe"]

        immediate = 0.0
        if faster:
            if outgoing_lethal:
                immediate += self.OHKO_BONUS_FAST * 5
            else:
                immediate += (best_out_frac - incoming_frac) * self.IMMEDIATE_TURN_WEIGHT
        else:
            if incoming_lethal:
                immediate += self.VOLUNTARY_DEATH_PENALTY
            else:
                immediate += (best_out_frac - incoming_frac) * self.IMMEDIATE_TURN_WEIGHT

        return immediate + self.estimate_matchup(me, opp, battle)
        # ================================================================
    # SWITCH TARGET (patched)
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

        immediate = ((dmg_me - dmg_cand) * self.IMMEDIATE_TURN_WEIGHT
                    if voluntary else self.SAFE_SWITCH_BONUS)

        if lethal_cand:
            immediate += self.VOLUNTARY_DEATH_PENALTY * 3 if voluntary else self.FORCED_DEATH_PENALTY

        post = self.estimate_matchup(cand, opp, battle)

        moves = self._get_moves_from_battle(cand.species, battle)
        best_frac = 0.0
        lethal = False

        for mv in moves:
            frac, _, leth_flag, _ = estimate_damage(mv, cand, opp)
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
    # SHOULD SWITCH? (same)
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

        return best_switch - stay >= self.SWITCH_OUT_MATCHUP_THRESHOLD


    # ================================================================
    # BEST SWITCH TARGET (unchanged)
    # ================================================================
    def best_switch_target(self, battle, voluntary=True):
        if not battle.available_switches:
            return None

        scored = [
            (self._evaluate_switch_target(mon,
                                          battle.opponent_active_pokemon,
                                          battle.active_pokemon,
                                          battle,
                                          voluntary),
             mon)
            for mon in battle.available_switches
        ]

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]