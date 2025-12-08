# advanced_switcher.py
# Full updated heuristic switcher compatible with danger/threat-based encoder

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.pokemon import Pokemon
from computed_stats import get_real_stats


class SwitchHeuristics:
    """
    Standalone switch AI using:
    - computed real stats
    - correct boost handling
    - danger score and kill-threat influence
    - improved matchup scoring
    """

    # You may tune this at any time. -1.0 is good for the new scoring scale.
    SWITCH_OUT_MATCHUP_THRESHOLD = -1.0

    # Coefficients (safe defaults)
    TYPE_COEFFICIENT = 1.0
    SPEED_COEFFICIENT = 0.10
    HP_COEFFICIENT = 0.30
    DANGER_COEFFICIENT = 1.0        # how scared we are of incoming damage
    KILL_THREAT_COEFFICIENT = 1.0   # how much we value being able to KO

    # ------------------------------------------------------------
    # Effective speed (real base speed + boost influence)
    # ------------------------------------------------------------
    def _effective_speed(self, mon: Pokemon) -> int:
        if mon is None:
            return 0

        stats = get_real_stats(mon.species)
        base_spe = stats["spe"]
        boost = mon.boosts.get("spe", 0)

        # This is intentionally simplified for heuristic comparison only.
        # Raise/lower the 50 constant if necessary.
        return base_spe + boost * 50

    # ------------------------------------------------------------
    # Damage multipliers for type matchup (best case)
    # ------------------------------------------------------------
    def _type_matchup_score(self, mon: Pokemon, opp: Pokemon) -> float:
        if mon is None or opp is None:
            return 0.0

        try:
            # Offensive advantage
            my_best = max(opp.damage_multiplier(t) for t in mon.types if t is not None)
        except:
            my_best = 1.0

        try:
            # Defensive disadvantage
            opp_best = max(mon.damage_multiplier(t) for t in opp.types if t is not None)
        except:
            opp_best = 1.0

        return (my_best - opp_best) * self.TYPE_COEFFICIENT

    # ------------------------------------------------------------
    # Estimate incoming danger (fractional damage)
    # ------------------------------------------------------------
    def _danger_score(self, opp: Pokemon, mon: Pokemon) -> float:
        """
        Rough danger estimate:
        - If opponent likely deals large damage, return high danger.
        - Uses approximate "best move" logic without needing encoder.
        """
        if opp is None or mon is None:
            return 0.0

        best_frac = 0.0
        for mv in opp.moves.values():
            try:
                eff = mv.type.damage_multiplier(mon.type_1, mon.type_2)
            except:
                eff = 1.0

            bp = mv.base_power or 0

            # Scale by stats (very coarse but consistent)
            ostats = get_real_stats(opp.species)
            mstats = get_real_stats(mon.species)

            if mv.category.name == "PHYSICAL":
                atk = ostats["atk"]
                defense = mstats["def"]
            elif mv.category.name == "SPECIAL":
                atk = ostats["spa"]
                defense = mstats["spd"]
            else:
                continue

            est = (bp * atk / max(1, defense)) * eff
            frac = est / max(1, mon.current_hp)
            best_frac = max(best_frac, min(frac, 1.0))

        return best_frac * self.DANGER_COEFFICIENT

    # ------------------------------------------------------------
    # Kill threat: how likely this mon is to KO the opponent
    # ------------------------------------------------------------
    def _kill_threat(self, mon: Pokemon, opp: Pokemon) -> float:
        if mon is None or opp is None:
            return 0.0

        best_frac = 0.0
        for mv in mon.moves.values():
            try:
                eff = mv.type.damage_multiplier(opp.type_1, opp.type_2)
            except:
                eff = 1.0

            bp = mv.base_power or 0

            ustats = get_real_stats(mon.species)
            ostats = get_real_stats(opp.species)

            if mv.category.name == "PHYSICAL":
                atk = ustats["atk"]
                defense = ostats["def"]
            elif mv.category.name == "SPECIAL":
                atk = ustats["spa"]
                defense = ostats["spd"]
            else:
                continue

            est = (bp * atk / max(1, defense)) * eff
            frac = est / max(1, opp.current_hp)
            best_frac = max(best_frac, min(frac, 1.0))

        return best_frac * self.KILL_THREAT_COEFFICIENT

    # ------------------------------------------------------------
    # Full matchup estimation
    # ------------------------------------------------------------
    def estimate_matchup(self, mon: Pokemon, opp: Pokemon) -> float:
        if mon is None or opp is None:
            return 0.0

        score = 0.0

        # 1. Type advantage/disadvantage
        score += self._type_matchup_score(mon, opp)

        # 2. Speed differential
        my_spe = self._effective_speed(mon)
        opp_spe = self._effective_speed(opp)

        if my_spe > opp_spe:
            score += self.SPEED_COEFFICIENT
        elif opp_spe > my_spe:
            score -= self.SPEED_COEFFICIENT

        # 3. HP fraction difference
        score += mon.current_hp_fraction * self.HP_COEFFICIENT
        score -= opp.current_hp_fraction * self.HP_COEFFICIENT

        # 4. Incoming danger (big negative when the opponent threatens you)
        score -= self._danger_score(opp, mon)

        # 5. Your kill threat (big positive when you threaten *them*)
        score += self._kill_threat(mon, opp)

        return score

    # ------------------------------------------------------------
    # Should switch out?
    # ------------------------------------------------------------
    def should_switch_out(self, battle: AbstractBattle) -> bool:
        active = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        if active is None or opp is None:
            return False

        # 1. If any teammate has a clearly better matchup, consider switching
        switch_options = battle.available_switches
        good_switches = [
            m for m in switch_options
            if self.estimate_matchup(m, opp) > 0
        ]

        matchup_now = self.estimate_matchup(active, opp)

        # 2. Stat-drop switching
        if active.boosts["def"] <= -3 or active.boosts["spd"] <= -3:
            return True
        if active.boosts["atk"] <= -3 and active.stats["atk"] >= active.stats["spa"]:
            return True
        if active.boosts["spa"] <= -3 and active.stats["spa"] >= active.stats["atk"]:
            return True

        # 3. Main condition: bad matchup + at least one good alternative
        if good_switches and matchup_now < self.SWITCH_OUT_MATCHUP_THRESHOLD:
            return True

        return False

    # ------------------------------------------------------------
    # Best switch target
    # ------------------------------------------------------------
    def best_switch_target(self, battle: AbstractBattle):
        if not battle.available_switches:
            return None

        opp = battle.opponent_active_pokemon

        return max(
            battle.available_switches,
            key=lambda mon: self.estimate_matchup(mon, opp)
        )