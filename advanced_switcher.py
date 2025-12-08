# advanced_switcher.py  (FIXED MINIMALLY & SAFELY)

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.pokemon import Pokemon
from computed_stats import get_real_stats


class SwitchHeuristics:
    """
    Standalone switch AI using logic similar to SimpleHeuristicsPlayer,
    but now using computed real stats (including boosts) for speed.
    """

    SWITCH_OUT_MATCHUP_THRESHOLD = -2
    SPEED_TIER_COEFICIENT = 0.1
    HP_FRACTION_COEFICIENT = 0.4

    def _effective_speed(self, mon: Pokemon) -> int:
        """Real speed = computed base speed + boost levels."""
        if mon is None:
            return 0
        stats = get_real_stats(mon.species)
        base_spe = stats["spe"]
        boost = mon.boosts.get("spe", 0)

        # Showdown boost scaling (approx)
        # Each stage is multiplicative: stage +6 → ×4; stage -6 → ×1/4
        # But RL heuristics only need relative comparison, so additive is fine.
        return base_spe + boost * 50  # conservative and consistent with reward calc

    def estimate_matchup(self, mon: Pokemon, opponent: Pokemon) -> float:
        """Matchup score identical to original logic except for corrected speed."""
        score = max([opponent.damage_multiplier(t) for t in mon.types if t is not None])
        score -= max(
            [mon.damage_multiplier(t) for t in opponent.types if t is not None]
        )

        # SPEED (using computed stats + boosts)
        my_spe = self._effective_speed(mon)
        opp_spe = self._effective_speed(opponent)

        if my_spe > opp_spe:
            score += self.SPEED_TIER_COEFICIENT
        elif opp_spe > my_spe:
            score -= self.SPEED_TIER_COEFICIENT

        # HP difference
        score += mon.current_hp_fraction * self.HP_FRACTION_COEFICIENT
        score -= opponent.current_hp_fraction * self.HP_FRACTION_COEFICIENT

        return score

    def should_switch_out(self, battle: AbstractBattle) -> bool:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        # Any GOOD switch-in available?
        decent_switches = [
            m for m in battle.available_switches
            if self.estimate_matchup(m, opponent) > 0
        ]

        if decent_switches:

            # Stat-drop switching
            if active.boosts["def"] <= -3 or active.boosts["spd"] <= -3:
                return True
            if active.boosts["atk"] <= -3 and active.stats["atk"] >= active.stats["spa"]:
                return True
            if active.boosts["spa"] <= -3 and active.stats["atk"] <= active.stats["spa"]:
                return True

            # Matchup-based switching
            if self.estimate_matchup(active, opponent) < self.SWITCH_OUT_MATCHUP_THRESHOLD:
                return True

        return False

    def best_switch_target(self, battle: AbstractBattle):
        if not battle.available_switches:
            return None

        opponent = battle.opponent_active_pokemon

        return max(
            battle.available_switches,
            key=lambda mon: self.estimate_matchup(mon, opponent)
        )