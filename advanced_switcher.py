# advanced_switcher.py

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.pokemon import Pokemon

class SwitchHeuristics:
    """
    Standalone switch AI using the exact logic from SimpleHeuristicsPlayer.
    Useful for RL reward shaping, or as a hybrid model.
    """

    SWITCH_OUT_MATCHUP_THRESHOLD = -2
    SPEED_TIER_COEFICIENT = 0.1
    HP_FRACTION_COEFICIENT = 0.4

    def estimate_matchup(self, mon: Pokemon, opponent: Pokemon) -> float:
        """Same matchup formula used in SimpleHeuristicsPlayer."""
        score = max([opponent.damage_multiplier(t) for t in mon.types if t is not None])
        score -= max(
            [mon.damage_multiplier(t) for t in opponent.types if t is not None]
        )

        # Speed tier bonus
        if mon.base_stats["spe"] > opponent.base_stats["spe"]:
            score += self.SPEED_TIER_COEFICIENT
        elif opponent.base_stats["spe"] > mon.base_stats["spe"]:
            score -= self.SPEED_TIER_COEFICIENT

        # HP contribution
        score += mon.current_hp_fraction * self.HP_FRACTION_COEFICIENT
        score -= opponent.current_hp_fraction * self.HP_FRACTION_COEFICIENT

        return score

    def should_switch_out(self, battle: AbstractBattle) -> bool:
        """Direct extraction of SimpleHeuristicsPlayer._should_switch_out."""
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        # Any GOOD switch-in?
        decent_switches = [
            m for m in battle.available_switches
            if self.estimate_matchup(m, opponent) > 0
        ]
        if decent_switches:

            # Stat-drop based switching
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
        """Return the optimal switch destination."""
        if not battle.available_switches:
            return None

        opponent = battle.opponent_active_pokemon

        return max(
            battle.available_switches,
            key=lambda mon: self.estimate_matchup(mon, opponent)
        )
