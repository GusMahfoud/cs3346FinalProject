import numpy as np
from poke_env.battle.battle import Battle


class RewardCalculator:
    def __init__(
        self,
        enable_intermediate=True,
        terminal_win=100.0,
        terminal_loss=-100.0,
        terminal_draw=0.0,
        hp_weight=0.5,
        ko_reward=20.0,
        ko_penalty=-15.0,
        status_reward=2.0,
        status_penalty=-1.0,
        type_effectiveness_reward=5.0,
        type_effectiveness_penalty=-3.0,
        switch_penalty=-1.0,
        turn_penalty=-0.1,
        momentum_reward=3.0,
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
        
        # Track previous state for delta calculations
        self.prev_hp_advantage = 0.0
        self.consecutive_favorable_turns = 0

    def _get_hp_percentage(self, battle: Battle, is_opponent: bool = False):
        """Calculate total HP percentage for a team."""
        team = battle.opponent_team if is_opponent else battle.team
        total_hp = 0.0
        total_max_hp = 0.0
        
        for poke in team.values():
            if poke.max_hp:
                total_max_hp += poke.max_hp
                if poke.current_hp is not None:
                    total_hp += poke.current_hp
        
        if total_max_hp == 0:
            return 0.0
        return total_hp / total_max_hp
    
    def _get_hp_percentage_from_dict(self, state_dict: dict, is_opponent: bool = False):
        """Calculate total HP percentage from state dict."""
        team_key = 'opponent_team' if is_opponent else 'team'
        team = state_dict.get(team_key, {})
        total_hp = 0.0
        total_max_hp = 0.0
        
        for poke_data in team.values():
            max_hp = poke_data.get('max_hp', 0)
            if max_hp:
                total_max_hp += max_hp
                current_hp = poke_data.get('current_hp', 0)
                if current_hp is not None:
                    total_hp += current_hp
        
        if total_max_hp == 0:
            return 0.0
        return total_hp / total_max_hp

    def _count_fainted(self, battle: Battle, is_opponent: bool = False):
        """Count fainted Pokémon."""
        team = battle.opponent_team if is_opponent else battle.team
        return sum(1 for poke in team.values() if poke.fainted)

    def _count_status_conditions(self, battle: Battle, is_opponent: bool = False):
        """Count Pokémon with status conditions."""
        team = battle.opponent_team if is_opponent else battle.team
        return sum(1 for poke in team.values() if poke.status)

    def compute_turn_reward(
        self,
        prev_battle,
        current_battle: Battle,
        action_was_switch: bool = False,
    ) -> float:
        """
        Compute intermediate reward for a turn transition.
        
        Args:
            prev_battle: Battle state before action
            current_battle: Battle state after action
            action_was_switch: Whether the action taken was a switch
        
        Returns:
            Reward value for this turn
        """
        if not self.enable_intermediate:
            return 0.0

        reward = 0.0

        # HP differential
        our_hp = self._get_hp_percentage(current_battle, is_opponent=False)
        opp_hp = self._get_hp_percentage(current_battle, is_opponent=True)
        hp_advantage = our_hp - opp_hp
        hp_reward = hp_advantage * self.hp_weight
        reward += hp_reward

        # HP change (momentum)
        if prev_battle and isinstance(prev_battle, dict):
            # Handle dict state from agent
            prev_our_hp = self._get_hp_percentage_from_dict(prev_battle, is_opponent=False)
            prev_opp_hp = self._get_hp_percentage_from_dict(prev_battle, is_opponent=True)
            prev_hp_advantage = prev_our_hp - prev_opp_hp
        elif prev_battle:
            prev_our_hp = self._get_hp_percentage(prev_battle, is_opponent=False)
            prev_opp_hp = self._get_hp_percentage(prev_battle, is_opponent=True)
            prev_hp_advantage = prev_our_hp - prev_opp_hp
        else:
            prev_hp_advantage = 0.0
        
        hp_change = hp_advantage - prev_hp_advantage
        if hp_change > 0:
            self.consecutive_favorable_turns += 1
            if self.consecutive_favorable_turns >= 2:
                reward += self.momentum_reward
        else:
            self.consecutive_favorable_turns = 0
        
        self.prev_hp_advantage = hp_advantage

        # KO rewards
        if prev_battle and isinstance(prev_battle, dict):
            prev_our_fainted = sum(1 for v in prev_battle['team'].values() if v['fainted'])
            prev_opp_fainted = sum(1 for v in prev_battle['opponent_team'].values() if v['fainted'])
        elif prev_battle:
            prev_our_fainted = self._count_fainted(prev_battle, is_opponent=False)
            prev_opp_fainted = self._count_fainted(prev_battle, is_opponent=True)
        else:
            prev_our_fainted = 0
            prev_opp_fainted = 0
        
        our_fainted = self._count_fainted(current_battle, is_opponent=False)
        opp_fainted = self._count_fainted(current_battle, is_opponent=True)
        
        our_kos = our_fainted - prev_our_fainted
        opp_kos = opp_fainted - prev_opp_fainted
        
        if our_kos > 0:
            reward += our_kos * self.ko_penalty
        
        if opp_kos > 0:
            reward += opp_kos * self.ko_reward

        # Status conditions
        if prev_battle and isinstance(prev_battle, dict):
            prev_our_status = sum(1 for v in prev_battle['team'].values() if v['status'])
            prev_opp_status = sum(1 for v in prev_battle['opponent_team'].values() if v['status'])
        elif prev_battle:
            prev_our_status = self._count_status_conditions(prev_battle, is_opponent=False)
            prev_opp_status = self._count_status_conditions(prev_battle, is_opponent=True)
        else:
            prev_our_status = 0
            prev_opp_status = 0
        
        our_status = self._count_status_conditions(current_battle, is_opponent=False)
        opp_status = self._count_status_conditions(current_battle, is_opponent=True)
        
        our_status_change = our_status - prev_our_status
        opp_status_change = opp_status - prev_opp_status
        
        if our_status_change > 0:
            reward += our_status_change * self.status_penalty
        
        if opp_status_change > 0:
            reward += opp_status_change * self.status_reward

        # Switch penalty
        if action_was_switch:
            reward += self.switch_penalty

        # Turn penalty
        reward += self.turn_penalty

        return reward

    def compute_terminal_reward(self, battle: Battle) -> float:
        """
        Compute terminal reward when battle ends.
        
        Args:
            battle: Final battle state
        
        Returns:
            Terminal reward value
        """
        if battle.won:
            return self.terminal_win
        elif battle.lost:
            return self.terminal_loss
        else:
            return self.terminal_draw

    def reset(self):
        """Reset internal state for new battle."""
        self.prev_hp_advantage = 0.0
        self.consecutive_favorable_turns = 0

