"""
Heuristic Action Definitions

Defines strategic actions that the agent can take, each with associated logic
to determine when and how to execute them.
"""

from typing import List, Optional, Dict, Any


class ActionType:
    """Enum-like class for action types."""
    USE_BEST_TYPE_EFFECTIVE = 0
    USE_HIGHEST_POWER = 1
    USE_STATUS_MOVE = 2
    SWITCH_TO_COUNTER = 3
    SWITCH_TO_HEALTHY = 4
    USE_SETUP_MOVE = 5
    KEEP_CURRENT = 6


def get_legal_actions(battle) -> List[int]:
    """
    Get list of legal action IDs for current battle state.
    
    Args:
        battle: Current battle state from poke-env
        
    Returns:
        List of legal action IDs
    """
    legal = []
    
    # Can always use moves (if available)
    if battle.available_moves:
        legal.extend([
            ActionType.USE_BEST_TYPE_EFFECTIVE,
            ActionType.USE_HIGHEST_POWER,
            ActionType.USE_STATUS_MOVE,
            ActionType.USE_SETUP_MOVE
        ])
    
    # Can switch if we have other mons
    if len(battle.available_switches) > 0:
        legal.extend([
            ActionType.SWITCH_TO_COUNTER,
            ActionType.SWITCH_TO_HEALTHY
        ])
    
    # Can always keep current mon
    legal.append(ActionType.KEEP_CURRENT)
    
    return list(set(legal))  # Remove duplicates


def execute_action(action_id: int, battle):
    """
    Execute a heuristic action and return the move or switch to perform.
    
    This implements the action logic: given an action type and battle state,
    determine the actual move or switch to perform.
    
    Args:
        action_id: Action type ID
        battle: Current battle state
        
    Returns:
        Move or switch object (to be wrapped in create_order by Player)
    """
    opponent = battle.opponent_active_pokemon
    
    if action_id == ActionType.USE_BEST_TYPE_EFFECTIVE:
        # Find move with highest type effectiveness
        best_move = None
        best_damage = 0
        
        for move in battle.available_moves:
            if move.base_power > 0:  # Only damaging moves
                # Calculate type effectiveness (simplified)
                damage = move.base_power
                if opponent:
                    # Type effectiveness multiplier (simplified - would need full type chart)
                    effectiveness = estimate_type_effectiveness(move.type, opponent.types)
                    damage *= effectiveness
                
                if damage > best_damage:
                    best_damage = damage
                    best_move = move
        
        if best_move:
            return best_move
    
    elif action_id == ActionType.USE_HIGHEST_POWER:
        # Use move with highest base power
        best_move = max(battle.available_moves, 
                       key=lambda m: m.base_power if m.base_power > 0 else 0,
                       default=None)
        if best_move:
            return best_move
    
    elif action_id == ActionType.USE_STATUS_MOVE:
        # Use non-damaging move (status/utility)
        status_moves = [m for m in battle.available_moves if m.base_power == 0]
        if status_moves:
            return status_moves[0]  # Use first status move
    
    elif action_id == ActionType.USE_SETUP_MOVE:
        # Use stat-boosting move (simplified: any move that boosts stats)
        # In practice, would check move effects
        # Move objects use .id attribute, not .name
        setup_moves = [m for m in battle.available_moves 
                      if any(keyword in m.id.lower() for keyword in ['swordsdance', 'bulkup', 'calmmind', 'shellsmash', 'dragon', 'nastyplot'])]
        if setup_moves:
            return setup_moves[0]
        # Fallback to status move
        status_moves = [m for m in battle.available_moves if m.base_power == 0]
        if status_moves:
            return status_moves[0]
    
    elif action_id == ActionType.SWITCH_TO_COUNTER:
        # Switch to Pokémon that best counters opponent
        if opponent and battle.available_switches:
            best_switch = None
            best_score = -float('inf')
            
            for switch_option in battle.available_switches:
                # Score based on type advantage (simplified)
                score = estimate_type_advantage(switch_option, opponent)
                if score > best_score:
                    best_score = score
                    best_switch = switch_option
            
            if best_switch:
                return best_switch
    
    elif action_id == ActionType.SWITCH_TO_HEALTHY:
        # Switch to Pokémon with highest HP%
        if battle.available_switches:
            best_switch = max(battle.available_switches,
                            key=lambda p: p.current_hp / p.max_hp if p.max_hp > 0 else 0)
            return best_switch
    
    # Default: use best available move or random move
    if battle.available_moves:
        return battle.available_moves[0]
    
    # Last resort: random switch
    if battle.available_switches:
        return battle.available_switches[0]
    
    # Shouldn't reach here, but return None (will use choose_random_move)
    return None


def estimate_type_effectiveness(move_type: str, defender_types: List[str]) -> float:
    """
    Estimate type effectiveness multiplier (simplified).
    
    In a full implementation, would use complete type chart.
    For now, returns neutral (1.0) or makes simple assumptions.
    
    Args:
        move_type: Attacking move type
        defender_types: Defender's types
        
    Returns:
        Effectiveness multiplier (0.0, 0.5, 1.0, 2.0)
    """
    # Simplified type chart (common matchups)
    super_effective = {
        'Fire': ['Grass', 'Bug', 'Steel', 'Ice'],
        'Water': ['Fire', 'Ground', 'Rock'],
        'Electric': ['Water', 'Flying'],
        'Grass': ['Water', 'Ground', 'Rock'],
        'Ice': ['Grass', 'Ground', 'Flying', 'Dragon'],
        'Fighting': ['Normal', 'Rock', 'Steel', 'Ice', 'Dark'],
        'Poison': ['Grass', 'Fairy'],
        'Ground': ['Fire', 'Electric', 'Poison', 'Rock', 'Steel'],
        'Flying': ['Grass', 'Fighting', 'Bug'],
        'Psychic': ['Fighting', 'Poison'],
        'Bug': ['Grass', 'Psychic', 'Dark'],
        'Rock': ['Fire', 'Ice', 'Flying', 'Bug'],
        'Ghost': ['Psychic', 'Ghost'],
        'Dragon': ['Dragon'],
        'Dark': ['Psychic', 'Ghost'],
        'Steel': ['Ice', 'Rock', 'Fairy'],
        'Fairy': ['Fighting', 'Dragon', 'Dark']
    }
    
    not_very_effective = {
        'Fire': ['Water', 'Rock', 'Dragon'],
        'Water': ['Water', 'Grass', 'Dragon'],
        'Electric': ['Electric', 'Grass', 'Dragon'],
        'Grass': ['Fire', 'Grass', 'Poison', 'Flying', 'Bug', 'Dragon', 'Steel'],
        'Ice': ['Fire', 'Water', 'Ice', 'Steel'],
        'Fighting': ['Flying', 'Poison', 'Psychic', 'Bug', 'Fairy'],
        'Poison': ['Poison', 'Ground', 'Rock', 'Ghost'],
        'Ground': ['Grass', 'Bug'],
        'Flying': ['Electric', 'Rock', 'Steel'],
        'Psychic': ['Psychic', 'Steel'],
        'Bug': ['Fire', 'Fighting', 'Poison', 'Flying', 'Ghost', 'Steel', 'Fairy'],
        'Rock': ['Fighting', 'Ground', 'Steel'],
        'Ghost': ['Dark'],
        'Dragon': ['Steel'],
        'Dark': ['Fighting', 'Dark', 'Fairy'],
        'Steel': ['Fire', 'Water', 'Electric', 'Steel'],
        'Fairy': ['Fire', 'Poison', 'Steel']
    }
    
    if not move_type or not defender_types:
        return 1.0
    
    effectiveness = 1.0
    for def_type in defender_types:
        if move_type in super_effective and def_type in super_effective[move_type]:
            effectiveness *= 2.0
        elif move_type in not_very_effective and def_type in not_very_effective[move_type]:
            effectiveness *= 0.5
    
    return effectiveness


def estimate_type_advantage(pokemon, opponent) -> float:
    """
    Estimate type advantage score for switching.
    Higher score = better matchup.
    
    Args:
        pokemon: Pokémon to switch to
        opponent: Opponent's active Pokémon
        
    Returns:
        Advantage score
    """
    if not opponent:
        return 0.0
    
    score = 0.0
    # Score based on how well our types match against opponent's moves
    # Simplified: assume opponent will use moves, check our resistances
    # In practice, would consider opponent's likely moveset
    
    # Basic heuristic: more types = more coverage (simplified)
    score += len(pokemon.types) * 0.5
    
    return score

