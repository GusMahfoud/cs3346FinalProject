"""
State Feature Extraction

Extracts feature vectors from poke-env battle states for the RL agent.
This implements the state representation φ(s) for the MDP.
"""

import numpy as np
from typing import List, Optional


def extract_features(battle) -> np.ndarray:
    """
    Extract feature vector φ(s) from battle state.
    
    Features include:
    - Our active Pokémon: HP%, types, status, boosts
    - Opponent active Pokémon: HP%, types, status
    - Team composition: Remaining mons, types
    - Battle state: Weather, terrain, hazards
    
    Args:
        battle: Current battle state from poke-env
        
    Returns:
        Feature vector as numpy array
    """
    features = []
    
    # Our active Pokémon features
    our_active = battle.active_pokemon
    if our_active:
        features.extend([
            our_active.current_hp / max(our_active.max_hp, 1),  # HP%
            len(our_active.types),  # Number of types
            int(our_active.status is not None),  # Has status
            our_active.boosts.get('atk', 0) / 6.0,  # Attack boost (normalized)
            our_active.boosts.get('def', 0) / 6.0,  # Defense boost
            our_active.boosts.get('spa', 0) / 6.0,  # Sp. Attack boost
            our_active.boosts.get('spd', 0) / 6.0,  # Sp. Defense boost
            our_active.boosts.get('spe', 0) / 6.0,  # Speed boost
        ])
        
        # Type encoding (one-hot for common types)
        type_features = encode_types(our_active.types)
        features.extend(type_features)
    else:
        # No active Pokémon
        features.extend([0.0] * (8 + 18))  # 8 stats + 18 type features
    
    # Opponent active Pokémon features
    opponent = battle.opponent_active_pokemon
    if opponent:
        features.extend([
            opponent.current_hp / max(opponent.max_hp, 1),  # HP%
            len(opponent.types),  # Number of types
            int(opponent.status is not None),  # Has status
            opponent.boosts.get('atk', 0) / 6.0,
            opponent.boosts.get('def', 0) / 6.0,
            opponent.boosts.get('spa', 0) / 6.0,
            opponent.boosts.get('spd', 0) / 6.0,
            opponent.boosts.get('spe', 0) / 6.0,
        ])
        type_features = encode_types(opponent.types)
        features.extend(type_features)
    else:
        features.extend([0.0] * (8 + 18))
    
    # Team composition
    our_team_size = len([p for p in battle.team.values() if p.current_hp > 0])
    opponent_team_size = len([p for p in battle.opponent_team.values() if p.current_hp > 0])
    
    features.extend([
        our_team_size / 6.0,  # Our remaining mons (normalized)
        opponent_team_size / 6.0,  # Opponent remaining mons
    ])
    
    # Battle state (weather, terrain, etc.)
    try:
        weather = getattr(battle, 'weather', None)
        weather_features = encode_weather(weather)
        features.extend(weather_features)
    except Exception:
        # Default to no weather if error
        features.extend([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Hazards (simplified - poke-env may not expose all)
    try:
        side_conditions = getattr(battle, 'side_conditions', {})
        features.extend([
            int(side_conditions.get('stealth_rock', False) if isinstance(side_conditions, dict) else False),
            int(side_conditions.get('spikes', False) if isinstance(side_conditions, dict) else False),
            int(side_conditions.get('toxic_spikes', False) if isinstance(side_conditions, dict) else False),
        ])
    except Exception:
        # Default to no hazards if error
        features.extend([0.0, 0.0, 0.0])
    
    # Turn number (normalized)
    features.append(min(battle.turn / 200.0, 1.0))
    
    # Available moves count (normalized)
    features.append(len(battle.available_moves) / 4.0)
    
    # Available switches count (normalized)
    features.append(len(battle.available_switches) / 5.0)
    
    return np.array(features, dtype=np.float32)


def encode_types(types: List[str]) -> List[float]:
    """
    Encode Pokémon types as one-hot features.
    
    Args:
        types: List of type strings
        
    Returns:
        List of 18 floats (one per type)
    """
    type_list = [
        'Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice',
        'Fighting', 'Poison', 'Ground', 'Flying', 'Psychic', 'Bug',
        'Rock', 'Ghost', 'Dragon', 'Dark', 'Steel', 'Fairy'
    ]
    
    features = [0.0] * 18
    for ptype in types:
        if ptype in type_list:
            idx = type_list.index(ptype)
            features[idx] = 1.0
    
    return features


def encode_weather(weather) -> List[float]:
    """
    Encode weather condition.
    
    Args:
        weather: Weather string, dict, or None (poke-env may return dict)
        
    Returns:
        List of weather features
    """
    weathers = ['none', 'sunnyday', 'raindance', 'sandstorm', 'hail', 'snow', 'snowscape']
    features = [0.0] * 6  # Only 6 weather types in our encoding
    
    if weather:
        # Handle dict case (poke-env sometimes returns weather as dict)
        if isinstance(weather, dict):
            weather_str = str(weather).lower()
        else:
            weather_str = str(weather).lower()
        
        # Map common weather names
        if 'snow' in weather_str or 'snowscape' in weather_str:
            features[5] = 1.0  # Snow
        elif 'sun' in weather_str or 'sunny' in weather_str:
            features[1] = 1.0  # Sunny Day
        elif 'rain' in weather_str:
            features[2] = 1.0  # Rain Dance
        elif 'sand' in weather_str:
            features[3] = 1.0  # Sandstorm
        elif 'hail' in weather_str:
            features[4] = 1.0  # Hail
        else:
            features[0] = 1.0  # None/unknown
    else:
        features[0] = 1.0  # No weather
    
    return features


def get_state_dim() -> int:
    """
    Get the dimension of the feature vector.
    
    Returns:
        Feature vector dimension
    """
    # Our active: 8 stats + 18 types = 26
    # Opponent active: 8 stats + 18 types = 26
    # Team: 2
    # Weather: 6
    # Hazards: 3
    # Misc: 3 (turn, moves, switches)
    return 26 + 26 + 2 + 6 + 3 + 3  # = 66

