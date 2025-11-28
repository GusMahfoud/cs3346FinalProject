from poke_env.player.player import Player

import numpy as np

# Correct engine modules (found on your system)
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.move import Move
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.weather import Weather
from poke_env.battle.field import Field

# And you SHOULD also import Battle from the same directory:
from poke_env.battle.battle import Battle

# -----------------------------
# CONSTANTS
# -----------------------------

STATUS_LIST = ["brn", "par", "slp", "tox", "frz"]

TYPES = [
    "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison",
    "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark",
    "steel", "fairy"
]
TYPE_INDEX = {t: i for i, t in enumerate(TYPES)}

BOOSTS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]

# Weather and terrain maps – only snow + grassy actually appear in our 20-mon pool,
# but we keep them generic so this stays usable if you expand later.
WEATHER_KEYS = [
    Weather.SUNNYDAY,
    Weather.RAINDANCE,
    Weather.SANDSTORM,
    Weather.HAIL,       # or Weather.SNOW in newer gens; poke-env maps appropriately
]
TERRAIN_KEYS = [
    Field.GRASSY_TERRAIN,
    Field.ELECTRIC_TERRAIN,
    Field.MISTY_TERRAIN,
    Field.PSYCHIC_TERRAIN,
]



# -----------------------------
# PER-POKÉMON ENCODING
# -----------------------------

def encode_pokemon(poke):
    """Encode one Pokémon into a fixed-length feature vector."""
    vec = []

    # Active / fainted
    vec.append(1.0 if getattr(poke, "active", False) else 0.0)
    vec.append(1.0 if poke.fainted else 0.0)

    # HP fraction
    if poke.max_hp and poke.current_hp is not None:
        vec.append(poke.current_hp / poke.max_hp)
    else:
        vec.append(0.0)

    # Status (one-hot)
    status_vec = [0] * len(STATUS_LIST)
    if poke.status:
        # poke.status is a Status enum -> use name.lower()
        name = str(poke.status).lower()
        if name in STATUS_LIST:
            status_vec[STATUS_LIST.index(name)] = 1
    vec.extend(status_vec)

    # Types (one-hot each)
    type1 = [0] * len(TYPES)
    type2 = [0] * len(TYPES)
    if poke.type_1 in TYPE_INDEX:
        type1[TYPE_INDEX[poke.type_1]] = 1
    if poke.type_2 in TYPE_INDEX:
        type2[TYPE_INDEX[poke.type_2]] = 1
    vec.extend(type1)
    vec.extend(type2)

    # Stat boosts
    for b in BOOSTS:
        boost_value = poke.boosts.get(b, 0)
        vec.append(boost_value / 6.0)

    # Moves: up to 4 slots
    moves_list = list(poke.moves.values())
    for i in range(4):
        if i < len(moves_list):
            move = moves_list[i]

            # Move exists
            vec.append(1.0)

            # PP fraction
            if move.max_pp:
                vec.append(move.current_pp / move.max_pp)
            else:
                vec.append(0.0)

            # Base power normalized (0 if purely status)
            bp = move.base_power or 0
            vec.append(bp / 250.0)

            # Move type one-hot
            type_vec = [0] * len(TYPES)
            if move.type in TYPE_INDEX:
                type_vec[TYPE_INDEX[move.type]] = 1
            vec.extend(type_vec)
        else:
            # Empty slot: no move
            vec.extend([0.0, 0.0, 0.0] + [0.0] * len(TYPES))

    return vec

# -----------------------------
# SIDE CONDITION HELPERS
# -----------------------------

def encode_side_conditions(sc_dict):
    """
    Encode side conditions for one side (ours or opponent's).
    Returns a small fixed-length vector.
    Only Stealth Rock, Spikes and Future Sight are relevant for our 20-mon pool.
    """
    vec = []

    # Stealth Rock: boolean
    rocks_active = 1.0 if SideCondition.STEALTH_ROCK in sc_dict else 0.0
    vec.append(rocks_active)

    # Spikes: layers 0-3 normalized
    spikes_layers = sc_dict.get(SideCondition.SPIKES, 0)
    vec.append(spikes_layers / 3.0)


    return vec

def encode_weather_and_terrain(battle):
    """
    Encode global weather and terrain into a one-hot + duration vector.
    For this pool:
      - Weather: Snow can be set by Chilly Reception (G-Slowking)
      - Terrain: Grassy Terrain from Rillaboom's Grassy Surge
    """
    vec = []

    # Weather one-hot + normalized duration
    weather_one_hot = [0.0] * len(WEATHER_KEYS)
    weather_duration = 0.0
    if battle.weather:
        w, turns = battle.weather
        if w in WEATHER_KEYS:
            weather_one_hot[WEATHER_KEYS.index(w)] = 1.0
        weather_duration = min(turns, 8) / 8.0
    vec.extend(weather_one_hot)
    vec.append(weather_duration)

    # Terrain one-hot + normalized duration
    terrain_one_hot = [0.0] * len(TERRAIN_KEYS)
    terrain_duration = 0.0
    if battle.fields:
        for f, turns in battle.fields.items():
            if f in TERRAIN_KEYS:
                terrain_one_hot[TERRAIN_KEYS.index(f)] = 1.0
                terrain_duration = min(turns, 8) / 8.0
                break
    vec.extend(terrain_one_hot)
    vec.append(terrain_duration)

    return vec

# -----------------------------
# FULL BATTLE STATE ENCODING
# -----------------------------

def encode_state(battle):
    """Encode a full battle state into a vector."""
    vec = []

    # Our 6 Pokémon
    for poke in battle.team.values():
        vec.extend(encode_pokemon(poke))

    # Opponent's 6 Pokémon (public info only – unrevealed stays mostly zeroed)
    for poke in battle.opponent_team.values():
        vec.extend(encode_pokemon(poke))

    # Our side conditions (Stealth Rock, Spikes, Future Sight)
    vec.extend(encode_side_conditions(battle.side_conditions))

    # Opponent side conditions
    vec.extend(encode_side_conditions(battle.opponent_side_conditions))

    # Global weather + terrain
    vec.extend(encode_weather_and_terrain(battle))

    # Trick Room – not present in this 20-mon sample, but cheap and generic
    # Trick Room (pseudoWeather stored in battle.fields)
    tr = battle.fields.get("trickroom", 0)
    vec.append(1.0 if tr > 0 else 0.0)
    vec.append(min(tr, 8) / 8.0)    # duration normalization


    # Turn number normalized
    vec.append(battle.turn / 100.0)

    return np.array(vec, dtype=np.float32)

# -----------------------------
# RL AGENT
# -----------------------------

class MyRLAgent(Player):
    """
    Base RL-ready agent with:
    - 10 discrete actions (4 moves + 6 switch options)
    - Rich game state encoding tailored to the 20-Pokémon mini-meta
    - Random policy by default (plug your model into compute_policy)
    """

    def __init__(self, battle_format="gen9ou", **kwargs):
        super().__init__(battle_format=battle_format, **kwargs)
        self.action_size = 10

    # -------------------------
    # POLICY / ACTION SELECTION
    # -------------------------

    def compute_policy(self, state_vec: np.ndarray) -> np.ndarray:
        """
        Placeholder for your RL network.
        Right now: uniform distribution (random strategy).
        Replace this with your NN forward pass.
        """
        logits = np.ones(self.action_size, dtype=np.float32)
        return logits / logits.sum()

    def choose_move(self, battle):
        """Main hook called by poke-env to choose an action."""
        state_vec = encode_state(battle)
        policy = self.compute_policy(state_vec)

        # Greedy action (you can sample instead if you want exploration)
        action_index = int(np.argmax(policy))

        legal = self.legal_actions(battle)
        if action_index not in legal:
            # Fallback: random legal action
            return self.choose_random_legal_action(battle)

        return self.action_to_move(action_index, battle)

    # -------------------------
    # ACTION SPACE / MAPPING
    # -------------------------

    def legal_actions(self, battle):
        """
        Return list of legal action indices in [0, 9]:
        - 0–3: move slots
        - 4–9: switch slots
        """
        indices = []

        # Moves
        for i, move in enumerate(battle.available_moves):
            if move is not None and i < 4:
                indices.append(i)

        # Switches
        for i, poke in enumerate(battle.available_switches):
            if poke is not None and i < 6:
                indices.append(4 + i)

        return indices

    def action_to_move(self, action, battle):
        """Map an integer action index to a poke-env Order object."""
        # 0–3: use move in slot (if exists)
        if 0 <= action <= 3:
            moves = battle.available_moves
            if action < len(moves) and moves[action] is not None:
                return self.create_order(moves[action])
            # Illegal or missing -> fallback
            return self.choose_random_legal_action(battle)

        # 4–9: switch to team slot (if exists)
        if 4 <= action <= 9:
            idx = action - 4
            switches = battle.available_switches
            if idx < len(switches) and switches[idx] is not None:
                return self.create_order(switches[idx])
            return self.choose_random_legal_action(battle)

        # Completely out-of-range? Just do something legal.
        return self.choose_random_legal_action(battle)

    def choose_random_legal_action(self, battle):
        """Sample uniformly from currently legal actions."""
        legal = self.legal_actions(battle)
        if not legal:
            # Let poke-env pick something completely random as a last resort
            return self.choose_random_move(battle)
        a = np.random.choice(legal)
        return self.action_to_move(a, battle)
