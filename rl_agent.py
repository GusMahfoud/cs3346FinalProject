# rl_agent.py
from poke_env.player import Player

class RLAgent(Player):
    """
    Skeleton for our RL-controlled agent.

    Right now:
    - It behaves exactly like a RandomPlayer (choose_random_move).
    Later:
    - We'll replace choose_move with logic that:
        1. Encodes `battle` into a state vector
        2. Asks a model (policy network, etc.) for an action
        3. Returns the chosen move/switch
    """

    def __init__(self, model=None, **kwargs):
        # `model` is a placeholder for a future RL policy (PyTorch/TF/etc.)
        super().__init__(**kwargs)
        self.model = model

    def choose_move(self, battle):
        """
        Decide what to do this turn.

        For now:
            - Just return a random legal move/switch.
        Later:
            - Use `battle` -> numeric state
            - Feed into self.model
            - Map model's action back to a move/switch
        """
        # TODO: replace this with RL policy later
        return self.choose_random_move(battle)
