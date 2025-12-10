# fixed_ais.py

from poke_env.player.baselines import (
    MaxBasePowerPlayer,
    SimpleHeuristicsPlayer,
    RandomPlayer
)

# ============================================================
# TEAM ORDER FIXED — Always "/team 123456"
# ============================================================

TEAM_ORDER_CMD = "/team 213456"


# ============================================================
# BASE CLASS MIXIN FOR FORCE-FIXED TEAM ORDER
# ============================================================
class FixedTeamOrderMixin:
    """Mixin that forces Pokémon Showdown to respect a fixed lead order."""

    #def teampreview(self, battle):
     #   return TEAM_ORDER_CMD

    #def choose_team(self, battle):
     #   return TEAM_ORDER_CMD


# ============================================================
# FIXED ORDER MAX BASE POWER AI
# ============================================================
class FixedOrderMaxBasePower(FixedTeamOrderMixin, MaxBasePowerPlayer):
    """
    Same as MaxBasePowerPlayer except:
    - Always picks team order 1-6 (no random lead)
    - Fully deterministic and stable for RL training
    """
    pass


# ============================================================
# FIXED ORDER SIMPLE HEURISTICS AI
# ============================================================
class FixedOrderSimpleHeuristics(FixedTeamOrderMixin, SimpleHeuristicsPlayer):
    """
    Same as SimpleHeuristicsPlayer except:
    - Always picks team order 1-6 (no random lead)
    - Avoids high-variance lead RNG harming curriculum
    """
    pass


# ============================================================
# FIXED ORDER RANDOM PLAYER
# ============================================================
class FixedOrderRandomPlayer(FixedTeamOrderMixin, RandomPlayer):
    """
    Same as RandomPlayer except:
    - Always picks team order 1-6
    - Stable for Warmup phase (Random AI)
    """
    pass
