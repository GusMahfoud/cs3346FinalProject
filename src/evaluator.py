"""
Evaluation Module

Evaluates trained agent against different opponents and computes metrics.
"""

import yaml
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from poke_env.player import Player
from poke_env.ps_client.server_configuration import ServerConfiguration

# Create local server configuration (ServerConfiguration is a NamedTuple, use positional args)
# Server URL needs ws:// scheme and /showdown/websocket path for WebSocket connection
LocalhostServerConfiguration = ServerConfiguration(
    "ws://localhost:8000/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?"
)

from src.agent import WeightBasedAgent
from src.state_features import get_state_dim
from src.trainer import RLPlayer, RandomPlayer, SimpleHeuristicsPlayer


async def evaluate_agent_async(agent: WeightBasedAgent, opponent: Player,
                               num_battles: int, config: dict) -> Dict[str, float]:
    """
    Evaluate agent against an opponent.
    
    Args:
        agent: Trained RL agent
        opponent: Opponent player
        num_battles: Number of battles to run
        config: Configuration dict
        
    Returns:
        Dictionary with evaluation metrics
    """
    rl_player = RLPlayer(agent=agent, battle_format=config['battle']['format'])
    rl_player.training = False  # No exploration during evaluation
    
    wins = 0
    total_turns = 0
    episode_rewards = []
    
    for i in range(num_battles):
        try:
            # Run battle (async)
            await rl_player.battle_against(opponent, n_battles=1)
            
            # Get battle result
            battles = list(rl_player._battles.values())
            if battles:
                battle = battles[-1]
                
                # Compute reward
                reward = 0.0
                if battle.won:
                    reward += config['rewards']['win']
                    wins += 1
                else:
                    reward += config['rewards']['loss']
                
                reward += battle.turn * config['rewards']['step_penalty']
                episode_rewards.append(reward)
                total_turns += battle.turn
            
        except Exception as e:
            print(f"Error in battle {i}: {e}")
            continue
    
    win_rate = wins / num_battles
    avg_turns = total_turns / num_battles
    avg_reward = np.mean(episode_rewards)
    
    return {
        'win_rate': win_rate,
        'avg_turns': avg_turns,
        'avg_reward': avg_reward,
        'num_battles': num_battles
    }


async def evaluate_async(config_path: str = "config.yaml", checkpoint_path: Optional[str] = None):
    """
    Evaluate trained agent against multiple opponents.
    
    Args:
        config_path: Path to configuration file
        checkpoint_path: Path to agent checkpoint (optional)
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load agent
    state_dim = get_state_dim()
    num_actions = config['agent']['num_actions']
    
    agent = WeightBasedAgent(
        state_dim=state_dim,
        num_actions=num_actions,
        learning_rate=config['agent']['learning_rate'],
        epsilon_start=0.0,  # No exploration during evaluation
        epsilon_end=0.0,
        epsilon_decay=1.0,
        use_baseline=False
    )
    
    if checkpoint_path:
        agent.load(checkpoint_path)
        print(f"Loaded agent from {checkpoint_path}")
    else:
        checkpoint_path = Path("checkpoints") / "trained_agent.npz"
        if checkpoint_path.exists():
            agent.load(str(checkpoint_path))
            print(f"Loaded agent from {checkpoint_path}")
        else:
            print("No checkpoint found. Using random weights.")
    
    print("\n" + "=" * 50)
    print("Agent Evaluation")
    print("=" * 50)
    
    num_battles = 100
    
    # Evaluate against Random bot
    random_opponent = RandomPlayer(
        battle_format=config['battle']['format'],
        server_configuration=LocalhostServerConfiguration
    )
    results_random = await evaluate_agent_async(agent, random_opponent, num_battles, config)
    
    print(f"\nvs Random Bot:")
    print(f"  Win rate: {results_random['win_rate']:.2%}")
    print(f"  Avg turns: {results_random['avg_turns']:.1f}")
    print(f"  Avg reward: {results_random['avg_reward']:.2f}")
    
    # Evaluate against stronger bot
    strong_opponent = SimpleHeuristicsPlayer(
        battle_format=config['battle']['format'],
        server_configuration=LocalhostServerConfiguration
    )
    results_strong = await evaluate_agent_async(agent, strong_opponent, num_battles, config)
    
    print(f"\nvs Stronger Bot:")
    print(f"  Win rate: {results_strong['win_rate']:.2%}")
    print(f"  Avg turns: {results_strong['avg_turns']:.1f}")
    print(f"  Avg reward: {results_strong['avg_reward']:.2f}")
    
    print("\n" + "=" * 50)


def evaluate(config_path: str = "config.yaml", checkpoint_path: Optional[str] = None):
    """Synchronous wrapper for async evaluation."""
    asyncio.run(evaluate_async(config_path, checkpoint_path))


if __name__ == "__main__":
    import sys
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else None
    evaluate(checkpoint_path=checkpoint)

