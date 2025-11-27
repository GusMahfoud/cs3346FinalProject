"""
Training Loop for RL Agent

Implements the training pipeline: agent vs bots, then self-play.
Uses REINFORCE to update agent policy.
"""

import yaml
import asyncio
import logging
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import numpy as np

from poke_env.player import Player, RandomPlayer
from poke_env.ps_client.server_configuration import ServerConfiguration

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create local server configuration (ServerConfiguration is a NamedTuple, use positional args)
# Server URL needs ws:// scheme and /showdown/websocket path for WebSocket connection
LocalhostServerConfiguration = ServerConfiguration(
    "ws://localhost:8000/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?"
)

from src.agent import WeightBasedAgent
from src.state_features import get_state_dim
from src.actions import ActionType


class SimpleHeuristicsPlayer(Player):
    """Simple heuristic opponent that uses basic strategies."""
    
    def __init__(self, *args, **kwargs):
        # Use local server configuration
        kwargs.setdefault('server_configuration', LocalhostServerConfiguration)
        super().__init__(*args, **kwargs)
    
    def choose_move(self, battle):
        """Select move using simple heuristics."""
        # Prefer damaging moves
        if battle.available_moves:
            damaging_moves = [m for m in battle.available_moves if m.base_power > 0]
            if damaging_moves:
                # Use move with highest power
                best_move = max(damaging_moves, key=lambda m: m.base_power)
                return self.create_order(best_move)
            else:
                # Use any move if no damaging moves
                return self.create_order(battle.available_moves[0])
        
        # Switch if no moves available
        if battle.available_switches:
            return self.create_order(battle.available_switches[0])
        
        # Default: random move
        return self.choose_random_move(battle)


class RLPlayer(Player):
    """Wrapper to use our RL agent with poke-env."""
    
    def __init__(self, agent: WeightBasedAgent, *args, **kwargs):
        # Use local server configuration
        kwargs.setdefault('server_configuration', LocalhostServerConfiguration)
        super().__init__(*args, **kwargs)
        self.agent = agent
        self.training = True
    
    def choose_move(self, battle):
        """Select move using RL agent."""
        move_or_switch = self.agent.select_action(battle, training=self.training)
        if move_or_switch is None:
            return self.choose_random_move(battle)
        return self.create_order(move_or_switch)


def compute_reward(battle, config: dict) -> float:
    """
    Compute reward for episode.
    
    Args:
        battle: Final battle state
        config: Configuration dict with reward settings
        
    Returns:
        Total reward
    """
    rewards = config['rewards']
    reward = 0.0
    
    # Win/loss reward
    if battle.won:
        reward += rewards['win']
    else:
        reward += rewards['loss']
    
    # Step penalty (encourage efficiency)
    reward += battle.turn * rewards['step_penalty']
    
    # Optional: KO bonus / lose penalty
    # Would need to track KOs during battle (simplified for now)
    
    return reward


async def train_phase_async(agent: WeightBasedAgent, opponent: Player, 
                            num_episodes: int, config: dict, phase_name: str):
    """
    Train agent against an opponent for a phase (async version).
    
    Args:
        agent: RL agent to train
        opponent: Opponent player
        num_episodes: Number of episodes to train
        config: Configuration dict
        phase_name: Name of training phase (for logging)
    """
    logger.info(f"Starting {phase_name}")
    logger.info(f"RL Player username: will be generated")
    logger.info(f"Opponent username: {opponent.username}")
    
    rl_player = RLPlayer(agent=agent, battle_format=config['battle']['format'])
    
    print(f"\n=== {phase_name} ===")
    print(f"Training for {num_episodes} episodes...")
    logger.info(f"RL Player created: {rl_player.username}")
    
    wins = 0
    episode_rewards = []
    
    for episode in tqdm(range(num_episodes), desc=phase_name):
        # Reset agent episode data
        agent.episode_states = []
        agent.episode_actions = []
        
        logger.info(f"Episode {episode + 1}/{num_episodes}: Starting battle...")
        
        # Run battle (async)
        try:
            # battle_against handles team generation automatically
            logger.info(f"Episode {episode + 1}: Challenging opponent...")
            await rl_player.battle_against(opponent, n_battles=1)
            logger.info(f"Episode {episode + 1}: Battle completed")
            
            # Wait a bit for battle to start
            await asyncio.sleep(1)
            
            # Get battle result - poke-env stores battles in _battles dict
            battles = list(rl_player._battles.values())
            logger.info(f"Episode {episode + 1}: Found {len(battles)} battles in history")
            
            if battles:
                battle = battles[-1]  # Get last battle
                
                # Wait for battle to finish if still ongoing
                max_wait = 300  # Max 30 seconds
                wait_count = 0
                while not battle.finished and wait_count < max_wait:
                    await asyncio.sleep(0.1)
                    wait_count += 1
                    if wait_count % 50 == 0:
                        logger.info(f"Episode {episode + 1}: Waiting for battle to finish... (turn {battle.turn})")
                
                if not battle.finished:
                    logger.warning(f"Episode {episode + 1}: Battle did not finish in time")
                    continue
                
                logger.info(f"Episode {episode + 1}: Battle finished: won={battle.won}, turns={battle.turn}")
                
                # Compute reward
                reward = compute_reward(battle, config)
                episode_rewards.append(reward)
                logger.info(f"Episode {episode + 1}: Reward={reward:.2f}")
                
                # Update policy
                logger.info(f"Episode {episode + 1}: Updating policy...")
                agent.update_policy(reward)
                logger.info(f"Episode {episode + 1}: Policy updated")
                
                # Track wins
                if battle.won:
                    wins += 1
                    logger.info(f"Episode {episode + 1}: WIN!")
                else:
                    logger.info(f"Episode {episode + 1}: Loss")
                
                # Decay exploration
                agent.decay_epsilon()
                logger.info(f"Episode {episode + 1}: Epsilon={agent.epsilon:.3f}")
            else:
                logger.warning(f"Episode {episode + 1}: No battles found after battle_against call")
            
        except Exception as e:
            logger.error(f"Error in episode {episode + 1}: {e}", exc_info=True)
            print(f"Error in episode {episode + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Logging
        if (episode + 1) % 10 == 0:
            win_rate = wins / (episode + 1)
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
            logger.info(f"Episode {episode+1} Progress: Win rate: {win_rate:.2%}, "
                       f"Avg reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
            print(f"\nEpisode {episode+1}: Win rate: {win_rate:.2%}, "
                  f"Avg reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    final_win_rate = wins / num_episodes if num_episodes > 0 else 0.0
    avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
    logger.info(f"{phase_name} Complete: Win rate={final_win_rate:.2%}, Avg reward={avg_reward:.2f}")
    print(f"\n{phase_name} Complete:")
    print(f"  Final win rate: {final_win_rate:.2%}")
    print(f"  Average reward: {avg_reward:.2f}")


def train(config_path: str = "config.yaml"):
    """
    Main training function.
    
    Implements hybrid training:
    1. Phase 1: Train against Random bot
    2. Phase 2: Train against stronger bot
    3. Phase 3: Self-play (optional)
    
    Args:
        config_path: Path to configuration file
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize agent
    state_dim = get_state_dim()
    num_actions = config['agent']['num_actions']
    
    agent = WeightBasedAgent(
        state_dim=state_dim,
        num_actions=num_actions,
        learning_rate=config['agent']['learning_rate'],
        epsilon_start=config['agent']['epsilon_start'],
        epsilon_end=config['agent']['epsilon_end'],
        epsilon_decay=config['agent']['epsilon_decay'],
        use_baseline=config['training']['use_baseline']
    )
    
    print("=" * 50)
    print("Pokémon Battle RL Agent Training")
    print("=" * 50)
    print(f"State dimension: {state_dim}")
    print(f"Number of actions: {num_actions}")
    print(f"Learning rate: {config['agent']['learning_rate']}")
    print(f"Battle format: {config['battle']['format']}")
    print("=" * 50)
    logger.info("Training initialized")
    logger.info(f"State dim: {state_dim}, Actions: {num_actions}, LR: {config['agent']['learning_rate']}")
    
    # Run async training
    async def run_training():
        # Phase 1: Train against Random bot
        random_opponent = RandomPlayer(
            battle_format=config['battle']['format'],
            server_configuration=LocalhostServerConfiguration
        )
        await train_phase_async(
            agent=agent,
            opponent=random_opponent,
            num_episodes=config['phases']['phase1_episodes'],
            config=config,
            phase_name="Phase 1: vs Random Bot"
        )
        
        # Phase 2: Train against stronger bot
        if config['phases'].get('phase2_bot') == 'MaxDamage':
            strong_opponent = SimpleHeuristicsPlayer(
                battle_format=config['battle']['format'],
                server_configuration=LocalhostServerConfiguration
            )
            await train_phase_async(
                agent=agent,
                opponent=strong_opponent,
                num_episodes=config['phases']['phase2_episodes'],
                config=config,
                phase_name="Phase 2: vs Stronger Bot"
            )
        
        # Phase 3: Self-play (optional)
        if config['phases'].get('phase3_selfplay', False):
            print("\n=== Phase 3: Self-Play ===")
            print("Training agent against itself...")
            
            # Create copy of agent for opponent
            opponent_agent = WeightBasedAgent(
                state_dim=state_dim,
                num_actions=num_actions,
                learning_rate=config['agent']['learning_rate'],
                epsilon_start=0.1,  # Lower exploration for opponent
                epsilon_end=0.01,
                epsilon_decay=0.995,
                use_baseline=False
            )
            opponent_agent.weights = agent.weights.copy()  # Copy weights
            
            opponent_player = RLPlayer(agent=opponent_agent, 
                                      battle_format=config['battle']['format'])
            opponent_player.training = False  # Opponent doesn't learn
            
            await train_phase_async(
                agent=agent,
                opponent=opponent_player,
                num_episodes=config['phases']['phase3_episodes'],
                config=config,
                phase_name="Phase 3: Self-Play"
            )
    
    # Run async training
    asyncio.run(run_training())
    
    # Save trained agent
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "trained_agent.npz"
    agent.save(str(checkpoint_path))
    print(f"\nSaved trained agent to {checkpoint_path}")
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    train(config_path)
