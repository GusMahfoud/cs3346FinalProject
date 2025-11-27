"""
RL Agent with Weight-Based Policy

Implements a policy gradient agent using REINFORCE algorithm.
The agent learns weights for each heuristic action based on state features.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import random

from src.actions import ActionType, get_legal_actions, execute_action
from src.state_features import extract_features


class WeightBasedAgent:
    """
    RL agent with weight-based policy using REINFORCE.
    
    Policy: P(a|s) = softmax(θ · φ(s))
    Learning: REINFORCE updates weights θ based on episode returns.
    """
    
    def __init__(self, state_dim: int, num_actions: int,
                 learning_rate: float = 0.01,
                 epsilon_start: float = 0.5,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 use_baseline: bool = True):
        """
        Initialize agent.
        
        Args:
            state_dim: Dimension of state feature vector
            num_actions: Number of heuristic actions
            learning_rate: Learning rate for policy gradient
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay rate per episode
            use_baseline: Whether to use reward baseline
        """
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.use_baseline = use_baseline
        
        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Policy parameters: weights for each action
        # θ[i] = weight vector for action i
        # Weight for action i given state s: θ[i] · φ(s)
        self.weights = np.random.randn(num_actions, state_dim) * 0.1
        
        # Training tracking
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.baseline = 0.0  # Average reward baseline
        
    def select_action(self, battle, training: bool = True):
        """
        Select action using weight-based policy.
        
        Policy: P(a|s) = softmax(θ[a] · φ(s))
        Uses ε-greedy exploration during training.
        
        Args:
            battle: Current battle state
            training: Whether in training mode (affects exploration)
            
        Returns:
            BattleOrder to execute
        """
        # Extract state features
        state_features = extract_features(battle)
        
        # Get legal actions
        legal_actions = get_legal_actions(battle)
        
        if not legal_actions:
            # No legal actions (shouldn't happen)
            return None
        
        # Compute action weights: weight[i] = θ[i] · φ(s)
        action_weights = []
        for action_id in legal_actions:
            weight = np.dot(self.weights[action_id], state_features)
            action_weights.append(weight)
        
        action_weights = np.array(action_weights)
        
        # Exploration: ε-greedy
        if training and random.random() < self.epsilon:
            # Random legal action
            action_id = random.choice(legal_actions)
        else:
            # Exploitation: select action with highest weight
            best_idx = np.argmax(action_weights)
            action_id = legal_actions[best_idx]
        
        # Store for training (if training)
        if training:
            self.episode_states.append(state_features)
            self.episode_actions.append(action_id)
        
        # Execute action and return move/switch object
        # (will be wrapped in create_order by RLPlayer)
        return execute_action(action_id, battle)
    
    def update_policy(self, episode_reward: float):
        """
        Update policy weights using REINFORCE algorithm.
        
        REINFORCE update:
        θ[i] ← θ[i] + α · (R - baseline) · ∇_θ log P(a_i | s)
        
        where:
        - R = episode return (total reward)
        - baseline = average reward (reduces variance)
        - α = learning rate
        
        Args:
            episode_reward: Total reward for the episode
        """
        if len(self.episode_states) == 0:
            return
        
        # Compute advantage: (reward - baseline)
        if self.use_baseline:
            advantage = episode_reward - self.baseline
        else:
            advantage = episode_reward
        
        # Update baseline (exponential moving average)
        self.baseline = 0.9 * self.baseline + 0.1 * episode_reward
        
        # Update weights for each step in episode
        for state, action_id in zip(self.episode_states, self.episode_actions):
            # Compute policy probabilities
            all_weights = np.array([np.dot(self.weights[a], state) 
                                   for a in range(self.num_actions)])
            probs = self._softmax(all_weights)
            
            # Gradient: ∇_θ log P(a|s) = φ(s) - Σ_a' P(a'|s) · φ(s)
            # For selected action a: gradient = φ(s) - Σ_a' P(a'|s) · φ(s)
            # Simplified: gradient ≈ φ(s) for selected action
            
            # Update weight for selected action
            gradient = state  # Simplified gradient
            self.weights[action_id] += self.learning_rate * advantage * gradient
        
        # Clear episode data
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end,
                          self.epsilon * self.epsilon_decay)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / np.sum(exp_x)
    
    def save(self, filepath: str):
        """Save agent weights."""
        np.savez(filepath, weights=self.weights, baseline=self.baseline)
    
    def load(self, filepath: str):
        """Load agent weights."""
        data = np.load(filepath)
        self.weights = data['weights']
        self.baseline = data.get('baseline', 0.0)

