"""
PPO Agent for Adaptive Productivity Optimization
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple
from collections import deque

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        shared_features = self.shared(state)
        return self.actor(shared_features), self.critic(shared_features)
    
    def get_action(self, state):
        """Sample action from policy"""
        action_probs, _ = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()
    
    def evaluate(self, state, action):
        """Evaluate action for training"""
        action_probs, value = self.forward(state)
        dist = Categorical(action_probs)
        action_log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action_log_prob, value, entropy


class PPOAgent:
    """PPO agent for productivity optimization"""
    def __init__(
        self,
        state_dim: int = 128,
        action_dim: int = 50,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Networks
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Experience buffer
        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': [],
            'values': []
        }
        
        # Action mapping
        self.action_map = self._create_action_map(action_dim)
    
    def _create_action_map(self, action_dim: int) -> Dict[int, str]:
        """Create mapping from action indices to ability names"""
        return {
            0: "suggest_objective",
            1: "activate_focus_boost",
            2: "activate_velocity_boost",
            3: "schedule_break",
            4: "prioritize_tasks",
            5: "delegate_task",
            # ... add all actions
        }
    
    def encode_state(self, user_data: Dict) -> torch.Tensor:
        """Encode user state into embedding"""
        # Normalize features
        features = [
            user_data.get('momentum', 0) / 1000.0,  # Normalized momentum
            user_data.get('current_level', 1) / 100.0,  # Normalized level
            user_data.get('task_completion_rate', 0),  # 0-1
            user_data.get('daily_streak', 0) / 365.0,  # Normalized streak
            self._encode_time_of_day(user_data.get('time_of_day', 'afternoon')),
            user_data.get('work_intensity', 0.5),  # 0-1
            user_data.get('stress_level', 0.3),  # 0-1
            len(user_data.get('active_tasks', [])) / 20.0,  # Normalized task count
            len(user_data.get('active_boosts', [])) / 5.0,  # Normalized boost count
            user_data.get('recent_efficiency', 0.7),  # 0-1
            # Add more features as needed (up to state_dim)
        ]
        
        # Pad to state_dim
        while len(features) < 128:
            features.append(0.0)
        
        return torch.tensor(features[:128], dtype=torch.float32).to(self.device)
    
    def _encode_time_of_day(self, time_of_day: str) -> float:
        """Encode time of day as continuous value"""
        encoding = {
            'morning': 0.25,
            'afternoon': 0.5,
            'evening': 0.75,
            'night': 1.0
        }
        return encoding.get(time_of_day, 0.5)
    
    def select_action(self, user_data: Dict) -> Tuple[int, str, float]:
        """Select action based on current policy"""
        state = self.encode_state(user_data)
        
        with torch.no_grad():
            action, log_prob, _ = self.policy.get_action(state.unsqueeze(0))
        
        action_idx = action.item()
        action_name = self.action_map.get(action_idx, "unknown")
        confidence = torch.exp(log_prob).item()
        
        return action_idx, action_name, confidence
    
    def recommend_action(self, user_data: Dict) -> Dict:
        """Get recommendation with reasoning"""
        action_idx, action_name, confidence = self.select_action(user_data)
        
        # Generate reasoning based on state
        reasoning = self._generate_reasoning(user_data, action_name)
        
        # Estimate expected benefit
        expected_benefit = self._estimate_benefit(user_data, action_name)
        
        return {
            "action_type": self._get_action_type(action_name),
            "ability": action_name,
            "confidence": confidence,
            "reasoning": reasoning,
            "expected_benefit": expected_benefit
        }
    
    def _generate_reasoning(self, user_data: Dict, action_name: str) -> str:
        """Generate human-readable reasoning"""
        momentum = user_data.get('momentum', 0)
        efficiency = user_data.get('recent_efficiency', 0)
        streak = user_data.get('daily_streak', 0)
        
        if action_name == "activate_focus_boost":
            if efficiency > 0.8:
                return "You're in high-efficiency mode. A focus boost now could maximize your productivity."
            else:
                return "Your efficiency has room for improvement. A focus boost can help you concentrate."
        
        elif action_name == "schedule_break":
            return "You've been working consistently. A break will help maintain your healthy streak."
        
        elif action_name == "suggest_objective":
            return f"You have {momentum} momentum. Creating a new objective can help channel this energy."
        
        return f"Based on your current state, {action_name} is recommended."
    
    def _get_action_type(self, action_name: str) -> str:
        """Determine action type"""
        if 'boost' in action_name:
            return 'activate_boost'
        elif 'suggest' in action_name or 'create' in action_name:
            return 'suggest_objective'
        elif 'schedule' in action_name:
            return 'schedule'
        return 'other'
    
    def _estimate_benefit(self, user_data: Dict, action_name: str) -> Dict:
        """Estimate expected benefit of action"""
        # Simple heuristic-based estimation
        # In production, this would use learned value function
        
        return {
            "momentum_gain": 25,
            "time_saved": 30,
            "efficiency_boost": 0.15,
            "satisfaction_increase": 0.2
        }
    
    def compute_reward(self, outcome: Dict) -> float:
        """
        Calculate reward based on outcome
        
        Args:
            outcome: Dict containing:
                - task_completed: bool
                - efficiency: float (0-1)
                - user_rating: int (1-5)
                - time_saved: int (minutes)
                - momentum_gained: int
                - user_frustrated: bool
        
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Task completion reward
        if outcome.get('task_completed', False):
            base_reward = 10.0
            efficiency = outcome.get('efficiency', 0.5)
            reward += base_reward * efficiency
        
        # User satisfaction (explicit feedback)
        user_rating = outcome.get('user_rating', 3)
        reward += (user_rating - 3) * 5.0  # Centered at 3
        
        # Time efficiency
        time_saved = outcome.get('time_saved', 0)
        if time_saved > 0:
            reward += time_saved * 0.5
        
        # Momentum growth
        momentum_gained = outcome.get('momentum_gained', 0)
        reward += momentum_gained * 0.1
        
        # Penalties
        if outcome.get('user_frustrated', False):
            reward -= 20.0
        
        if outcome.get('ability_not_used', False):
            reward -= 10.0
        
        return reward
    
    def store_transition(
        self,
        state: torch.Tensor,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float
    ):
        """Store transition in memory"""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['log_probs'].append(log_prob)
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)
        self.memory['values'].append(value)
    
    def update(self, epochs: int = 10, batch_size: int = 64) -> Dict:
        """Update policy using PPO"""
        # Convert memory to tensors
        states = torch.stack(self.memory['states'])
        actions = torch.tensor(self.memory['actions'])
        old_log_probs = torch.tensor(self.memory['log_probs'])
        rewards = torch.tensor(self.memory['rewards'])
        dones = torch.tensor(self.memory['dones'])
        old_values = torch.tensor(self.memory['values'])
        
        # Compute returns and advantages
        returns = self._compute_returns(rewards, dones)
        advantages = returns - old_values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_loss = 0
        for _ in range(epochs):
            # Evaluate actions with current policy
            new_log_probs, values, entropy = self.policy.evaluate(states, actions)
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Compute surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            
            # Compute losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), returns)
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Clear memory
        for key in self.memory:
            self.memory[key] = []
        
        return {
            "avg_loss": total_loss / epochs,
            "avg_return": returns.mean().item()
        }
    
    def _compute_returns(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Compute discounted returns"""
        returns = []
        R = 0
        
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        return torch.tensor(returns)
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# Example usage
if __name__ == "__main__":
    agent = PPOAgent()
    
    # Example user data
    user_data = {
        'momentum': 450,
        'current_level': 15,
        'task_completion_rate': 0.85,
        'daily_streak': 7,
        'time_of_day': 'afternoon',
        'work_intensity': 0.7,
        'stress_level': 0.3,
        'active_tasks': [1, 2, 3],
        'active_boosts': [],
        'recent_efficiency': 0.82
    }
    
    # Get recommendation
    recommendation = agent.recommend_action(user_data)
    print(f"Recommendation: {recommendation}")

