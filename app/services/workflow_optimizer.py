"""
Workflow Optimizer
Reinforcement Learning agent for adaptive productivity optimization
Tier 3: Adaptive, Long-Term Learning
Uses PPO (Proximal Policy Optimization) for sequential decision-making
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from ..logging_config import get_logger
from ..settings import settings

logger = get_logger("cyrex.workflow_optimizer")


class WorkflowActorCritic(nn.Module):
    """Actor-Critic network for PPO agent"""
    
    def __init__(self, state_dim: int = 128, action_dim: int = 50, hidden_dim: int = 256):
        super(WorkflowActorCritic, self).__init__()
        
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        shared_features = self.shared(state)
        action_probs = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_probs, value
    
    def get_action(self, state):
        """Sample action from policy"""
        action_probs, value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
    
    def evaluate(self, state, action):
        """Evaluate action for training"""
        action_probs, value = self.forward(state)
        dist = Categorical(action_probs)
        action_log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action_log_prob, value, entropy


class WorkflowOptimizer:
    """
    Reinforcement Learning agent that learns optimal productivity strategies.
    Uses PPO to learn which abilities/actions maximize long-term user productivity.
    """
    
    def __init__(
        self,
        state_dim: int = 128,
        action_dim: int = 50,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        model_path: Optional[str] = None
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Networks
        self.policy = WorkflowActorCritic(state_dim, action_dim).to(self.device)
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
        self.action_registry = self._create_action_registry(action_dim)
        
        # Load model if path provided
        if model_path:
            self.load(model_path)
        
        logger.info(f"Workflow optimizer initialized: {state_dim}D state, {action_dim} actions")
    
    def _create_action_registry(self, action_dim: int) -> Dict[int, Dict]:
        """Create mapping from action indices to action definitions"""
        return {
            0: {"id": "suggest_objective", "type": "suggest", "category": "gamification"},
            1: {"id": "activate_focus_boost", "type": "boost", "category": "boost"},
            2: {"id": "activate_velocity_boost", "type": "boost", "category": "boost"},
            3: {"id": "schedule_break", "type": "wellness", "category": "wellness"},
            4: {"id": "prioritize_tasks", "type": "organize", "category": "productivity"},
            5: {"id": "suggest_odyssey", "type": "suggest", "category": "gamification"},
            6: {"id": "cash_in_streak", "type": "reward", "category": "gamification"},
            7: {"id": "suggest_collaboration", "type": "social", "category": "collaboration"},
            # ... add all actions
        }
    
    def encode_state(self, user_data: Dict) -> torch.Tensor:
        """Encode user state into embedding vector"""
        features = []
        
        # Normalized momentum (0-1)
        momentum = user_data.get('momentum', 0)
        features.append(min(momentum / 1000.0, 1.0))
        
        # Normalized level (0-1)
        level = user_data.get('current_level', 1)
        features.append(min(level / 100.0, 1.0))
        
        # Task completion rate (0-1)
        features.append(user_data.get('task_completion_rate', 0.5))
        
        # Normalized streak (0-1)
        daily_streak = user_data.get('daily_streak', 0)
        features.append(min(daily_streak / 365.0, 1.0))
        
        # Time of day encoding (0-1)
        time_of_day = user_data.get('time_of_day', 'afternoon')
        time_encoding = {
            'morning': 0.25,
            'afternoon': 0.5,
            'evening': 0.75,
            'night': 1.0
        }
        features.append(time_encoding.get(time_of_day, 0.5))
        
        # Work intensity (0-1)
        features.append(user_data.get('work_intensity', 0.5))
        
        # Stress level (0-1)
        features.append(user_data.get('stress_level', 0.3))
        
        # Normalized active tasks (0-1)
        active_tasks = user_data.get('active_tasks', [])
        features.append(min(len(active_tasks) / 20.0, 1.0))
        
        # Normalized active boosts (0-1)
        active_boosts = user_data.get('active_boosts', [])
        features.append(min(len(active_boosts) / 5.0, 1.0))
        
        # Recent efficiency (0-1)
        features.append(user_data.get('recent_efficiency', 0.7))
        
        # Momentum growth rate (0-1)
        momentum_growth = user_data.get('momentum_growth_rate', 0.0)
        features.append(min(abs(momentum_growth), 1.0))
        
        # Pad to state_dim
        while len(features) < 128:
            features.append(0.0)
        
        return torch.tensor(features[:128], dtype=torch.float32).to(self.device)
    
    def recommend_action(self, user_data: Dict) -> Dict:
        """
        Get RL agent's recommended action with reasoning
        
        Args:
            user_data: User's current state
        
        Returns:
            Recommendation with action, confidence, reasoning, and expected benefit
        """
        state = self.encode_state(user_data)
        
        with torch.no_grad():
            action, log_prob, _, value = self.policy.get_action(state.unsqueeze(0))
        
        action_idx = action.item()
        action_def = self.action_registry.get(action_idx, {"id": "unknown", "type": "unknown"})
        confidence = torch.exp(log_prob).item()
        state_value = value.item()
        
        # Generate reasoning
        reasoning = self._generate_reasoning(user_data, action_def)
        
        # Estimate expected benefit
        expected_benefit = self._estimate_benefit(user_data, action_def)
        
        return {
            "action_type": action_def.get("type", "unknown"),
            "ability_id": action_def.get("id", "unknown"),
            "category": action_def.get("category", "unknown"),
            "confidence": confidence,
            "state_value": state_value,
            "reasoning": reasoning,
            "expected_benefit": expected_benefit
        }
    
    def _generate_reasoning(self, user_data: Dict, action_def: Dict) -> str:
        """Generate human-readable reasoning for recommendation"""
        momentum = user_data.get('momentum', 0)
        efficiency = user_data.get('recent_efficiency', 0.7)
        streak = user_data.get('daily_streak', 0)
        active_boosts = user_data.get('active_boosts', [])
        
        action_id = action_def.get("id", "")
        
        if action_id == "activate_focus_boost":
            if efficiency > 0.8:
                return "You're in high-efficiency mode. A focus boost now could maximize your productivity and help you complete 2-3 more tasks."
            elif efficiency < 0.5:
                return "Your efficiency has room for improvement. A focus boost can help you concentrate and get back on track."
            else:
                return "A focus boost can help you maintain your current momentum and avoid distractions."
        
        elif action_id == "schedule_break":
            if streak >= 7:
                return f"You've maintained a {streak}-day streak! A break will help you maintain this healthy pattern and prevent burnout."
            else:
                return "You've been working consistently. A scheduled break will help you recharge and maintain productivity."
        
        elif action_id == "suggest_objective":
            return f"You have {momentum} momentum. Creating a new objective can help channel this energy into meaningful progress."
        
        elif action_id == "cash_in_streak":
            return f"Your {streak}-day streak can be cashed in for boost credits, giving you more flexibility in your workflow."
        
        elif action_id == "prioritize_tasks":
            active_tasks = user_data.get('active_tasks', [])
            return f"You have {len(active_tasks)} active tasks. Prioritizing them can help you focus on high-impact work."
        
        return f"Based on your current productivity state, {action_id} is recommended to optimize your workflow."
    
    def _estimate_benefit(self, user_data: Dict, action_def: Dict) -> Dict:
        """Estimate expected benefit of action (simplified - would use learned value function in production)"""
        # In production, this would use the learned value function
        # For now, use heuristics based on action type
        
        action_id = action_def.get("id", "")
        
        base_benefits = {
            "activate_focus_boost": {
                "momentum_gain": 25,
                "time_saved": 30,
                "efficiency_boost": 0.15,
                "satisfaction_increase": 0.2
            },
            "suggest_objective": {
                "momentum_gain": 15,
                "time_saved": 0,
                "efficiency_boost": 0.05,
                "satisfaction_increase": 0.1
            },
            "schedule_break": {
                "momentum_gain": 5,
                "time_saved": -15,  # Break time
                "efficiency_boost": 0.1,
                "satisfaction_increase": 0.3
            }
        }
        
        return base_benefits.get(action_id, {
            "momentum_gain": 10,
            "time_saved": 15,
            "efficiency_boost": 0.05,
            "satisfaction_increase": 0.1
        })
    
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
                - ability_used: bool
        
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
        """Store transition in experience buffer"""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['log_probs'].append(log_prob)
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)
        self.memory['values'].append(value)
    
    def update(self, epochs: int = 10, batch_size: int = 64) -> Dict:
        """Update policy using PPO algorithm"""
        if len(self.memory['states']) < batch_size:
            return {"status": "insufficient_data", "buffer_size": len(self.memory['states'])}
        
        # Convert memory to tensors
        states = torch.stack(self.memory['states'])
        actions = torch.tensor(self.memory['actions']).to(self.device)
        old_log_probs = torch.tensor(self.memory['log_probs']).to(self.device)
        rewards = torch.tensor(self.memory['rewards']).to(self.device)
        dones = torch.tensor(self.memory['dones']).to(self.device)
        old_values = torch.tensor(self.memory['values']).to(self.device)
        
        # Compute returns and advantages
        returns = self._compute_returns(rewards, dones)
        advantages = returns - old_values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_loss = 0
        for epoch in range(epochs):
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
            "status": "updated",
            "avg_loss": total_loss / epochs,
            "avg_return": returns.mean().item(),
            "avg_advantage": advantages.mean().item()
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
        
        return torch.tensor(returns, dtype=torch.float32).to(self.device)
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': {
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef
            }
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")


# Singleton instance
_workflow_optimizer = None

def get_workflow_optimizer() -> WorkflowOptimizer:
    """Get singleton WorkflowOptimizer instance"""
    global _workflow_optimizer
    if _workflow_optimizer is None:
        model_path = getattr(settings, 'PRODUCTIVITY_AGENT_MODEL_PATH', None)
        _workflow_optimizer = WorkflowOptimizer(model_path=model_path)
    return _workflow_optimizer

