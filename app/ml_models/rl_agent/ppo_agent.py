"""
PPO Agent for Adaptive Productivity Optimization
Integrates with AbilityClassifier for intelligent action selection
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from ..classifiers.ability_classifier import AbilityClassifier
except ImportError:
    # Fallback if import fails
    AbilityClassifier = None

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
    """
    PPO agent for productivity optimization
    
    Integrates with AbilityClassifier to:
    - Filter actions based on user commands/context
    - Use classifier confidence scores to guide policy
    - Align action space with predefined abilities
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
        use_classifier: bool = True,
        classifier_model_path: str = "bert-base-uncased"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.use_classifier = use_classifier and AbilityClassifier is not None
        
        # Initialize ability classifier if available
        if self.use_classifier:
            try:
                self.classifier = AbilityClassifier(
                    model_path=classifier_model_path,
                    num_abilities=action_dim
                )
                # Load ability map from classifier
                self.action_map = self._load_action_map_from_classifier()
            except Exception as e:
                print(f"Warning: Could not initialize AbilityClassifier: {e}")
                self.use_classifier = False
                self.classifier = None
                self.action_map = self._create_action_map(action_dim)
        else:
            self.classifier = None
            self.action_map = self._create_action_map(action_dim)
        
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
        
        # Action filtering cache
        self._action_filter_cache = {}
    
    def _load_action_map_from_classifier(self) -> Dict[int, str]:
        """Load action map from AbilityClassifier"""
        if self.classifier and hasattr(self.classifier, 'ability_map'):
            # Reverse the classifier's ability_map (name -> id) to (id -> name)
            reverse_map = {v: k for k, v in self.classifier.ability_map.items()}
            return reverse_map
        return self._create_action_map(50)
    
    def _create_action_map(self, action_dim: int) -> Dict[int, str]:
        """Create default mapping from action indices to ability names"""
        default_abilities = [
            "suggest_objective", "activate_focus_boost", "activate_velocity_boost",
            "schedule_break", "prioritize_tasks", "delegate_task",
            "summarize_text", "create_objective", "generate_code_review",
            "refactor_suggest", "documentation_gen", "test_generation",
            "commit_message_gen", "design_critique", "color_palette_gen",
            "layout_suggest", "export_assets", "design_system_check",
            "feature_breakdown", "user_story_gen", "sprint_planning",
            "roadmap_suggest", "stakeholder_update", "copy_gen",
            "campaign_suggest", "audience_analysis", "content_calendar",
            "seo_optimize", "debug_assist", "performance_optimize",
            "security_audit", "dependency_update", "migration_assist",
            "api_documentation", "error_handling", "logging_setup",
            "test_coverage", "code_quality", "architecture_review",
            "database_optimize", "cache_strategy", "load_balancing",
            "monitoring_setup", "deployment_automation", "ci_cd_setup",
            "containerization", "scaling_strategy", "backup_strategy"
        ]
        
        # Create mapping for available actions
        action_map = {}
        for i in range(min(action_dim, len(default_abilities))):
            action_map[i] = default_abilities[i]
        
        # Fill remaining with generic names
        for i in range(len(default_abilities), action_dim):
            action_map[i] = f"ability_{i}"
        
        return action_map
    
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
    
    def select_action(
        self, 
        user_data: Dict, 
        user_command: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> Tuple[int, str, float]:
        """
        Select action based on current policy
        
        Args:
            user_data: User state data
            user_command: Optional user command/request (used with classifier)
            context: Optional context dict (used with classifier)
        
        Returns:
            Tuple of (action_idx, action_name, confidence)
        """
        state = self.encode_state(user_data)
        
        # Get policy action probabilities
        with torch.no_grad():
            action_probs, value = self.policy.forward(state.unsqueeze(0))
            action_probs = action_probs.squeeze(0)
        
        # If classifier is available and user command provided, filter/boost actions
        if self.use_classifier and user_command:
            action_probs = self._apply_classifier_filter(
                action_probs, 
                user_command, 
                user_data.get('user_role'),
                context
            )
        
        # Sample from filtered distribution
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        action_idx = action.item()
        action_name = self.action_map.get(action_idx, "unknown")
        confidence = action_probs[action_idx].item()
        
        return action_idx, action_name, confidence
    
    def _apply_classifier_filter(
        self,
        action_probs: torch.Tensor,
        user_command: str,
        user_role: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Apply ability classifier to filter/boost action probabilities
        
        Uses classifier to identify relevant abilities and adjusts probabilities
        """
        try:
            # Get classifier predictions
            if context:
                classification = self.classifier.classify_with_context(
                    user_command, 
                    context, 
                    user_role
                )
                top_abilities = [(classification.get('ability'), classification.get('confidence', 0.0))]
            else:
                top_abilities = self.classifier.classify(user_command, user_role, top_k=5)
            
            # Create boost mask for relevant abilities
            boost_mask = torch.ones_like(action_probs) * 0.1  # Base probability boost
            total_boost = 0.0
            
            for ability_name, ability_confidence in top_abilities:
                # Find action index for this ability
                action_idx = None
                for idx, name in self.action_map.items():
                    if name == ability_name or ability_name in name:
                        action_idx = idx
                        break
                
                if action_idx is not None:
                    # Boost probability based on classifier confidence
                    boost = ability_confidence * 0.5  # Scale classifier confidence
                    boost_mask[action_idx] += boost
                    total_boost += boost
            
            # Normalize: redistribute probability mass
            if total_boost > 0:
                # Apply boost
                boosted_probs = action_probs * (1.0 + boost_mask)
                # Renormalize
                boosted_probs = boosted_probs / boosted_probs.sum()
                return boosted_probs
            
            return action_probs
            
        except Exception as e:
            print(f"Warning: Classifier filter failed: {e}")
            return action_probs
    
    def recommend_action(
        self, 
        user_data: Dict,
        user_command: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Get recommendation with reasoning
        
        Args:
            user_data: User state data
            user_command: Optional user command/request
            context: Optional context dict
        
        Returns:
            Dict with recommendation details
        """
        action_idx, action_name, confidence = self.select_action(
            user_data, 
            user_command, 
            context
        )
        
        # Generate reasoning based on state and command
        reasoning = self._generate_reasoning(user_data, action_name, user_command)
        
        # Estimate expected benefit
        expected_benefit = self._estimate_benefit(user_data, action_name)
        
        # Get classifier insights if available
        classifier_insights = None
        if self.use_classifier and user_command:
            try:
                if context:
                    classification = self.classifier.classify_with_context(
                        user_command,
                        context,
                        user_data.get('user_role')
                    )
                    classifier_insights = {
                        "detected_ability": classification.get('ability'),
                        "classifier_confidence": classification.get('confidence'),
                        "extracted_parameters": classification.get('parameters', {})
                    }
                else:
                    predictions = self.classifier.classify(
                        user_command,
                        user_data.get('user_role'),
                        top_k=3
                    )
                    classifier_insights = {
                        "top_predictions": predictions,
                        "matched_action": action_name in [pred[0] for pred in predictions]
                    }
            except Exception as e:
                print(f"Warning: Could not get classifier insights: {e}")
        
        result = {
            "action_type": self._get_action_type(action_name),
            "ability": action_name,
            "confidence": confidence,
            "reasoning": reasoning,
            "expected_benefit": expected_benefit
        }
        
        if classifier_insights:
            result["classifier_insights"] = classifier_insights
        
        return result
    
    def _generate_reasoning(
        self, 
        user_data: Dict, 
        action_name: str, 
        user_command: Optional[str] = None
    ) -> str:
        """Generate human-readable reasoning"""
        momentum = user_data.get('momentum', 0)
        efficiency = user_data.get('recent_efficiency', 0)
        streak = user_data.get('daily_streak', 0)
        user_role = user_data.get('user_role', 'general')
        
        # Base reasoning on action type
        if action_name == "activate_focus_boost":
            if efficiency > 0.8:
                return "You're in high-efficiency mode. A focus boost now could maximize your productivity."
            else:
                return "Your efficiency has room for improvement. A focus boost can help you concentrate."
        
        elif action_name == "schedule_break":
            return "You've been working consistently. A break will help maintain your healthy streak."
        
        elif action_name == "suggest_objective" or action_name == "create_objective":
            return f"You have {momentum} momentum. Creating a new objective can help channel this energy."
        
        # Role-specific reasoning
        if user_role == "software_engineer":
            if "code" in action_name or "refactor" in action_name:
                return f"Based on your current state and role, {action_name} aligns with your software engineering workflow."
        
        # Command-aware reasoning
        if user_command:
            return f"Based on your request '{user_command[:50]}...' and current state, {action_name} is recommended."
        
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
    # Initialize agent with classifier integration
    agent = PPOAgent(use_classifier=True)
    
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
        'recent_efficiency': 0.82,
        'user_role': 'software_engineer'
    }
    
    # Test 1: Recommendation without command
    print("=== Test 1: Recommendation without user command ===")
    recommendation = agent.recommend_action(user_data)
    print(f"Recommendation: {recommendation}\n")
    
    # Test 2: Recommendation with user command (uses classifier)
    print("=== Test 2: Recommendation with user command ===")
    user_command = "Can you help me review this code for security issues?"
    context = {
        'file_path': 'src/auth.py',
        'project': 'backend-service'
    }
    recommendation_with_command = agent.recommend_action(
        user_data, 
        user_command=user_command,
        context=context
    )
    print(f"User Command: {user_command}")
    print(f"Recommendation: {recommendation_with_command}\n")
    
    # Test 3: Action selection with classifier filtering
    print("=== Test 3: Action selection with classifier ===")
    action_idx, action_name, confidence = agent.select_action(
        user_data,
        user_command="I need to refactor this codebase"
    )
    print(f"Selected Action: {action_name} (idx: {action_idx}, confidence: {confidence:.3f})")

