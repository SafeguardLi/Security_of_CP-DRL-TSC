import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# PPOActor is now a standard PyTorch nn.Module
class PPOActor(nn.Module):
    def __init__(self, input_d, hidden_d, output_d, hidden_act=nn.ReLU, lr=1e-4, beta_initial=0.01):
        super(PPOActor, self).__init__()
        self.output_d = output_d
        
        # 1. Use nn.Parameter for the mutable beta (easily saved/loaded with state_dict)
        self.beta = nn.Parameter(torch.tensor(beta_initial, dtype=torch.float32), requires_grad=False)
        
        # Build the multilayer perceptron (MLP)
        layers = []
        prev_d = input_d
        for h_d in hidden_d:
            layers.append(nn.Linear(prev_d, h_d))
            layers.append(hidden_act())
            prev_d = h_d
            
        # Final layer outputs logits for the softmax distribution
        layers.append(nn.Linear(prev_d, output_d))
        self.model = nn.Sequential(*layers)

        # Create the optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Initialize weights using He initialization (similar to 'he_uniform' in Keras)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self.apply(init_weights)

    def forward(self, state):
        # state is expected to be a torch.tensor
        logits = self.model(state)
        # Output the probability distribution (softmax)
        action_probs = F.softmax(logits, dim=-1)
        return action_probs

    # --- MODIFIED: Added s_effs and eta to the method signature ---
    def calculate_loss(self, states, actions, old_probs, advantages, s_effs, eta, epsilon=0.2):
        """Calculates PPO Actor Loss by calling the external loss function."""
        mu = self.forward(states)
        # The loss function now also returns entropy for logging purposes
        # Pass the new parameters for loss shaping to the external function
        actor_loss, _ = ppo_actor_loss(mu, old_probs, advantages, actions, epsilon, self.get_beta(), s_effs, eta)
        return actor_loss

    # --- Compatibility and Save/Load Methods ---

    def set_beta(self, new_beta):
        """Sets the new value for the dynamic entropy coefficient (beta)."""
        if isinstance(new_beta, (int, float, np.float32)):
            self.beta.data.fill_(new_beta)

    def get_beta(self):
        """Retrieves the current value of the dynamic entropy coefficient (beta) as a Python float."""
        return self.beta.item()

    def get_weights(self, nettype=None):
        # PyTorch equivalent to get weights
        return self.state_dict()

    def set_weights(self, weights, nettype=None):
        # PyTorch equivalent to set weights
        self.load_state_dict(weights)

    def save_weights(self, nettype, path, fname):
        # PyTorch save method
        if not fname.endswith('.pt'):
            fname += '.pt'
        filepath = os.path.join(path, fname)
        torch.save(self.state_dict(), filepath)

    def load_weights(self, path):
        # PyTorch load method
        if not path.endswith('.pt'):
            path += '.pt'
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            print(f"Successfully loaded weights from {path}")
        else:
            print(f"Warning: Weight file not found at {path}")


# PPO Actor Loss is now calculated as a simple Python function
def ppo_actor_loss(mu, old_probs, advantages, actions, epsilon, beta, s_effs, eta):
    """
    Calculates PPO Actor Loss with Entropy Bonus and Inefficiency Penalty using PyTorch.
    
    New Args:
        s_effs (torch.Tensor): Tensor of effectiveness scores (0.0 for failure, 1.0 for success).
        eta (float): The Inefficiency Coefficient hyperparameter.
    """
    
    # 1. Calculate Log Probabilities
    # Clamp mu to avoid log(0)
    mu = mu.clamp(1e-10, 1.0)
    log_probs = torch.log(mu)
    
    # 2. Get Selected Action Log Prob and Old Prob
    # actions is a tensor of indices [batch_size]
    selected_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
    
    # old_probs is a tensor of scalar probabilities [batch_size], requires log conversion
    old_log_probs = torch.log(old_probs.clamp(1e-10, 1.0))

    # 3. Ratio: pi_new(a|s) / pi_old(a|s)
    # log(A/B) = log(A) - log(B) -> exp(log(A/B)) = A/B
    ratios = torch.exp(selected_log_probs - old_log_probs)

    # 4. Surrogate Loss (L_CLIP)
    surrogate_loss_1 = ratios * advantages
    clipped_ratios = torch.clamp(ratios, 1.0 - epsilon, 1.0 + epsilon)
    surrogate_loss_2 = clipped_ratios * advantages

    # Take the minimum of the two surrogate losses (PPO-Clip objective)
    surrogate_loss = torch.min(surrogate_loss_1, surrogate_loss_2)

    # 5. Entropy Bonus (H)
    # Entropy = -sum(p * log(p))
    entropy = -(mu * log_probs).sum(dim=-1).mean()

    # 6. --- NEW: Inefficiency Loss Penalty (L_ineff) ---
    # This term penalizes the log-probability of an action if it was ineffective (s_eff=0).
    inefficiency_loss = (1.0 - s_effs) * selected_log_probs

    # 7. --- MODIFIED: Final Combined PPO Loss ---
    # We add the scaled inefficiency term to the final loss.
    # Since we minimize the loss, this term pushes the probability of
    # ineffective actions towards zero.
    actor_loss = -surrogate_loss.mean() - beta * entropy - eta * inefficiency_loss.mean()
    
    return actor_loss, entropy.item()